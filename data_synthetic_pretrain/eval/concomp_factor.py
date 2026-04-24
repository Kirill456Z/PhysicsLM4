"""ConCompFactor evaluation implementation."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import wandb

from apps.main.transformer import LMTransformer, LMTransformerArgs
from data_synthetic_pretrain.shared_vocab import split_into_words
from lingua.args import dataclass_from_dict, dump_config
from lingua.checkpoint import CONSOLIDATE_FOLDER, CONSOLIDATE_NAME, consolidate_checkpoints
from lingua.distributed import DistributedArgs, get_global_rank, setup_torch_distributed
from lingua.logger import init_logger
from lingua.tokenizer import MockTokenizer

logger = logging.getLogger(__name__)


@dataclass
class ConCompFactorEvalArgs:
    ckpt_dir: str = ""
    eval_dir: str = ""
    max_examples: Optional[int] = None
    temperature: float = 0.0
    max_gen_len: int = 128
    base_vocab_size: int = 8
    min_token_length: int = 2
    max_token_length: int = 6
    dump_dir: str = ""
    global_step: Optional[int] = None
    metric_log_dir: Optional[str] = None
    wandb: Optional[Dict] = None
    eval_max_seqlen: Optional[int] = 2048
    device: str = "cuda"
    max_answer_size: int = 8
    num_examples_generate: int = 500


def _eos_token(base_vocab_size: int) -> int:
    return 2 * base_vocab_size + 4


def _decode_words(tokens: List[int], base_vocab_size: int) -> List[Tuple[int, ...]]:
    return [tuple(w) for w in split_into_words(tokens, base_vocab_size)]


def _load_model(ckpt_dir: str, eval_max_seqlen: Optional[int] = None):
    consolidate_path = Path(ckpt_dir)
    with open(consolidate_path / "params.json") as f:
        params = json.load(f)
    model_params = params.get("model", params)
    if eval_max_seqlen is not None:
        model_params["max_seqlen"] = eval_max_seqlen
    model_args = dataclass_from_dict(LMTransformerArgs, model_params)
    model = LMTransformer(model_args)
    ckpt = torch.load(consolidate_path / CONSOLIDATE_NAME, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt.get("model", ckpt), strict=True)
    model = model.cuda().eval()
    tokenizer = MockTokenizer()
    tokenizer.n_words = model_args.vocab_size
    return model, tokenizer


class ConCompFactorEvaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        base_vocab_size: int,
        temperature: float = 0.0,
        max_gen_len: int = 128,
        eval_max_seqlen: int = 2048,
        device: str = "cuda",
    ):
        self.model = model
        self.base_vocab_size = base_vocab_size
        self.temperature = temperature
        self.max_gen_len = max_gen_len
        self.device = device
        self.eos = _eos_token(base_vocab_size)
        self._max_prompt_len = max(1, eval_max_seqlen - max_gen_len)

    def generate_answer(self, prompt_tokens: List[int]) -> List[int]:
        tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=self.device).unsqueeze(0)
        generated: List[int] = []
        with torch.no_grad():
            for _ in range(self.max_gen_len):
                logits = self.model(tokens)
                if self.temperature == 0.0:
                    next_tok = int(logits[:, -1, :].argmax(dim=-1).item())
                else:
                    probs = torch.softmax(logits[:, -1, :] / self.temperature, dim=-1)
                    next_tok = int(torch.multinomial(probs, 1).item())
                if next_tok == self.eos:
                    break
                generated.append(next_tok)
                tokens = torch.cat(
                    [tokens, torch.tensor([[next_tok]], dtype=torch.long, device=self.device)],
                    dim=1,
                )
        return generated

    def evaluate_example(self, example: Dict) -> Dict:
        answer_start_idx = example.get("answer_start_idx", -1)
        if answer_start_idx <= 0:
            return {"skipped": True}
        prompt_tokens = example["text"][:answer_start_idx]
        if not prompt_tokens:
            return {"skipped": True}
        if len(prompt_tokens) > self._max_prompt_len:
            prompt_tokens = prompt_tokens[-self._max_prompt_len:]

        predicted_raw = self.generate_answer(prompt_tokens)
        expected_tokens = example.get("answer_tokens", [])

        pred_words = _decode_words(predicted_raw, self.base_vocab_size)
        exp_words = _decode_words(expected_tokens, self.base_vocab_size)
        pred_set = set(pred_words)
        exp_set = set(exp_words)

        return {
            "skipped": False,
            "representative_set_accuracy": pred_set == exp_set,
            "representative_count_match": len(pred_words) == len(exp_words),
            "exact_seq_match": predicted_raw == expected_tokens,
            "pred_len": len(pred_words),
            "expected_len": len(exp_words),
        }

    def evaluate_size(self, eval_file: str, answer_size: int, max_examples: Optional[int] = None) -> Dict:
        examples = []
        with open(eval_file) as f:
            for i, line in enumerate(f):
                if max_examples is not None and i >= max_examples:
                    break
                examples.append(json.loads(line))

        n_set = n_count = n_seq = total_pred = total_exp = n_eval = n_skip = 0
        for ex in examples:
            res = self.evaluate_example(ex)
            if res["skipped"]:
                n_skip += 1
                continue
            n_eval += 1
            if res["representative_set_accuracy"]:
                n_set += 1
            if res["representative_count_match"]:
                n_count += 1
            if res["exact_seq_match"]:
                n_seq += 1
            total_pred += res["pred_len"]
            total_exp += res["expected_len"]

        denom = max(n_eval, 1)
        avg_exp = total_exp / denom
        return {
            "answer_size": answer_size,
            "n_evaluated": n_eval,
            "n_skipped": n_skip,
            "representative_set_accuracy": n_set / denom,
            "representative_count_match": n_count / denom,
            "exact_seq_match": n_seq / denom,
            "length_ratio": (total_pred / denom) / max(avg_exp, 1),
        }


def evaluate_concomp_factor(cfg: ConCompFactorEvalArgs):
    if not torch.distributed.is_initialized():
        setup_torch_distributed(DistributedArgs())
    if get_global_rank() != 0:
        return
    Path(cfg.dump_dir).mkdir(parents=True, exist_ok=True)
    init_logger(Path(cfg.dump_dir) / "eval.log")
    dump_config(cfg, Path(cfg.dump_dir) / "config.yaml", log_config=False)

    ckpt_path = Path(cfg.ckpt_dir)
    if (
        ckpt_path.exists()
        and (ckpt_path / "params.json").exists()
        and next(ckpt_path.glob("*.pth"), None) is not None
    ):
        consolidate_path = ckpt_path
    else:
        consolidate_path = ckpt_path / CONSOLIDATE_FOLDER
        if not consolidate_path.exists():
            consolidate_path = consolidate_checkpoints(cfg.ckpt_dir)

    model, _ = _load_model(str(consolidate_path), cfg.eval_max_seqlen)
    evaluator = ConCompFactorEvaluator(
        model=model,
        base_vocab_size=cfg.base_vocab_size,
        temperature=cfg.temperature,
        max_gen_len=cfg.max_gen_len,
        eval_max_seqlen=cfg.eval_max_seqlen or 2048,
        device=cfg.device,
    )

    eval_files = sorted(Path(cfg.eval_dir).glob("concomp_factor.eval.size_*.jsonl"))
    if not eval_files:
        logger.error("No evaluation files found in %s", cfg.eval_dir)
        return

    all_results: Dict[str, Dict] = {}
    for eval_file in eval_files:
        size = int(eval_file.stem.split("_")[-1])
        all_results[f"size_{size}"] = evaluator.evaluate_size(
            str(eval_file), answer_size=size, max_examples=cfg.max_examples
        )

    out_file = Path(cfg.dump_dir) / "concomp_factor_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)

    if wandb.run is not None:
        metrics = {}
        for key, res in all_results.items():
            metrics[f"eval/concomp_factor/{key}/representative_set_accuracy"] = res[
                "representative_set_accuracy"
            ]
            metrics[f"eval/concomp_factor/{key}/representative_count_match"] = res[
                "representative_count_match"
            ]
            metrics[f"eval/concomp_factor/{key}/exact_seq_match"] = res["exact_seq_match"]
            metrics[f"eval/concomp_factor/{key}/length_ratio"] = res["length_ratio"]
        avg_set = sum(
            r["representative_set_accuracy"] for r in all_results.values()
        ) / max(len(all_results), 1)
        metrics["eval/concomp_factor/avg_representative_set_accuracy"] = avg_set
        if cfg.global_step is not None:
            wandb.log(metrics, step=cfg.global_step)
        else:
            wandb.log(metrics)


__all__ = ["ConCompFactorEvalArgs", "evaluate_concomp_factor"]
