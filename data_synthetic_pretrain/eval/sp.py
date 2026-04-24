"""ShortestPath evaluation implementation."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import wandb

from apps.main.transformer import LMTransformer, LMTransformerArgs
from data_synthetic_pretrain.shared_vocab import split_into_words, vocab_layout
from lingua.args import dataclass_from_dict, dump_config
from lingua.checkpoint import CONSOLIDATE_FOLDER, CONSOLIDATE_NAME, consolidate_checkpoints
from lingua.distributed import DistributedArgs, get_global_rank, setup_torch_distributed
from lingua.logger import init_logger
from lingua.tokenizer import MockTokenizer

logger = logging.getLogger(__name__)

_NUM_LOG_EXAMPLES = 5


@dataclass
class SPEvalArgs:
    ckpt_dir: str = ""
    eval_dir: str = ""
    max_examples: Optional[int] = None
    temperature: float = 0.0
    max_gen_len: int = 128
    base_vocab_size: int = 4
    min_token_length: int = 4
    max_token_length: int = 6
    dump_dir: str = ""
    global_step: Optional[int] = None
    metric_log_dir: Optional[str] = None
    wandb: Optional[Dict] = None
    eval_max_seqlen: Optional[int] = 2048
    device: str = "cuda"
    max_answer_size: int = 6
    num_examples_generate: int = 500


def _load_model(consolidate_path: str, eval_max_seqlen: Optional[int]):
    consolidate_path = Path(consolidate_path)
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


class SPEvaluator:
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
        self._max_prompt_len = max(1, eval_max_seqlen - max_gen_len)
        self.eos = vocab_layout(base_vocab_size)["EOS"]
        self._split_into_words = split_into_words

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
        pred_words = self._split_into_words(predicted_raw, self.base_vocab_size)
        exp_words = self._split_into_words(expected_tokens, self.base_vocab_size)
        pred_path = [tuple(w) for w in pred_words]
        exp_path = [tuple(w) for w in exp_words]
        exact_match = pred_path == exp_path
        count_match = len(pred_path) == len(exp_path)
        endpoint_ok = len(pred_path) > 0 and len(exp_path) > 0 and pred_path[-1] == exp_path[-1]
        return {
            "skipped": False,
            "exact_path_accuracy": exact_match,
            "node_count_match": count_match,
            "endpoint_correct": endpoint_ok,
            "pred_len": len(pred_path),
            "expected_len": len(exp_path),
            "_prompt": prompt_tokens,
            "_predicted_raw": predicted_raw,
            "_expected_tokens": expected_tokens,
        }

    def evaluate_path_length(self, eval_file: str, path_length: int, max_examples: Optional[int] = None) -> Dict:
        examples = []
        with open(eval_file) as f:
            for i, line in enumerate(f):
                if max_examples is not None and i >= max_examples:
                    break
                examples.append(json.loads(line))
        n_exact = n_count = n_end = total_pred = total_exp = n_eval = n_skip = 0
        all_results: List[Dict] = []
        for ex in examples:
            res = self.evaluate_example(ex)
            all_results.append(res)
            if res["skipped"]:
                n_skip += 1
                continue
            n_eval += 1
            if res["exact_path_accuracy"]:
                n_exact += 1
            if res["node_count_match"]:
                n_count += 1
            if res["endpoint_correct"]:
                n_end += 1
            total_pred += res["pred_len"]
            total_exp += res["expected_len"]
        logger.info("\n%s", "=" * 70)
        logger.info("  SAMPLE EXAMPLES  path_length=%s", path_length)
        logger.info("%s", "=" * 70)
        logged = 0
        for res in all_results:
            if res.get("skipped") or logged >= _NUM_LOG_EXAMPLES:
                continue
            logged += 1
            correct = "CORRECT" if res["exact_path_accuracy"] else "WRONG"
            logger.info(
                "  %s  pred=%s nodes  exp=%s nodes\n    predicted : %s\n    expected  : %s",
                correct,
                res["pred_len"],
                res["expected_len"],
                res["_predicted_raw"][:30],
                res["_expected_tokens"][:30],
            )
        logger.info("%s", "=" * 70)
        denom = max(n_eval, 1)
        avg_exp = total_exp / denom
        return {
            "path_length": path_length,
            "n_evaluated": n_eval,
            "n_skipped": n_skip,
            "exact_path_accuracy": n_exact / denom,
            "node_count_match": n_count / denom,
            "endpoint_correct": n_end / denom,
            "length_ratio": (total_pred / denom) / max(avg_exp, 1),
        }


def evaluate_sp(cfg: SPEvalArgs):
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
    evaluator = SPEvaluator(
        model=model,
        base_vocab_size=cfg.base_vocab_size,
        temperature=cfg.temperature,
        max_gen_len=cfg.max_gen_len,
        eval_max_seqlen=cfg.eval_max_seqlen or 2048,
        device=cfg.device,
    )
    eval_files = sorted(Path(cfg.eval_dir).glob("sp.eval.path_*.jsonl"))
    if not eval_files:
        logger.warning("No eval files found in %s", cfg.eval_dir)
        return
    all_metrics: List[Dict] = []
    for eval_file in eval_files:
        suffix = eval_file.stem.split("path_")[-1]
        if suffix == "unreachable":
            pl = -1
        else:
            try:
                pl = int(suffix)
            except ValueError:
                continue
        metrics = evaluator.evaluate_path_length(str(eval_file), path_length=pl, max_examples=cfg.max_examples)
        all_metrics.append(metrics)
        tag = "unreachable" if pl == -1 else str(pl)
        if wandb.run is not None:
            wandb.log(
                {
                    f"eval/sp/path_{tag}/exact_path_accuracy": metrics["exact_path_accuracy"],
                    f"eval/sp/path_{tag}/node_count_match": metrics["node_count_match"],
                    f"eval/sp/path_{tag}/endpoint_correct": metrics["endpoint_correct"],
                    f"eval/sp/path_{tag}/length_ratio": metrics["length_ratio"],
                },
                step=cfg.global_step,
            )
    reachable = [m for m in all_metrics if m["path_length"] >= 0]
    if reachable and wandb.run is not None:
        avg_exact = sum(m["exact_path_accuracy"] for m in reachable) / len(reachable)
        wandb.log({"eval/sp/avg_exact_path_accuracy": avg_exact}, step=cfg.global_step)
    summary_path = Path(cfg.dump_dir) / "eval_results.json"
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)


__all__ = ["SPEvalArgs", "evaluate_sp"]
