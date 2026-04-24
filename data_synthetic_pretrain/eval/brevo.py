"""Brevo evaluation implementation.

Brevo predicts the set of ancestors reachable from a query node in a DAG, in a
valid topological order, using the *unified* shared vocabulary (multi-token
node words + special tokens from shared_vocab.py).
"""

import json
import logging
from collections import defaultdict
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


@dataclass
class BrevoEvalArgs:
    ckpt_dir: str = ""
    eval_dir: str = ""
    max_examples: Optional[int] = None
    temperature: float = 0.0
    max_gen_len: int = 256
    max_nodes: int = 110
    multi: bool = True
    dump_dir: str = ""
    global_step: Optional[int] = None
    metric_log_dir: Optional[str] = None
    wandb: Optional[Dict] = None
    eval_max_seqlen: Optional[int] = 2048
    device: str = "cuda"
    max_depth: int = 8
    num_examples_generate: int = 500


def _parse_tokens_multi(tokens: List[int], base_vocab_size: int = 4):
    """Standalone Brevo parser to avoid circular imports."""
    layout = vocab_layout(base_vocab_size)
    task_brevo = layout["TASK_BREVO"]
    brevo_query = layout["BREVO_QUERY"]
    brevo_ans = layout["BREVO_ANS"]
    eos = layout["EOS"]

    if not tokens or tokens[0] != task_brevo or tokens[-1] != eos:
        return False, None, None
    try:
        idx_query = tokens.index(brevo_query)
        idx_ans = tokens.index(brevo_ans)
    except ValueError:
        return False, None, None

    edge_words = split_into_words(tokens[1:idx_query], base_vocab_size)
    query_words = split_into_words(tokens[idx_query + 1 : idx_ans], base_vocab_size)
    answer_words = split_into_words(tokens[idx_ans + 1 : -1], base_vocab_size)
    if len(edge_words) % 2 != 0 or len(query_words) != 1:
        return False, None, None

    query_word = tuple(query_words[0])
    all_words = set(tuple(w) for w in edge_words)
    all_words.add(query_word)
    word_to_id = {w: i for i, w in enumerate(sorted(all_words))}

    dag = defaultdict(list)
    for i in range(0, len(edge_words), 2):
        p_id = word_to_id[tuple(edge_words[i])]
        c_id = word_to_id[tuple(edge_words[i + 1])]
        dag[c_id].append(p_id)

    query_id = word_to_id[query_word]
    topo_ids = [word_to_id[tuple(w)] for w in answer_words]

    reachable = set()
    stack = [query_id]
    while stack:
        node = stack.pop()
        reachable.add(node)
        for parent in dag.get(node, []):
            if parent not in reachable:
                stack.append(parent)

    if set(topo_ids) != reachable:
        return False, query_word, [tuple(w) for w in answer_words]

    seen = set()
    for node in topo_ids:
        for parent in dag.get(node, []):
            if parent not in seen:
                return False, query_word, [tuple(w) for w in answer_words]
        seen.add(node)

    return True, query_word, [tuple(w) for w in answer_words]


def load_consolidated_model(consolidate_path: str, eval_max_seqlen: Optional[int] = None):
    consolidate_path = Path(consolidate_path)
    with open(consolidate_path / "params.json") as f:
        params = json.load(f)
    model_params = params.get("model", params)
    if eval_max_seqlen is not None:
        model_params["max_seqlen"] = eval_max_seqlen
    model_args = dataclass_from_dict(LMTransformerArgs, model_params)
    model = LMTransformer(model_args)
    checkpoint = torch.load(consolidate_path / CONSOLIDATE_NAME, map_location="cpu", weights_only=True)
    model_state = checkpoint.get("model", checkpoint)
    model.load_state_dict(model_state, strict=True)
    model = model.cuda().eval()
    tokenizer = MockTokenizer()
    tokenizer.n_words = model_args.vocab_size
    return model, tokenizer


class BrevoEvaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        base_vocab_size: int,
        temperature: float = 0.0,
        max_gen_len: int = 256,
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
        self._parse_tokens = _parse_tokens_multi

        self._logged_examples = 0
        self._max_logged_examples = 8

    def generate_answer(self, prompt_tokens: List[int]) -> List[int]:
        tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=self.device).unsqueeze(0)
        generated: List[int] = []
        with torch.no_grad():
            for _ in range(self.max_gen_len):
                logits = self.model(tokens)
                if self.temperature == 0.0:
                    next_tok = logits[:, -1, :].argmax(dim=-1).item()
                else:
                    probs = torch.softmax(logits[:, -1, :] / self.temperature, dim=-1)
                    next_tok = torch.multinomial(probs, num_samples=1).item()
                if next_tok == self.eos:
                    break
                generated.append(int(next_tok))
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
        predicted = self.generate_answer(prompt_tokens)

        expected_tokens = example.get("answer_tokens", [])
        pred_words = [tuple(w) for w in split_into_words(predicted, self.base_vocab_size)]
        exp_words = [tuple(w) for w in split_into_words(expected_tokens, self.base_vocab_size)]

        # Validate topological consistency by parsing the full sequence.
        full_seq = example["text"][:answer_start_idx] + predicted + [self.eos]
        valid_topo, _query, _topo = self._parse_tokens(full_seq, base_vocab_size=self.base_vocab_size)

        pred_set = set(pred_words)
        exp_set = set(exp_words)

        exact_seq_match = predicted == expected_tokens

        if (not valid_topo or pred_set != exp_set) and self._logged_examples < self._max_logged_examples:
            self._logged_examples += 1
            logger.info(
                "[brevo][debug] example_mismatch valid_topo=%s node_set_ok=%s exact_seq=%s "
                "pred_words=%s exp_words=%s prompt_tail=%s pred_tail=%s exp_tail=%s",
                valid_topo,
                pred_set == exp_set,
                exact_seq_match,
                len(pred_words),
                len(exp_words),
                prompt_tokens[-30:],
                predicted[:60],
                expected_tokens[:60],
            )

        return {
            "skipped": False,
            "valid_topo": valid_topo,
            "node_set_accuracy": pred_set == exp_set,
            "node_count_match": len(pred_words) == len(exp_words),
            "exact_seq_match": exact_seq_match,
            "pred_len": len(pred_words),
            "expected_len": len(exp_words),
        }

    def evaluate_depth(self, eval_file: str, depth: int, max_examples: Optional[int] = None) -> Dict:
        examples = []
        with open(eval_file) as f:
            for i, line in enumerate(f):
                if max_examples is not None and i >= max_examples:
                    break
                examples.append(json.loads(line))
        n_valid_topo = n_node_set = n_node_count = total_pred_len = total_exp_len = n_evaluated = n_skipped = 0
        n_exact_seq = 0
        for ex in examples:
            res = self.evaluate_example(ex)
            if res.get("skipped"):
                n_skipped += 1
                continue
            n_evaluated += 1
            if res["valid_topo"]:
                n_valid_topo += 1
            if res["node_set_accuracy"]:
                n_node_set += 1
            if res["node_count_match"]:
                n_node_count += 1
            if res.get("exact_seq_match"):
                n_exact_seq += 1
            total_pred_len += res["pred_len"]
            total_exp_len += res["expected_len"]
        denom = max(n_evaluated, 1)
        avg_exp_len = total_exp_len / denom
        return {
            "depth": depth,
            "n_evaluated": n_evaluated,
            "n_skipped": n_skipped,
            "valid_topo_rate": n_valid_topo / denom,
            "node_set_accuracy": n_node_set / denom,
            "node_count_match": n_node_count / denom,
            "exact_seq_match": n_exact_seq / denom,
            "length_ratio": (total_pred_len / denom) / max(avg_exp_len, 1),
        }


def evaluate_brevo(cfg: BrevoEvalArgs):
    if not torch.distributed.is_initialized():
        setup_torch_distributed(DistributedArgs())
    if get_global_rank() != 0:
        return
    Path(cfg.dump_dir).mkdir(parents=True, exist_ok=True)
    init_logger(Path(cfg.dump_dir) / "eval.log")
    dump_config(cfg, Path(cfg.dump_dir) / "config.yaml", log_config=False)
    if (
        Path(cfg.ckpt_dir).exists()
        and (Path(cfg.ckpt_dir) / "params.json").exists()
        and next(Path(cfg.ckpt_dir).glob("*.pth"), None) is not None
    ):
        consolidate_path = Path(cfg.ckpt_dir)
    else:
        consolidate_path = Path(cfg.ckpt_dir) / CONSOLIDATE_FOLDER
        if not consolidate_path.exists():
            consolidate_path = consolidate_checkpoints(cfg.ckpt_dir)
    model, _tokenizer = load_consolidated_model(str(consolidate_path), eval_max_seqlen=cfg.eval_max_seqlen)
    evaluator = BrevoEvaluator(
        model=model,
        base_vocab_size=cfg.base_vocab_size if hasattr(cfg, "base_vocab_size") else 4,
        temperature=cfg.temperature,
        max_gen_len=cfg.max_gen_len,
        eval_max_seqlen=cfg.eval_max_seqlen or 2048,
        device=cfg.device,
    )
    eval_files = sorted(Path(cfg.eval_dir).glob("brevo.eval.depth_*.jsonl"))
    if not eval_files:
        logger.error("No evaluation files found in %s", cfg.eval_dir)
        return
    all_results: Dict[str, Dict] = {}
    for eval_file in eval_files:
        depth = int(eval_file.stem.split("_")[-1])
        all_results[f"depth_{depth}"] = evaluator.evaluate_depth(
            str(eval_file), depth=depth, max_examples=cfg.max_examples
        )
    output_file = Path(cfg.dump_dir) / "brevo_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    if wandb.run is not None:
        metrics = {}
        for key, res in all_results.items():
            metrics[f"eval/brevo/{key}/valid_topo_rate"] = res["valid_topo_rate"]
            metrics[f"eval/brevo/{key}/node_set_accuracy"] = res["node_set_accuracy"]
            metrics[f"eval/brevo/{key}/node_count_match"] = res["node_count_match"]
            metrics[f"eval/brevo/{key}/exact_seq_match"] = res["exact_seq_match"]
            metrics[f"eval/brevo/{key}/length_ratio"] = res["length_ratio"]
        avg_valid = sum(r["valid_topo_rate"] for r in all_results.values()) / max(len(all_results), 1)
        metrics["eval/brevo/avg_valid_topo_rate"] = avg_valid
        if cfg.global_step is not None:
            wandb.log(metrics, step=cfg.global_step)
        else:
            wandb.log(metrics)


__all__ = ["BrevoEvalArgs", "evaluate_brevo"]
