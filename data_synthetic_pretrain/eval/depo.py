#!/usr/bin/env python3
"""
Depo Evaluation Script

This script evaluates a trained model on the Depo dataset across different hop distances.
It measures sequence-level accuracy for each hop distance and logs results to wandb.

Usage:
    python eval_depo.py --ckpt_dir /path/to/checkpoint --eval_dir /path/to/eval/data
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import wandb
from omegaconf import OmegaConf
from tqdm import tqdm

from apps.main.transformer import LMTransformer, LMTransformerArgs
from lingua.args import dataclass_from_dict, dump_config
from lingua.checkpoint import CONSOLIDATE_FOLDER, CONSOLIDATE_NAME, consolidate_checkpoints
from lingua.distributed import DistributedArgs, get_global_rank, setup_torch_distributed
from lingua.logger import init_logger
from lingua.tokenizer import MockTokenizer, Tokenizer

logger = logging.getLogger(__name__)


@dataclass
class DepoEvalArgs:
    """Configuration for Depo evaluation"""

    ckpt_dir: str = ""
    eval_dir: str = ""
    max_examples: Optional[int] = None
    temperature: float = 0.0
    max_gen_len: int = 10
    vocab_size: int = 4
    dump_dir: str = ""
    global_step: Optional[int] = None
    metric_log_dir: Optional[str] = None
    wandb: Optional[Dict] = None
    eval_max_seqlen: Optional[int] = 2048
    device: str = "cuda"


def load_consolidated_model_and_tokenizer(
    consolidate_path: str,
    model_cls=LMTransformer,
    model_args_cls=LMTransformerArgs,
    eval_max_seqlen: Optional[int] = None,
):
    consolidate_path = Path(consolidate_path)

    with open(consolidate_path / "params.json", "r") as f:
        params = json.load(f)
    model_params = params["model"] if "model" in params else params

    if eval_max_seqlen is not None:
        logger.info(
            "Overriding max_seqlen: %s -> %s",
            model_params.get("max_seqlen", 1024),
            eval_max_seqlen,
        )
        model_params["max_seqlen"] = eval_max_seqlen

    model_args = dataclass_from_dict(model_args_cls, model_params)
    model = model_cls(model_args)

    checkpoint = torch.load(
        consolidate_path / CONSOLIDATE_NAME,
        map_location="cpu",
        weights_only=True,
    )
    model_state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(model_state_dict, strict=True)
    model = model.cuda()
    model.eval()

    tokenizer = MockTokenizer()
    tokenizer.n_words = model_args.vocab_size
    return model, tokenizer


class DepoEvaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Tokenizer,
        temperature: float = 0.0,
        max_gen_len: int = 10,
        vocab_size: int = 30,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.max_gen_len = max_gen_len
        self.vocab_size = vocab_size
        self.device = device

    def load_eval_file(self, file_path: str, max_examples: Optional[int] = None) -> List[Dict]:
        examples = []
        with open(file_path, "r") as f:
            for i, line in enumerate(f):
                if max_examples is not None and i >= max_examples:
                    break
                examples.append(json.loads(line))
        return examples

    def _log_example_details(self, hop_distance: int, results: List[Dict], num_examples_to_log: int = 5):
        logger.info("\n%s", "=" * 80)
        logger.info(
            "  DETAILED RESULTS FOR FIRST %s EXAMPLES (HOP %s)",
            num_examples_to_log,
            hop_distance,
        )
        logger.info("%s", "=" * 80)

        logged_count = 0
        for result in results:
            if result.get("skipped", False):
                continue
            if logged_count >= num_examples_to_log:
                break

            example = result["example"]
            example_idx = result["example_idx"]

            answer_start_idx = example.get("answer_start_idx", -1)
            if answer_start_idx > 0:
                prompt_tokens = example["text"][:answer_start_idx]
            else:
                prompt_tokens = []

            predicted = result["predicted_tokens"]
            ground_truth = result["ground_truth_tokens"]
            correct = result["correct"]

            logger.info("\n  Example %s %s:", example_idx, "✓ CORRECT" if correct else "✗ INCORRECT")
            logger.info("    Full sequence length: %s tokens", len(example["text"]))
            logger.info("    Answer starts at index: %s", example.get("answer_start_idx", "NOT SET"))
            logger.info("    Prompt length: %s tokens", len(prompt_tokens))
            logger.info("    Prompt last 20 tokens: %s", prompt_tokens[-20:])
            logger.info(
                "    Full sequence around answer: %s",
                example["text"][max(0, answer_start_idx - 5) : answer_start_idx + 10],
            )
            logger.info("    Ground truth:     %s", ground_truth)
            logger.info("    Predicted:        %s", predicted)

            if not correct:
                if len(predicted) != len(ground_truth):
                    logger.info(
                        "    → Length mismatch: predicted %s, expected %s",
                        len(predicted),
                        len(ground_truth),
                    )
                else:
                    for i, (p, g) in enumerate(zip(predicted, ground_truth)):
                        if p != g:
                            logger.info(
                                "    → First mismatch at position %s: predicted %s, expected %s",
                                i,
                                p,
                                g,
                            )
                            break

            logged_count += 1

        logger.info("%s\n", "=" * 80)

    def generate_answer(self, prompt_tokens: List[int]) -> List[int]:
        tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=self.device).unsqueeze(0)
        generated_tokens = []
        eow_boundary = 2 * self.vocab_size

        with torch.no_grad():
            for _ in range(self.max_gen_len):
                logits = self.model(tokens)
                if self.temperature == 0.0:
                    next_token = logits[:, -1, :].argmax(dim=-1)
                else:
                    probs = torch.softmax(logits[:, -1, :] / self.temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(1)

                next_token_id = next_token.item()
                if next_token_id > eow_boundary:
                    break

                generated_tokens.append(next_token_id)
                tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)

        return generated_tokens

    def evaluate_example(self, example: Dict) -> Dict:
        answer_start_idx = example.get("answer_start_idx", -1)
        if answer_start_idx in (-1, 0):
            logger.warning("Skipping malformed example: answer_start_idx=%s", answer_start_idx)
            return {
                "correct": False,
                "predicted_tokens": [],
                "ground_truth_tokens": example.get("answer_tokens", []),
                "skipped": True,
            }

        prompt_tokens = example["text"][:answer_start_idx]
        if len(prompt_tokens) == 0:
            logger.warning("Skipping example with empty prompt")
            return {
                "correct": False,
                "predicted_tokens": [],
                "ground_truth_tokens": example.get("answer_tokens", []),
                "skipped": True,
            }

        predicted_tokens = self.generate_answer(prompt_tokens)
        eos = 2 * self.vocab_size + 4
        ground_truth_tokens = [t for t in example["answer_tokens"] if t != eos]
        correct = predicted_tokens == ground_truth_tokens

        return {
            "correct": correct,
            "predicted_tokens": predicted_tokens,
            "ground_truth_tokens": ground_truth_tokens,
            "skipped": False,
        }

    def evaluate_hop_distance(
        self,
        eval_file: str,
        hop_distance: int,
        max_examples: Optional[int] = None,
    ) -> Dict:
        logger.info("Evaluating hop distance %s from %s", hop_distance, eval_file)
        examples = self.load_eval_file(eval_file, max_examples)
        logger.info("  Loaded %s examples", len(examples))

        num_correct = 0
        num_skipped = 0
        results = []

        for idx, example in enumerate(tqdm(examples, desc=f"Hop {hop_distance}", disable=True)):
            result = self.evaluate_example(example)
            if result.get("skipped", False):
                num_skipped += 1
            elif result["correct"]:
                num_correct += 1
            result["example_idx"] = idx
            result["example"] = example
            results.append(result)

        num_evaluated = len(examples) - num_skipped
        accuracy = num_correct / num_evaluated if num_evaluated > 0 else 0.0

        if num_skipped > 0:
            logger.warning("  Skipped %s/%s malformed examples", num_skipped, len(examples))

        logger.info("  Hop %s: %s/%s correct (%.4f)", hop_distance, num_correct, num_evaluated, accuracy)
        self._log_example_details(hop_distance, results, num_examples_to_log=5)

        return {
            "accuracy": accuracy,
            "num_correct": num_correct,
            "num_total": num_evaluated,
            "num_skipped": num_skipped,
            "hop_distance": hop_distance,
            "results": results,
        }


def evaluate_depo(cfg: DepoEvalArgs):
    if not torch.distributed.is_initialized():
        setup_torch_distributed(DistributedArgs())
    if get_global_rank() != 0:
        return

    Path(cfg.dump_dir).mkdir(parents=True, exist_ok=True)
    init_logger(Path(cfg.dump_dir) / "eval.log")
    dump_config(cfg, Path(cfg.dump_dir) / "config.yaml", log_config=False)

    logger.info("%s", "=" * 80)
    logger.info("Depo Evaluation")
    logger.info("%s", "=" * 80)
    logger.info("Checkpoint: %s", cfg.ckpt_dir)
    logger.info("Eval dir: %s", cfg.eval_dir)
    logger.info("Output: %s", cfg.dump_dir)
    logger.info("%s", "=" * 80)

    if (
        Path(cfg.ckpt_dir).exists()
        and (Path(cfg.ckpt_dir) / "params.json").exists()
        and next(Path(cfg.ckpt_dir).glob("*.pth"), None) is not None
    ):
        consolidate_path = Path(cfg.ckpt_dir)
    else:
        consolidate_path = Path(cfg.ckpt_dir) / CONSOLIDATE_FOLDER
        if not consolidate_path.exists():
            logger.info("Consolidating checkpoint...")
            consolidate_path = consolidate_checkpoints(cfg.ckpt_dir)

    logger.info("Loading model...")
    model, tokenizer = load_consolidated_model_and_tokenizer(
        str(consolidate_path),
        model_cls=LMTransformer,
        model_args_cls=LMTransformerArgs,
        eval_max_seqlen=cfg.eval_max_seqlen,
    )
    logger.info("Model loaded: %s", model)

    evaluator = DepoEvaluator(
        model=model,
        tokenizer=tokenizer,
        temperature=cfg.temperature,
        max_gen_len=cfg.max_gen_len,
        vocab_size=cfg.vocab_size,
        device=cfg.device,
    )

    eval_files = sorted(Path(cfg.eval_dir).glob("depo.eval.hop_*.jsonl"))
    if not eval_files:
        logger.error("No evaluation files found in %s", cfg.eval_dir)
        return

    all_results = {}
    for eval_file in eval_files:
        hop_distance = int(eval_file.stem.split("_")[-1])
        results = evaluator.evaluate_hop_distance(
            str(eval_file),
            hop_distance=hop_distance,
            max_examples=cfg.max_examples,
        )
        all_results[f"hop_{hop_distance}"] = results

    output_file = Path(cfg.dump_dir) / "depo_results.json"
    with open(output_file, "w") as f:
        json.dump(
            all_results,
            f,
            indent=2,
            default=lambda x: x if isinstance(x, (int, float, str, bool, list)) else str(x),
        )
    logger.info("Results saved to %s", output_file)

    if wandb.run is not None:
        metrics = {}
        for hop_key, results in all_results.items():
            metrics[f"eval/depo/{hop_key}/accuracy"] = results["accuracy"]
        avg_accuracy = sum(r["accuracy"] for r in all_results.values()) / len(all_results)
        metrics["eval/depo/avg_accuracy"] = avg_accuracy
        if cfg.global_step is not None:
            wandb.log(metrics, step=cfg.global_step)
        else:
            wandb.log(metrics)
    else:
        logger.warning("No active wandb run - metrics not logged to wandb")

    logger.info("%s", "=" * 80)
    logger.info("Evaluation Summary")
    logger.info("%s", "=" * 80)
    for hop_key, results in sorted(all_results.items()):
        logger.info("%s: %.4f (%s/%s)", hop_key, results["accuracy"], results["num_correct"], results["num_total"])
    avg_accuracy = sum(r["accuracy"] for r in all_results.values()) / len(all_results)
    logger.info("Average: %.4f", avg_accuracy)
    logger.info("%s", "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Depo model")
    parser.add_argument("--config", type=str, help="Path to config YAML file")
    parser.add_argument("--ckpt_dir", type=str, help="Path to checkpoint directory")
    parser.add_argument("--eval_dir", type=str, help="Path to evaluation data directory")
    parser.add_argument("--dump_dir", type=str, help="Path to output directory")
    parser.add_argument("--max_examples", type=int, default=None, help="Maximum examples per hop")
    parser.add_argument("--global_step", type=int, default=None, help="Training step for logging")
    args = parser.parse_args()

    if args.config:
        cfg = OmegaConf.load(args.config)
        cfg = dataclass_from_dict(DepoEvalArgs, OmegaConf.to_object(cfg))
    else:
        cfg = DepoEvalArgs()

    if args.ckpt_dir:
        cfg.ckpt_dir = args.ckpt_dir
    if args.eval_dir:
        cfg.eval_dir = args.eval_dir
    if args.dump_dir:
        cfg.dump_dir = args.dump_dir
    if args.max_examples:
        cfg.max_examples = args.max_examples
    if args.global_step:
        cfg.global_step = args.global_step
    evaluate_depo(cfg)


__all__ = ["DepoEvalArgs", "evaluate_depo", "main"]


if __name__ == "__main__":
    main()
