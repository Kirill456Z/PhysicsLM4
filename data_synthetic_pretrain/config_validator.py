"""Validate tasks_config.yaml sequence lengths.

Generates N examples per task and reports the token-length distribution
so you can choose an appropriate seq_len for training configs.

Usage:
    uv run python data_synthetic_pretrain/config_validator.py
    uv run python data_synthetic_pretrain/config_validator.py --samples 2000 --windows 1024 2048 4096
    uv run python data_synthetic_pretrain/config_validator.py --config path/to/other_tasks_config.yaml
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from data_synthetic_pretrain.dataloader.data_generation_args import load_generation_args_from_yaml
from data_synthetic_pretrain.tasks import SYNTHETIC_TASKS


PERCENTILES = [50, 90, 95, 99]


def sample_lengths(generator, n: int) -> np.ndarray:
    lengths = np.empty(n, dtype=np.int32)
    for i in range(n):
        task = generator.generate()
        lengths[i] = len(task.context)
    return lengths


def format_row(label: str, lengths: np.ndarray, windows: list[int]) -> str:
    percs = [int(np.percentile(lengths, p)) for p in PERCENTILES]
    window_fits = [f"{100 * np.mean(lengths <= w):5.1f}%" for w in windows]
    perc_str = "  ".join(f"p{p}={v:4d}" for p, v in zip(PERCENTILES, percs))
    win_str = "  ".join(f"w{w}={s}" for w, s in zip(windows, window_fits))
    return f"  {label:<38s}  {perc_str}  |  {win_str}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Report sequence-length distribution for all tasks in tasks_config.yaml")
    parser.add_argument("--config", type=Path, default=None, help="Path to tasks_config.yaml (defaults to package-level file)")
    parser.add_argument("--samples", type=int, default=1000, help="Examples to generate per task (default: 1000)")
    parser.add_argument("--windows", type=int, nargs="+", default=[1024, 2048, 4096], metavar="N",
                        help="Seq-len thresholds to report fit%% for (default: 1024 2048 4096)")
    args = parser.parse_args()

    generation_args = load_generation_args_from_yaml(args.config)
    tasks = generation_args.synthetic_tasks

    if not tasks:
        print("No tasks found in config — nothing to validate.", file=sys.stderr)
        sys.exit(1)

    header_percs = "  ".join(f"p{p}    " for p in PERCENTILES)
    header_wins = "  ".join(f"w{w}  " for w in args.windows)
    print(f"\nGenerating {args.samples} samples per task …\n")
    print(f"  {'task_name':<38s}  {header_percs}  |  {header_wins}")
    print("  " + "-" * (38 + 10 * len(PERCENTILES) + 7 * len(args.windows) + 6))

    all_lengths: dict[str, np.ndarray] = {}

    for task_cfg in tasks:
        task_name = task_cfg.generation_args.get("task_name", task_cfg.task_name)
        builder = SYNTHETIC_TASKS.get(task_cfg.task_name)
        if builder is None:
            print(f"  [SKIP] unknown task type '{task_cfg.task_name}' for '{task_name}'")
            continue

        generator = builder.build_from_dict(task_cfg.generation_args)
        lengths = sample_lengths(generator, args.samples)
        all_lengths[task_name] = lengths
        print(format_row(task_name, lengths, args.windows))

    if len(all_lengths) > 1:
        combined = np.concatenate(list(all_lengths.values()))
        print("  " + "-" * (38 + 10 * len(PERCENTILES) + 7 * len(args.windows) + 6))
        print(format_row("ALL TASKS (combined)", combined, args.windows))

    print()
    print("Recommendation: choose seq_len so that ≥90% of samples fit without truncation.")
    print(f"  p90 values above show how large that window needs to be per task.\n")


if __name__ == "__main__":
    main()
