from dataclasses import dataclass, field
from pathlib import Path

import yaml


TASKS_CONFIG_PATH = Path(__file__).parent.parent / "tasks_config.yaml"


@dataclass
class SynteticTaskGenerationArgs:
    task_name: str
    generation_args: dict
    weight: float = 1.0

@dataclass
class SyntheticTasksGenerationArgs:
    synthetic_tasks: list[SynteticTaskGenerationArgs] = field(default_factory=list)
    eval_dump_dir: str | None = None

@dataclass
class SyntheticTasksFormattingArgs:
    batch_size: int = 32
    seq_len: int = 2048
    pad_token: int = 0
    prefetch_size: int = 32
    n_workers: int = 8
    no_train_label_token: int = -100
    eval_dump_dir: str | None = None
    include_generators: list[str] | None = None


def load_generation_args_from_yaml(path: str | Path | None = None) -> SyntheticTasksGenerationArgs:
    """Load SyntheticTasksGenerationArgs from a yaml file.

    Defaults to the package-level tasks_config.yaml when *path* is not given.
    """
    resolved = Path(path) if path is not None else TASKS_CONFIG_PATH
    with open(resolved) as f:
        cfg = yaml.safe_load(f)
    gen_cfg = cfg.get("synthetic_tasks_generation_args", {}) or {}
    tasks = [
        SynteticTaskGenerationArgs(
            task_name=t["task_name"],
            generation_args=t.get("generation_args", {}),
            weight=t.get("weight", 1.0),
        )
        for t in gen_cfg.get("synthetic_tasks", [])
    ]
    return SyntheticTasksGenerationArgs(
        synthetic_tasks=tasks,
        eval_dump_dir=gen_cfg.get("eval_dump_dir", None),
    )
