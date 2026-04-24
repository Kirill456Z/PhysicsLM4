from dataclasses import dataclass, field


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
