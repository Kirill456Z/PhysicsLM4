# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from data_synthetic_pretrain.dataloader.data_generation_args import (
    SyntheticTasksFormattingArgs,
    load_generation_args_from_yaml,
)

import os

from lingua.checkpoint import CheckpointArgs
from lingua.data import (
    DataArgs,
)
from lingua.distributed import (
    DistributedArgs,
    EnvironmentArgs,
    get_world_size,
)
from lingua.metrics import (
    LoggingArgs,
)
from lingua.optim import OptimArgs
from lingua.profiling import ProfilerArgs
from apps.main.transformer import (
    LMTransformerArgs,
)


logger = logging.getLogger()

@dataclass
class TrainArgs:
    name: str = "lingua"
    dump_dir: str = ""

    seed: int = 42
    # Seed used for synthetic/text data generation and eval-data generation.
    # If None, falls back to `seed` for backward compatibility.
    data_seed: Optional[int] = None

    # Data mode: "depo" for synthetic Depo dataset, "text" for regular text training
    data_mode: str = "depo"

    # Number of gradient accumulation steps
    # Total batch size is batch_size*grad_acc_steps
    grad_acc_steps: int = 1

    gc_collect_freq: int = 1000
    probe_freq: Optional[int] = None

    # Nb optimizer steps to take
    steps: int = 1000

    # If set, per-task / per-encoding logging-only losses run every N optimizer steps only.
    # If None, that extra CE work is disabled entirely.
    decoupled_loss_log_steps: Optional[int] = None

    data: DataArgs = field(default_factory=DataArgs)
    optim: OptimArgs = field(default_factory=OptimArgs)
    model: LMTransformerArgs = field(default_factory=LMTransformerArgs)
    distributed: DistributedArgs = field(default_factory=DistributedArgs)
    env: EnvironmentArgs = field(default_factory=EnvironmentArgs)

    checkpoint: CheckpointArgs = field(default_factory=CheckpointArgs)
    profiling: ProfilerArgs = field(default_factory=ProfilerArgs)
    logging: LoggingArgs = field(default_factory=LoggingArgs)

    synthetic_tasks_formatting_args: SyntheticTasksFormattingArgs = field(default_factory=SyntheticTasksFormattingArgs)
    include_generators: list[str] | None = None

    # If set to None, eval is run locally otherwise it launches a new job with the given number of gpus
    async_eval_gpus: Optional[int] = None
    eval: Optional[Any] = None

def validate_train_args(args: TrainArgs, output_size: int):
    if args.model.vocab_size < 0:
        logger.info(f"Setting model output size to {output_size}")
        args.model.vocab_size = output_size
    assert args.model.vocab_size == output_size, (
        "Vocab size should be the same as output size"
    )

    assert args.dump_dir, "Dump dir not set"

    if args.checkpoint.path is None:
        logger.info(
            f"Setting checkpoint path to {str(Path(args.dump_dir) / 'checkpoints')}"
        )
        args.checkpoint.path = str(Path(args.dump_dir) / "checkpoints")

    if (
        args.distributed.dp_replicate
        * args.distributed.dp_shard
        * args.distributed.tp_size
        != get_world_size()
    ):
        assert get_world_size() % args.distributed.dp_shard == 0
        args.distributed.dp_replicate = get_world_size() // args.distributed.dp_shard

        assert args.distributed.dp_replicate % args.distributed.tp_size == 0
        args.distributed.dp_replicate = (
            args.distributed.dp_replicate // args.distributed.tp_size
        )

        logger.warning(
            f"Setting Data Parallel size to {args.distributed.dp_replicate * args.distributed.dp_shard}"
        )
        assert (
            args.distributed.dp_replicate
            * args.distributed.dp_shard
            * args.distributed.tp_size
            == get_world_size()
        )

        if args.distributed.fsdp_type == "no_shard":
            assert (
                args.distributed.dp_shard == 1
                and args.distributed.dp_replicate == get_world_size()
            )

    args.model.max_seqlen = args.data.seq_len

    if args.distributed.tp_size == 1:
        logger.warning(
            "Tensor parallelism has not been tested for a while, use at your own risk"
        )

    assert args.probe_freq != args.profiling.mem_steps, (
        "Don't profile during probe step"
    )
    assert args.probe_freq != args.profiling.profile_steps, (
        "Don't profile during probe step"
    )

    if args.logging.wandb is not None:
        args.logging.wandb.name = args.name

    if args.probe_freq is not None:
        assert args.distributed.tp_size == 1, (
            "Probing not supported with tensor parallelism"
        )
        assert args.distributed.selective_activation_checkpointing is False, (
            "Probing not supported with selective activation checkpointing"
        )

def parse_data_mode(data_mode: str) -> list[tuple[str, float]]:
    if "|" not in data_mode and ":" not in data_mode:
        return [(data_mode, 1.0)]
    parts = data_mode.split("|")
    result = []
    for part in parts:
        if ":" not in part:
            raise ValueError(f"Mixed data_mode requires 'task:weight' pairs, got: '{part}'")
        task, weight_str = part.split(":", 1)
        result.append((task, float(weight_str)))
    return result


def _depo_max_hops_and_nodes(args: TrainArgs) -> tuple[int, int]:
    generation_args = load_generation_args_from_yaml()
    for task in generation_args.synthetic_tasks:
        if task.task_name == "depo":
            gen = task.generation_args
            return int(gen["max_hops"]), int(gen["max_nodes"])
    raise ValueError(
        "data_mode is 'depo' but no depo task found in tasks_config.yaml synthetic_tasks"
    )


def build_run_name(args: TrainArgs) -> str:
    """Build a deterministic run name from canon settings (depo/text/mix)."""
    parsed_tasks = parse_data_mode(args.data_mode)
    if len(parsed_tasks) > 1:
        mix_str = "_".join(f"{t}{int(round(w * 100))}" for t, w in parsed_tasks)
        return (
            f"mix_{mix_str}"
            f"_bs_{args.data.batch_size}"
            f"_seq_len_{args.data.seq_len}"
        )

    model_args = args.model
    if not model_args.canon_set:
        name = f"{args.data_mode}_llama"
    elif model_args.canon_kernel is None and model_args.canon_init is None:
        name = f"{args.data_mode}_llama"
    else:
        canon_kernel = model_args.canon_kernel if model_args.canon_kernel is not None else 4
        canon_init = model_args.canon_init if model_args.canon_init is not None else "default"
        name = (
            f"{args.data_mode}_ks_{canon_kernel}_{canon_init}_init_"
            f"{'with' if model_args.canon_residual else 'no'}_residual_"
            f"{'trainable' if model_args.train_canon_weights else 'fixed'}_"
            f"{'gamma_scale' if model_args.canon_gamma else ''}"
        )
    if args.data_mode == "depo":
        max_hops, max_nodes = _depo_max_hops_and_nodes(args)
        name = f"{name.rstrip('_')}_{max_hops}_hops_{max_nodes}_nodes"
    elif args.data_mode == "brevo":
        brevo = args.brevo_generation_args
        name = f"{name.rstrip('_')}_{brevo.max_nodes}_nodes_B{brevo.base_vocab_size}"
    elif args.data_mode == "concomp":
        concomp = args.concomp_generation_args
        name = f"{name.rstrip('_')}_{concomp.max_vertices}_nodes_B{concomp.base_vocab_size}"
    elif args.data_mode == "concomp_factor":
        concomp_factor = args.concomp_factor_generation_args
        name = (
            f"{name.rstrip('_')}_{concomp_factor.max_vertices}_nodes_"
            f"ep{concomp_factor.edge_p}_B{concomp_factor.base_vocab_size}"
        )
    elif args.data_mode == "sp":
        sp = args.sp_generation_args
        name = f"{name.rstrip('_')}_{sp.max_nodes}_nodes_ep{sp.edge_p}_B{sp.base_vocab_size}"
    batch_size = args.data.batch_size
    seq_len = args.data.seq_len
    name = f"{name.rstrip('_')}_bs_{batch_size}_seq_len_{seq_len}"
    return name


def _apply_data_to_synthetic_tasks_formatting_args(args: TrainArgs) -> None:
    """Use ``data`` as the source of truth for fields shared with synthetic task formatting."""
    d = args.data
    fmt = args.synthetic_tasks_formatting_args
    fmt.batch_size = d.batch_size
    fmt.seq_len = d.seq_len
    fmt.pad_token = d.pad_token
    fmt.prefetch_size = d.prefetch_size
    fmt.n_workers = d.n_workers
    fmt.no_train_label_token = d.no_train_label_token
    fmt.eval_dump_dir = args.data.eval_dump_dir
    fmt.include_generators = args.include_generators

def prepare_train_args(args: TrainArgs):
    _apply_data_to_synthetic_tasks_formatting_args(args)
    checkpoint_version = os.environ.get('CHECKPOINT_VERSION', None)
    semver_tag = os.environ.get('SEMVER_TAG', None)
        
    version_for_dir = checkpoint_version or semver_tag
    version_for_naming = semver_tag  # Always use semver for wandb/run names

    name_suffix = os.environ.get('NAME_SUFFIX', None)
    wandb_run_base_name = os.environ.get('WANDB_RUN_BASE_NAME', None)

    if wandb_run_base_name:
        run_name = wandb_run_base_name
    else:
        run_name = build_run_name(args)
        if name_suffix:
            run_name = f"{run_name.rstrip('_')}_{name_suffix}"

    args.name = run_name
    args.logging.wandb.name = run_name

    if version_for_dir:
        args.dump_dir = os.path.join(args.dump_dir, f"v{version_for_dir}")

    if version_for_naming:
        if '{semver_tag}' in args.name:
            args.name = args.name.replace('{semver_tag}', version_for_naming)
        else:
            args.name = f"{args.name.rstrip('_')}_{version_for_naming}"
            
        if args.logging.wandb is not None and args.logging.wandb.name:
            if '{semver_tag}' in args.logging.wandb.name:
                args.logging.wandb.name = args.logging.wandb.name.replace('{semver_tag}', version_for_naming)
            else:
                args.logging.wandb.name = f"{args.logging.wandb.name.rstrip('_')}_{version_for_naming}"
            # Set a deterministic wandb id from the run name so we can resume the same run
            args.logging.wandb.id = args.logging.wandb.name
