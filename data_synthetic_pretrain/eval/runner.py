from __future__ import annotations

import importlib
import logging
import os
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional

from lingua.args import dataclass_from_dict
from data_synthetic_pretrain.eval.ensure import ensure_eval_data_dir


_TASK_TO_EVAL = {
    "depo": ("data_synthetic_pretrain.eval.depo", "DepoEvalArgs", "evaluate_depo"),
    "brevo": ("data_synthetic_pretrain.eval.brevo", "BrevoEvalArgs", "evaluate_brevo"),
    "concomp": ("data_synthetic_pretrain.eval.concomp", "ConCompEvalArgs", "evaluate_concomp"),
    "concomp_factor": (
        "data_synthetic_pretrain.eval.concomp_factor",
        "ConCompFactorEvalArgs",
        "evaluate_concomp_factor",
    ),
    "sp": ("data_synthetic_pretrain.eval.sp", "SPEvalArgs", "evaluate_sp"),
}

logger = logging.getLogger(__name__)


@dataclass
class EvalRequest:
    data_mode: str
    eval_cfg: Dict[str, Any]
    generation_args_by_task: Dict[str, Any]
    ckpt_dir: str
    dump_dir: str
    metric_log_dir: str
    global_step: int
    seed: int
    wandb_cfg: Any = None
    log: Optional[logging.Logger] = None


def run_task_eval(
    task_name: str,
    eval_cfg: Dict[str, Any],
    filter_eval_dict: Callable[[dict, Any], dict],
    eval_data_dir: str,
    ckpt_dir: str,
    dump_dir: str,
    metric_log_dir: str,
    global_step: int,
    generation_args: Any,
    wandb_cfg: Any = None,
) -> None:
    """Run one task eval through the synthetic library entrypoint.

    This centralizes task-dispatch so trainer code stays orchestration-only.
    """
    if task_name not in _TASK_TO_EVAL:
        raise ValueError(f"Unsupported synthetic eval task '{task_name}'")

    module_name, eval_args_cls_name, evaluate_fn_name = _TASK_TO_EVAL[task_name]
    module = importlib.import_module(module_name)
    eval_args_cls = getattr(module, eval_args_cls_name)
    evaluate_fn = getattr(module, evaluate_fn_name)

    cfg = dataclass_from_dict(eval_args_cls, filter_eval_dict(eval_cfg, eval_args_cls))
    cfg.eval_dir = eval_data_dir
    cfg.global_step = global_step
    cfg.ckpt_dir = ckpt_dir
    cfg.dump_dir = dump_dir
    cfg.metric_log_dir = metric_log_dir

    # Propagate shared generation parameters.
    if hasattr(cfg, "base_vocab_size") and hasattr(generation_args, "base_vocab_size"):
        cfg.base_vocab_size = generation_args.base_vocab_size
    if hasattr(cfg, "min_token_length") and hasattr(generation_args, "min_token_length"):
        cfg.min_token_length = generation_args.min_token_length
    if hasattr(cfg, "max_token_length") and hasattr(generation_args, "max_token_length"):
        cfg.max_token_length = generation_args.max_token_length
    if hasattr(cfg, "max_nodes") and hasattr(generation_args, "max_nodes"):
        cfg.max_nodes = generation_args.max_nodes
    if hasattr(cfg, "vocab_size") and hasattr(generation_args, "base_vocab_size"):
        cfg.vocab_size = generation_args.base_vocab_size

    if wandb_cfg is not None:
        cfg.wandb = asdict(wandb_cfg)

    evaluate_fn(cfg)


def _parse_eval_tasks(data_mode: str) -> List[str]:
    if "|" not in data_mode and ":" not in data_mode:
        return [data_mode]
    out = []
    for part in data_mode.split("|"):
        task, _weight = part.split(":", 1)
        out.append(task.strip())
    return out


def _filter_eval_dict(eval_dict: dict, cls) -> dict:
    return {k: v for k, v in eval_dict.items() if hasattr(cls, k) or k in cls.__dataclass_fields__}


def run_synthetic_eval_suite(req: EvalRequest) -> None:
    log = req.log or logger
    tasks = _parse_eval_tasks(req.data_mode)
    for task_name in tasks:
        gen_args = req.generation_args_by_task.get(task_name)
        if gen_args is None:
            raise ValueError(f"Missing generation args for task '{task_name}'")
        eval_data_dir = ensure_eval_data_dir(
            task_name=task_name,
            eval_base_dir=req.eval_cfg["eval_dir"],
            generation_args=gen_args,
            eval_config=req.eval_cfg,
            seed=req.seed,
            logger=log,
        )
        task_dump_dir = req.dump_dir if len(tasks) == 1 else os.path.join(req.dump_dir, task_name)
        log.info(f"Running {task_name} evaluation at step {req.global_step}")
        run_task_eval(
            task_name=task_name,
            eval_cfg=req.eval_cfg,
            filter_eval_dict=_filter_eval_dict,
            eval_data_dir=eval_data_dir,
            ckpt_dir=req.ckpt_dir,
            dump_dir=task_dump_dir,
            metric_log_dir=req.metric_log_dir,
            global_step=req.global_step,
            generation_args=gen_args,
            wandb_cfg=req.wandb_cfg,
        )
