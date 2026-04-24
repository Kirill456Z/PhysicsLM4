from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _ensure_eval_dir(
    task_name: str,
    eval_base_dir: str,
    generation_args: Any,
    eval_config: dict,
    seed: int,
    *,
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    Ensure evaluation jsonl files exist for the given synthetic task.

    This centralizes all `ensure_*` logic so training can stay data-agnostic.
    """
    log = logger or globals().get("logger")
    if log is None:
        log = logging.getLogger(__name__)

    eval_base = Path(eval_base_dir)

    if task_name == "depo":
        return _ensure_depo_eval_data_exists(
            eval_base_dir=eval_base_dir,
            depo_args=generation_args,
            eval_config=eval_config,
            seed=seed,
            logger=log,
        )
    if task_name == "brevo":
        return _ensure_brevo_eval_data_exists(
            eval_base_dir=eval_base_dir,
            brevo_args=generation_args,
            eval_config=eval_config,
            seed=seed,
            logger=log,
        )
    if task_name == "concomp":
        return _ensure_concomp_eval_data_exists(
            eval_base_dir=eval_base_dir,
            concomp_args=generation_args,
            eval_config=eval_config,
            seed=seed,
            logger=log,
        )
    if task_name == "concomp_factor":
        return _ensure_concomp_factor_eval_data_exists(
            eval_base_dir=eval_base_dir,
            concomp_factor_args=generation_args,
            eval_config=eval_config,
            seed=seed,
            logger=log,
        )
    if task_name == "sp":
        return _ensure_sp_eval_data_exists(
            eval_base_dir=eval_base_dir,
            sp_args=generation_args,
            eval_config=eval_config,
            seed=seed,
            logger=log,
        )

    raise ValueError(f"Unsupported synthetic eval task '{task_name}'")


def ensure_eval_data_dir(
    task_name: str,
    eval_base_dir: str,
    generation_args: Any,
    eval_config: dict,
    seed: int,
    *,
    logger: Optional[logging.Logger] = None,
) -> str:
    """Public API for training: returns eval-dir containing jsonl splits."""
    return _ensure_eval_dir(
        task_name,
        eval_base_dir,
        generation_args,
        eval_config,
        seed,
        logger=logger,
    )


def _get_depo_eval_subfolder_name(depo_args: Any) -> str:
    return (
        f"N_{depo_args.max_nodes}_K_{depo_args.max_hops}"
        f"_vs_{depo_args.base_vocab_size}"
        f"_from_{depo_args.min_token_length}_to_{depo_args.max_token_length}"
    )


def _ensure_depo_eval_data_exists(
    *,
    eval_base_dir: str,
    depo_args: Any,
    eval_config: dict,
    seed: int,
    logger: logging.Logger,
) -> str:
    subfolder = _get_depo_eval_subfolder_name(depo_args)
    eval_dir = Path(eval_base_dir) / subfolder

    existing_files = sorted(eval_dir.glob("depo.eval.hop_*.jsonl")) if eval_dir.exists() else []
    if existing_files:
        logger.info(f"Found existing eval data in {eval_dir} ({len(existing_files)} files)")
        return str(eval_dir)

    logger.info(f"No eval data found in {eval_dir}, generating...")
    eval_dir.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).resolve().parent.parent.parent
    depo_module_path = str(project_root / "data_synthetic_pretrain" / "Depo")
    sys.path.insert(0, depo_module_path)
    sys.modules.pop("generate_eval_jsonl", None)
    from generate_eval_jsonl import generate_eval_split

    N = eval_config.get("N") or depo_args.max_nodes
    K_max = eval_config.get("K_max") or depo_args.max_hops
    num_examples = eval_config.get("num_examples_generate", 200)

    logger.info(
        f"Generating eval data: N={N}, K_max={K_max}, num_examples={num_examples}, "
        f"vocab={depo_args.base_vocab_size}, token_len=[{depo_args.min_token_length}, {depo_args.max_token_length}]"
    )

    hop_distance = 1
    while hop_distance <= K_max:
        eval_file_path = eval_dir / f"depo.eval.hop_{hop_distance}.jsonl"
        logger.info(f"  Generating hop {hop_distance}...")
        generate_eval_split(
            hop_distance=hop_distance,
            num_examples=num_examples,
            output_path=str(eval_file_path),
            N=N,
            K_max=K_max,
            base_vocab_size=depo_args.base_vocab_size,
            min_token_length=depo_args.min_token_length,
            max_token_length=depo_args.max_token_length,
            seed=seed,
        )
        hop_distance *= 2

    logger.info(f"Eval data generation complete: {eval_dir}")
    for f in sorted(eval_dir.glob("*.jsonl")):
        logger.info(f"  {f.name}: {f.stat().st_size / 1024:.1f} KB")
    return str(eval_dir)


def _ensure_brevo_eval_data_exists(
    *,
    eval_base_dir: str,
    brevo_args: Any,
    eval_config: dict,
    seed: int,
    logger: logging.Logger,
) -> str:
    B = brevo_args.base_vocab_size
    subfolder = f"N_{brevo_args.max_nodes}_B{B}"
    eval_dir = Path(eval_base_dir) / subfolder

    existing_files = sorted(eval_dir.glob("brevo.eval.depth_*.jsonl")) if eval_dir.exists() else []
    if existing_files:
        logger.info(f"Found existing Brevo eval data in {eval_dir} ({len(existing_files)} files)")
        return str(eval_dir)

    logger.info(f"No Brevo eval data found in {eval_dir}, generating...")
    eval_dir.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).resolve().parent.parent.parent
    brevo_module_path = str(project_root / "data_synthetic_pretrain" / "Brevo")
    sys.path.insert(0, brevo_module_path)
    sys.modules.pop("generate_eval_jsonl", None)
    from generate_eval_jsonl import generate_brevo_eval_split

    N = eval_config.get("N") or brevo_args.max_nodes
    max_depth = eval_config.get("max_depth", 8)
    num_examples = eval_config.get("num_examples_generate", 200)

    logger.info(
        f"Generating Brevo eval data: N={N}, max_depth={max_depth}, num_examples={num_examples}"
    )
    for depth in range(1, max_depth + 1):
        eval_file_path = eval_dir / f"brevo.eval.depth_{depth}.jsonl"
        logger.info(f"  Generating depth {depth}...")
        generate_brevo_eval_split(
            target_depth=depth,
            num_examples=num_examples,
            output_path=str(eval_file_path),
            N=N,
            base_vocab_size=B,
            min_token_length=brevo_args.min_token_length,
            max_token_length=brevo_args.max_token_length,
            seed=seed,
        )

    logger.info(f"Brevo eval data generation complete: {eval_dir}")
    for f in sorted(eval_dir.glob("*.jsonl")):
        logger.info(f"  {f.name}: {f.stat().st_size / 1024:.1f} KB")
    return str(eval_dir)


def _ensure_concomp_eval_data_exists(
    *,
    eval_base_dir: str,
    concomp_args: Any,
    eval_config: dict,
    seed: int,
    logger: logging.Logger,
) -> str:
    B = concomp_args.base_vocab_size
    subfolder = f"N_{concomp_args.max_vertices}_B{B}"
    eval_dir = Path(eval_base_dir) / subfolder

    existing_files = sorted(eval_dir.glob("concomp.eval.size_*.jsonl")) if eval_dir.exists() else []
    if existing_files:
        logger.info(
            f"Found existing ConComp eval data in {eval_dir} ({len(existing_files)} files)"
        )
        return str(eval_dir)

    logger.info(f"No ConComp eval data found in {eval_dir}, generating...")
    eval_dir.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).resolve().parent.parent.parent
    concomp_module_path = str(project_root / "data_synthetic_pretrain" / "ConComp")
    sys.path.insert(0, concomp_module_path)
    sys.modules.pop("generate_eval_jsonl", None)
    from generate_eval_jsonl import generate_concomp_eval_split

    max_answer_size = eval_config.get("max_answer_size", 8)
    num_examples = eval_config.get("num_examples_generate", 500)

    logger.info(
        f"Generating ConComp eval data: N={concomp_args.max_vertices}, sizes=0..{max_answer_size}, "
        f"num_examples={num_examples}"
    )
    for answer_size in range(0, max_answer_size + 1):
        eval_file_path = eval_dir / f"concomp.eval.size_{answer_size}.jsonl"
        logger.info(f"  Generating size={answer_size}...")
        generate_concomp_eval_split(
            target_answer_size=answer_size,
            num_examples=num_examples,
            output_path=str(eval_file_path),
            max_vertices=concomp_args.max_vertices,
            min_vertices=concomp_args.min_vertices,
            edge_p=concomp_args.edge_p,
            base_vocab_size=B,
            min_token_length=concomp_args.min_token_length,
            max_token_length=concomp_args.max_token_length,
            seed=seed,
        )

    logger.info(f"ConComp eval data generation complete: {eval_dir}")
    for f in sorted(eval_dir.glob("*.jsonl")):
        logger.info(f"  {f.name}: {f.stat().st_size / 1024:.1f} KB")
    return str(eval_dir)


def _ensure_concomp_factor_eval_data_exists(
    *,
    eval_base_dir: str,
    concomp_factor_args: Any,
    eval_config: dict,
    seed: int,
    logger: logging.Logger,
) -> str:
    B = concomp_factor_args.base_vocab_size
    subfolder = f"N_{concomp_factor_args.max_vertices}_ep{concomp_factor_args.edge_p}_B{B}"
    eval_dir = Path(eval_base_dir) / subfolder

    existing_files = (
        sorted(eval_dir.glob("concomp_factor.eval.size_*.jsonl")) if eval_dir.exists() else []
    )
    if existing_files:
        logger.info(
            f"Found existing ConCompFactor eval data in {eval_dir} ({len(existing_files)} files)"
        )
        return str(eval_dir)

    logger.info(f"No ConCompFactor eval data found in {eval_dir}, generating...")
    eval_dir.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).resolve().parent.parent.parent
    concomp_module_path = str(project_root / "data_synthetic_pretrain" / "ConComp")
    sys.path.insert(0, concomp_module_path)
    sys.modules.pop("generate_eval_jsonl", None)
    from generate_eval_jsonl import generate_concomp_factor_eval_split

    max_answer_size = eval_config.get("max_answer_size", 8)
    num_examples = eval_config.get("num_examples_generate", 500)

    logger.info(
        f"Generating ConCompFactor eval data: N={concomp_factor_args.max_vertices}, "
        f"sizes=0..{max_answer_size}, num_examples={num_examples}"
    )
    for answer_size in range(0, max_answer_size + 1):
        eval_file_path = eval_dir / f"concomp_factor.eval.size_{answer_size}.jsonl"
        logger.info(f"  Generating size={answer_size}...")
        generate_concomp_factor_eval_split(
            target_answer_size=answer_size,
            num_examples=num_examples,
            output_path=str(eval_file_path),
            max_vertices=concomp_factor_args.max_vertices,
            min_vertices=concomp_factor_args.min_vertices,
            edge_p=concomp_factor_args.edge_p,
            base_vocab_size=B,
            min_token_length=concomp_factor_args.min_token_length,
            max_token_length=concomp_factor_args.max_token_length,
            seed=seed,
        )

    logger.info(f"ConCompFactor eval data generation complete: {eval_dir}")
    for f in sorted(eval_dir.glob("*.jsonl")):
        logger.info(f"  {f.name}: {f.stat().st_size / 1024:.1f} KB")
    return str(eval_dir)


def _ensure_sp_eval_data_exists(
    *,
    eval_base_dir: str,
    sp_args: Any,
    eval_config: dict,
    seed: int,
    logger: logging.Logger,
) -> str:
    B = sp_args.base_vocab_size
    subfolder = f"N_{sp_args.max_nodes}_ep{sp_args.edge_p}_B{B}"
    eval_dir = Path(eval_base_dir) / subfolder

    existing_files = sorted(eval_dir.glob("sp.eval.path_*.jsonl")) if eval_dir.exists() else []
    if existing_files:
        logger.info(f"Found existing SP eval data in {eval_dir} ({len(existing_files)} files)")
        return str(eval_dir)

    logger.info(f"No SP eval data found in {eval_dir}, generating...")
    eval_dir.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).resolve().parent.parent.parent
    sp_module_path = str(project_root / "data_synthetic_pretrain" / "ShortestPath")
    sys.path.insert(0, sp_module_path)
    sys.modules.pop("generate_eval_jsonl", None)
    from generate_eval_jsonl import generate_sp_eval_split

    max_path_length = eval_config.get("max_answer_size", 6)
    num_examples = eval_config.get("num_examples_generate", 500)

    logger.info(
        f"Generating SP eval data: N={sp_args.max_nodes}, edge_p={sp_args.edge_p}, path_lengths=0..{max_path_length}, "
        f"num_examples={num_examples}"
    )
    for pl in range(0, max_path_length + 1):
        eval_file_path = eval_dir / f"sp.eval.path_{pl}.jsonl"
        logger.info(f"  Generating path_length={pl}...")
        generate_sp_eval_split(
            target_path_length=pl,
            num_examples=num_examples,
            output_path=str(eval_file_path),
            max_nodes=sp_args.max_nodes,
            min_nodes=sp_args.min_nodes,
            edge_p=sp_args.edge_p,
            base_vocab_size=B,
            min_token_length=sp_args.min_token_length,
            max_token_length=sp_args.max_token_length,
            seed=seed,
        )

    logger.info(f"SP eval data generation complete: {eval_dir}")
    for f in sorted(eval_dir.glob("*.jsonl")):
        logger.info(f"  {f.name}: {f.stat().st_size / 1024:.1f} KB")
    return str(eval_dir)


__all__ = ["ensure_eval_data_dir"]

