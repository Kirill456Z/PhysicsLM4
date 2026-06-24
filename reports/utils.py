"""Utilities for renaming Weights & Biases runs."""

from __future__ import annotations

import os
from typing import Any

import wandb


def _resolve_run_path(run_id: str) -> str:
    """Return a full W&B run path: entity/project/run_id.

    If ``run_id`` already contains slashes, it is treated as a full path.
    Otherwise, ``WANDB_ENTITY`` and ``WANDB_PROJECT`` are used.
    """
    run_id = run_id.strip()
    if not run_id:
        raise ValueError("run_id must be a non-empty string")

    if "/" in run_id:
        return run_id

    entity = os.getenv("WANDB_ENTITY")
    project = os.getenv("WANDB_PROJECT")
    if not entity or not project:
        raise ValueError(
            "run_id is not a full path. Set WANDB_ENTITY and WANDB_PROJECT, "
            "or pass run_id as 'entity/project/run_id'."
        )
    return f"{entity}/{project}/{run_id}"


def rename_wandb_run(run_id: str, new_run_name: str) -> str:
    """Rename a single W&B run and return its full run path.

    Args:
        run_id: W&B run id or full path ``entity/project/run_id``.
        new_run_name: New human-readable run name.
    """
    if not isinstance(new_run_name, str) or not new_run_name.strip():
        raise ValueError("new_run_name must be a non-empty string")

    run_path = _resolve_run_path(run_id)
    api = wandb.Api()
    run = api.run(run_path)
    run.name = new_run_name.strip()
    run.update()
    return run_path


def batch_rename_wandb_run(rename_mapping: dict[str, str]) -> dict[str, Any]:
    """Rename multiple runs.

    ``rename_mapping`` format: ``{run_id: new_run_name}``.

    Returns:
        A summary dictionary with:
            - ``renamed``: list of full run paths successfully updated
            - ``failed``: dict of run_id -> error_message
    """
    if not isinstance(rename_mapping, dict):
        raise TypeError("rename_mapping must be a dict")

    renamed: list[str] = []
    failed: dict[str, str] = {}

    for run_id, new_name in rename_mapping.items():
        try:
            run_path = rename_wandb_run(run_id, new_name)
            renamed.append(run_path)
        except Exception as exc:  # Keep batch progress on partial failures.
            failed[str(run_id)] = str(exc)

    return {"renamed": renamed, "failed": failed}
