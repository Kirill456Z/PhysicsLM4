from __future__ import annotations
from pathlib import Path
import yaml
from pydantic import BaseModel
from typing import Any

DEFAULT_DEPO_CONFIG = """\
# Depo task — observer configuration
# Formatting args
seq_len: 1024
batch_size: 4
pad_token: 0
no_train_label_token: -100

synthetic_tasks_generation_args:
  synthetic_tasks:
    - task_name: depo
      weight: 1.0
      generation_args:
        task_index: 9        # TASK_DEPO token = 2*base_vocab_size+1
        max_nodes: 8
        max_hops: 3
        num_queries: 2
        query_token_base: 200  # HOP_k = 200 + k
        graph_generator_config:
          base_vocab_size: 4
          min_token_length: 2
          max_token_length: 3
          is_directed: true
          is_dag: true
          edge_probability: 0.5
          encoding_config:
            pair_sep: 0
            node_to_neighbors_sep: 0
            format: edges_list
"""


class FormattingConfig(BaseModel):
    seq_len: int = 256
    batch_size: int = 4
    pad_token: int = 0
    no_train_label_token: int = -100


class TaskConfig(BaseModel):
    task_name: str
    weight: float = 1.0
    generation_args: dict[str, Any] = {}


class ObserverConfig(BaseModel):
    formatting: FormattingConfig
    tasks: list[TaskConfig]


def save_task_tab_config(tab_name: str, single_task_config_yaml: str) -> tuple[str, str]:
    """
    Persist a single-task config back into data_synthetic_pretrain/tasks_config.yaml.

    The task entry is matched by generation_args.task_name == tab_name (preferred),
    falling back to the top-level synthetic task "task_name".

    Returns (task_name, updated_full_yaml).
    """
    # Validate the incoming YAML (syntactically + required fields)
    parsed, errors = parse_config(single_task_config_yaml)
    if errors or parsed is None:
        raise ValueError("; ".join(errors or ["Invalid config"]))
    if len(parsed.tasks) != 1:
        raise ValueError("Config must contain exactly one synthetic task")

    incoming = yaml.safe_load(single_task_config_yaml)
    if not isinstance(incoming, dict):
        raise ValueError("Config must be a YAML mapping")
    inc_tasks = (
        incoming.get("synthetic_tasks_generation_args", {}).get("synthetic_tasks", []) or []
    )
    if not isinstance(inc_tasks, list) or len(inc_tasks) != 1 or not isinstance(inc_tasks[0], dict):
        raise ValueError("Config must contain exactly one synthetic task entry")

    incoming_task = inc_tasks[0]
    incoming_task_name = str(incoming_task.get("task_name"))
    incoming_gen_args = incoming_task.get("generation_args", {}) or {}
    incoming_tab_name = (
        incoming_gen_args.get("task_name") if isinstance(incoming_gen_args, dict) else None
    )

    full_path = _tasks_config_path()
    full_yaml = full_path.read_text(encoding="utf-8")
    full = yaml.safe_load(full_yaml)
    if not isinstance(full, dict):
        raise ValueError("tasks_config.yaml must be a YAML mapping")

    # Update global formatting fields from the incoming YAML (if present)
    for k in ("seq_len", "batch_size", "pad_token", "no_train_label_token"):
        if k in incoming:
            full[k] = incoming[k]

    tasks_raw = (
        full.get("synthetic_tasks_generation_args", {}).get("synthetic_tasks", []) or []
    )
    if not isinstance(tasks_raw, list) or not tasks_raw:
        raise ValueError("tasks_config.yaml has no synthetic_tasks list to update")

    def _matches(entry: Any) -> bool:
        if not isinstance(entry, dict):
            return False
        gen_args = entry.get("generation_args", {}) or {}
        entry_tab = gen_args.get("task_name") if isinstance(gen_args, dict) else None
        if isinstance(entry_tab, str) and entry_tab == tab_name:
            return True
        return str(entry.get("task_name")) == tab_name

    idx = next((i for i, t in enumerate(tasks_raw) if _matches(t)), None)
    if idx is None:
        raise ValueError(f"Tab '{tab_name}' not found in tasks_config.yaml")

    # Sanity check: if tab_name refers to generation_args.task_name, keep it aligned
    if isinstance(incoming_tab_name, str) and incoming_tab_name and incoming_tab_name != tab_name:
        raise ValueError(
            f"Incoming config has generation_args.task_name='{incoming_tab_name}' but tab is '{tab_name}'"
        )

    tasks_raw[idx] = incoming_task
    # Write back
    updated_full_yaml = yaml.safe_dump(full, sort_keys=False)
    full_path.write_text(updated_full_yaml, encoding="utf-8")
    return incoming_task_name, updated_full_yaml

def _repo_root() -> Path:
    # backend/config.py -> backend/ -> data_observer/ -> dev/ -> data_synthetic_pretrain/ -> repo root
    return Path(__file__).resolve().parents[4]


def _tasks_config_path() -> Path:
    return _repo_root() / "data_synthetic_pretrain" / "tasks_config.yaml"


def load_tasks_config_yaml() -> str:
    """Load repo-default synthetic tasks config as YAML string."""
    path = _tasks_config_path()
    return path.read_text(encoding="utf-8")


def build_single_task_config_yaml(full_config_yaml: str, task_idx: int) -> tuple[str, str, str]:
    """
    Split repo config into a single-task YAML.

    Returns: (tab_name, generator_task_name, single_task_yaml)
    - tab_name: usually generation_args.task_name (e.g. shortest_path_edges_list)
    - generator_task_name: synthetic task generator key (e.g. shortest_path)
    """
    data = yaml.safe_load(full_config_yaml)
    if not isinstance(data, dict):
        raise ValueError("tasks_config.yaml must be a YAML mapping")

    tasks_raw = (
        data.get("synthetic_tasks_generation_args", {}).get("synthetic_tasks", []) or []
    )
    if not isinstance(tasks_raw, list) or not tasks_raw:
        raise ValueError("No tasks in synthetic_tasks_generation_args.synthetic_tasks")
    if task_idx < 0 or task_idx >= len(tasks_raw):
        raise ValueError(f"task_idx out of range (0..{len(tasks_raw)-1})")

    t = tasks_raw[task_idx]
    if not isinstance(t, dict) or "task_name" not in t:
        raise ValueError("Invalid task entry in synthetic_tasks list")

    gen_args = t.get("generation_args", {}) or {}
    tab_name = None
    if isinstance(gen_args, dict):
        tab_name = gen_args.get("task_name")
    if not isinstance(tab_name, str) or not tab_name:
        tab_name = str(t["task_name"])

    generator_task_name = str(t["task_name"])

    single = {
        "seq_len": data.get("seq_len", 256),
        "batch_size": data.get("batch_size", 4),
        "pad_token": data.get("pad_token", 0),
        "no_train_label_token": data.get("no_train_label_token", -100),
        "synthetic_tasks_generation_args": {
            "synthetic_tasks": [t],
        },
    }

    single_yaml = yaml.safe_dump(single, sort_keys=False)
    return tab_name, generator_task_name, single_yaml


def parse_config(config_yaml: str) -> tuple[ObserverConfig | None, list[str]]:
    errors: list[str] = []
    try:
        data = yaml.safe_load(config_yaml)
    except yaml.YAMLError as e:
        return None, [f"YAML parse error: {e}"]

    if not isinstance(data, dict):
        return None, ["Config must be a YAML mapping"]

    try:
        formatting = FormattingConfig(
            seq_len=data.get("seq_len", 256),
            batch_size=data.get("batch_size", 4),
            pad_token=data.get("pad_token", 0),
            no_train_label_token=data.get("no_train_label_token", -100),
        )
    except Exception as e:
        return None, [f"Formatting config error: {e}"]

    tasks_raw = (
        data.get("synthetic_tasks_generation_args", {}).get("synthetic_tasks", []) or []
    )
    tasks: list[TaskConfig] = []
    for t in tasks_raw:
        try:
            tasks.append(
                TaskConfig(
                    task_name=t["task_name"],
                    weight=t.get("weight", 1.0),
                    generation_args=t.get("generation_args", {}),
                )
            )
        except Exception as e:
            errors.append(f"Task config error: {e}")

    if not tasks:
        errors.append(
            "No tasks defined in synthetic_tasks_generation_args.synthetic_tasks"
        )

    if errors:
        return None, errors

    return ObserverConfig(formatting=formatting, tasks=tasks), []
