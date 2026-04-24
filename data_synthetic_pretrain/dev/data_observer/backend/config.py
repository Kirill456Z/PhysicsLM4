from __future__ import annotations
import yaml
from pydantic import BaseModel
from typing import Any

DEFAULT_DEPO_CONFIG = """\
# Depo task — observer configuration
# Formatting args
seq_len: 256
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
