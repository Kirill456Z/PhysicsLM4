from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

from data_synthetic_pretrain.graph.graph import Graph
from data_synthetic_pretrain.graph.models import GraphGeneratorConfig, NodeWord


class BaseSyntheticTaskConfig(BaseModel):
    task_index: int
    graph_generator_config: GraphGeneratorConfig
    eval_dump_dir: str | None = None


class SynteticTask(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    graph: Graph
    task_index: int
    context: list[int]
    loss_mask: list[int]
    answer_start_index: int

    @model_validator(mode="before")
    @classmethod
    def _json_lists_to_nodeword(cls, data: Any) -> Any:
        """
        model_validate_json leaves nested ``NodeWord``s as ``dict`` in extra
        and subclass fields, which then breaks ``model_dump`` (e.g. hashable
        graph keys) and invariants for task types (depo, bfs, ...).
        """
        if not isinstance(data, dict):
            return data
        for key in ("query_nodes", "answer_nodes", "answer_sequence"):
            if key not in data or not isinstance(data[key], list):
                continue
            data[key] = [
                NodeWord.model_validate(x) if isinstance(x, dict) else x
                for x in data[key]
            ]
        qn = data.get("query_node")
        if qn is not None and isinstance(qn, dict):
            data["query_node"] = NodeWord.model_validate(qn)
        nh = data.get("num_hops")
        if isinstance(nh, list):
            data["num_hops"] = [int(x) for x in nh]
        return data

    @classmethod
    def validate(cls, value):
        assert len(value.context) == len(value.loss_mask)
        return value
