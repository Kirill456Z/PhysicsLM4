from pydantic import BaseModel
from typing import Any, Optional


class GenerateSampleRequest(BaseModel):
    config_yaml: str
    task_name: str


class GenerateBatchRequest(BaseModel):
    config_yaml: str
    task_name: str
    batch_size: int = 4


class ValidateConfigRequest(BaseModel):
    config_yaml: str


class ValidateConfigResponse(BaseModel):
    valid: bool
    errors: list[str] = []
    parsed: Optional[dict[str, Any]] = None


class TasksResponse(BaseModel):
    tasks: list[str]


class DefaultConfigResponse(BaseModel):
    task_name: str
    config_yaml: str


class EvaluateRequest(BaseModel):
    config_yaml: str
    task_name: str
    generation: list[int]
    # Task data for reconstruction
    task_index: int
    context: list[int]
    loss_mask: list[int]
    answer_start_index: int
    query_nodes: list[list[int]]
    answer_nodes: list[list[int]]
    num_hops: list[int]
    # Graph data for reconstruction
    graph_nodes: list[list[int]]
    graph_edges: list[dict[str, int]]
