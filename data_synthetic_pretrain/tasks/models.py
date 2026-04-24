from pydantic import BaseModel, ConfigDict
from data_synthetic_pretrain.graph.models import GraphGeneratorConfig
from data_synthetic_pretrain.graph.graph import Graph

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

    @classmethod
    def validate(cls, value):
        assert len(value.context) == len(value.loss_mask)
        return value
