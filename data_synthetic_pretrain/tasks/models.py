from pydantic import BaseModel, ConfigDict
from data_synthetic_pretrain.graph.models import GraphGeneratorConfig
from data_synthetic_pretrain.graph.graph import Graph

class BaseSyntheticTaskConfig(BaseModel):
    task_index: int
    graph_generator_config: GraphGeneratorConfig

class SynteticTask(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    graph: Graph
    task_index: int
    context: list[int]
    loss_mask: list[int]

    @classmethod
    def validate(cls, value):
        assert len(value.context) == len(value.loss_mask)
        return value
