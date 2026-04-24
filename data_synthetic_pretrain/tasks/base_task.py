from abc import abstractmethod
from data_synthetic_pretrain.tasks.models import BaseSyntheticTaskConfig, SynteticTask
from data_synthetic_pretrain.graph.graph_generator import GraphGenerator
from pydantic import BaseModel
from data_synthetic_pretrain.graph.graph import Graph

class BaseSynteticTaskGenerator:
    """
    Base class for synthetic tasks in data generation module
    """
    name: str
    args_model : BaseModel = BaseSyntheticTaskConfig

    def __init__(self, config: BaseSyntheticTaskConfig):
        self.config = config
        self.graph_generator = GraphGenerator(config=config.graph_generator_config)
    
    def generate_graph(self, num_nodes: int) -> Graph:
        return self.graph_generator.generate(num_nodes=num_nodes)
    
    @abstractmethod
    def generate(self) -> SynteticTask:
        ...
    
    def generate_batch(self, batch_size: int) -> list[SynteticTask]:
        return [self.generate() for _ in range(batch_size)]
    
