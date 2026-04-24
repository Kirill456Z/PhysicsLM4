from abc import abstractmethod
from data_synthetic_pretrain.tasks.models import BaseSyntheticTaskConfig, SynteticTask
from data_synthetic_pretrain.graph.graph_generator import GraphGenerator
from pydantic import BaseModel
from data_synthetic_pretrain.graph.graph import Graph
from pathlib import Path
import hashlib
from collections import defaultdict
import numpy as np

class BaseSynteticTaskGenerator:
    """
    Base class for synthetic tasks in data generation module
    """
    name: str
    args_model : BaseModel = BaseSyntheticTaskConfig

    def __init__(self, config: BaseSyntheticTaskConfig):
        self.config = config
        self.graph_generator = GraphGenerator(config=config.graph_generator_config)
        if config.eval_dump_dir is not None:
            config_hash = hashlib.sha256(self.config.model_dump_json().encode("utf-8")).hexdigest()
            local_dir = Path(config.eval_dump_dir) / config_hash
            config_path = local_dir / "config.jsonl"
            eval_path = local_dir / "eval.jsonl"

            if not local_dir.exists():
                local_dir.mkdir(parents=True, exist_ok=False)
                config_path.write_text(self.config.model_dump_json() + "\n", encoding="utf-8")
                self.eval_set = self._generate_eval_set()
                with open(eval_path, "w", encoding="utf-8") as f:
                    for task in self.eval_set:
                        f.write(task.model_dump_json() + "\n")
            else:
                with open(eval_path, "r", encoding="utf-8") as f:
                    self.eval_set = [SynteticTask.model_validate_json(line) for line in f if line.strip()]
    
    def generate_graph(self, num_nodes: int) -> Graph:
        return self.graph_generator.generate(num_nodes=num_nodes)
    
    @abstractmethod
    def generate(self) -> SynteticTask:
        ...
    
    def generate_batch(self, batch_size: int) -> list[SynteticTask]:
        return [self.generate() for _ in range(batch_size)]
    
    def __hash__(self):
        return hash(self.name, self.config)
    
    @abstractmethod
    def _generate_eval_set(self) -> list[SynteticTask]:
        ...

    def get_eval_set(self) -> list[SynteticTask]:
        return self.eval_set

    @abstractmethod
    def evaluate(self, task: SynteticTask, generation: list[int]) -> dict[str, float]:
        ...

    def _aggregate_eval_results(results: list[dict[str, float]]) -> dict[str, float]:
        aggregated_results = defaultdict(list)
        for result in results:
            for key, value in result.items():
                aggregated_results[key].append(value)
        return {metric_name : np.mean(values) for metric_name, values in aggregated_results.items()}
    
    @abstractmethod
    def batch_evaluate(self, generations: list[list[int]]) -> dict[str, float]:
        results = []
        for task, generation in zip[tuple[SynteticTask, list[int]]](self.eval_set, generations):
            results.append(self.evaluate(task, generation))
        results = self._aggregate_eval_results(results)
        return {
            f"{self.name}/{metric_name}": value for metric_name, value in results.items()
        }