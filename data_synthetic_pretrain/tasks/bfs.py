from data_synthetic_pretrain.tasks.models import BaseSyntheticTaskConfig, SynteticTask
from data_synthetic_pretrain.tasks.base_task import BaseSynteticTaskGenerator
from typing import override
import numpy as np
from data_synthetic_pretrain.graph.graph import Graph
from data_synthetic_pretrain.graph.models import NodeWord
from data_synthetic_pretrain.graph.utils import break_up_sequence_into_words
from pydantic import field_validator

class BFSGenerationConfig(BaseSyntheticTaskConfig):
    query_token: int
    max_nodes: int
    eos_token: int
    max_nodes_in_output: int
    num_eval_samples: int = 100

class BFSSynteticTask(SynteticTask):
    query_node: NodeWord
    answer_sequence: list[NodeWord]

    @field_validator("query_node", "answer_sequence", mode="before")
    @classmethod
    def _lists_of_nodeword(cls, v):
        if v is None:
            return v
        if isinstance(v, np.ndarray):
            v = v.tolist()
        if not isinstance(v, list):
            return v
        return [NodeWord.model_validate(x) if isinstance(x, dict) else x for x in v]


class BFSTaskGenerator(BaseSynteticTaskGenerator):
    name = "bfs"

    def __init__(self, config: BFSGenerationConfig):
        super().__init__(config)

    @classmethod
    def build_from_dict(cls, config: dict) -> BaseSynteticTaskGenerator:
        return cls(BFSGenerationConfig.model_validate(config))

    def resolve_for_query(self, graph: Graph, query_node: NodeWord) -> list[NodeWord]:
        return list(graph.bfs(query_node).keys())

    def _sample_num_nodes(self):
        node_choices = list(range(3, self.config.max_nodes + 1))
        power, bias = 1, pow(self.config.max_nodes, 0.5)
        weights = [1.0 / (pow(i, power) + bias + 1e-12) for i in node_choices]
        total = sum(weights)
        weights = [w / total for w in weights]
        return np.random.choice(node_choices, size=1, p=weights)[0]

    def max_generation_length(self):
        return self.config.max_nodes_in_output * (self.config.graph_generator_config.max_token_length) + 1

    @override
    def generate(
        self,
        num_nodes: int | None = None,
    ) -> BFSSynteticTask:
        num_nodes = self._sample_num_nodes() if num_nodes is None else num_nodes

        graph = self.generate_graph(num_nodes=num_nodes)
        query_node = np.random.choice(graph.nodes, size=1)[0]
        context = [self.config.task_index] + graph.encode()
        loss_mask = [0] * len(context)
        context.append(self.config.query_token)
        loss_mask.append(0)
        context.extend(query_node.tokens)
        loss_mask.extend([0] * len(query_node.tokens))
        answer_nodes = self.resolve_for_query(graph, query_node)
        answer_nodes = answer_nodes[:self.config.max_nodes_in_output]
        answer_start_index = len(context) + 1
        for answer_node in answer_nodes:
            context.extend(answer_node.tokens)
            loss_mask.extend([1] * len(answer_node.tokens))
        context.append(self.config.eos_token)
        loss_mask.append(1)
        return BFSSynteticTask(
            task_index=self.config.task_index,
            context=context,
            loss_mask=loss_mask,
            graph=graph,
            query_node=query_node,
            answer_start_index=answer_start_index,
            answer_sequence=answer_nodes,
        )

    @override
    def _generate_eval_set(self) -> list[BFSSynteticTask]:
        eval_set = []
        for _ in range(self.config.num_eval_samples):
            eval_set.append(
                self.generate(num_nodes=self.config.max_nodes)
            )
        for eval_task in eval_set:
            eval_task.context = eval_task.context[:eval_task.answer_start_index]
        return eval_set

    @override
    def evaluate(
        self, task: BFSSynteticTask, generation: list[int]
    ) -> dict[str, float]:
        generation = list(generation)
        has_eos = 0
        if self.config.eos_token in generation:
            generation = generation[:generation.index(self.config.eos_token)]
            has_eos = 1
        sequence, remainder = break_up_sequence_into_words(generation, self.config.graph_generator_config.base_vocab_size)
        generated_nodes = [node for node in sequence if isinstance(node, NodeWord)]
        intersection = set(generated_nodes) & set(task.answer_sequence)
        prefix_acc = 0
        for generated_node, expected_node in zip(generated_nodes, task.answer_sequence):
            if generated_node == expected_node:
                prefix_acc += 1
            else:
                break
        return {
            "set_recall": len(intersection) / len(task.answer_sequence),
            "set_precision": (len(intersection) / len(generated_nodes)) if len(generated_nodes) > 0 else 0.0,
            "prefix_accuracy": prefix_acc / len(task.answer_sequence),
            "has_eos": has_eos,
        }
