from data_synthetic_pretrain.tasks.models import BaseSyntheticTaskConfig, SynteticTask
from data_synthetic_pretrain.tasks.base_task import BaseSynteticTaskGenerator
from typing import override
import numpy as np
from data_synthetic_pretrain.graph.graph import Graph
from data_synthetic_pretrain.graph.models import NodeWord
from collections import deque


class BFSGenerationConfig(BaseSyntheticTaskConfig):
    query_token: int


class BFSSynteticTask(SynteticTask):
    query_node: NodeWord
    answer_sequence: list[NodeWord]


class BFSTaskGenerator(BaseSynteticTaskGenerator):
    name = "bfs"

    def __init__(self, config: BFSGenerationConfig):
        super().__init__(config)

    @classmethod
    def build_from_dict(cls, config: dict) -> BaseSynteticTaskGenerator:
        return cls(BFSGenerationConfig.model_validate(config))

    def resolve_for_query(
        self, graph: Graph, query_node: NodeWord, num_steps: int
    ) -> NodeWord:
        bfs_queue = deque([query_node])
        result = []
        while len(bfs_queue) > 0:
            current_node = bfs_queue.popleft()
            result.append(current_node)
            neighbors = sorted(graph.edges[current_node], key=lambda x: x.tokens)
            for neighbor in neighbors:
                if neighbor not in result:
                    bfs_queue.append(neighbor)
        return result

    def _sample_num_nodes(self):
        node_choices = list(range(3, self.config.max_nodes + 1))
        power, bias = 1, pow(self.config.max_nodes, 0.5)
        weights = [1.0 / (pow(i, power) + bias + 1e-12) for i in node_choices]
        total = sum(weights)
        weights = [w / total for w in weights]
        return np.random.choice(node_choices, size=1, p=weights)[0]

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
        answer_start_index = len(context) + 1
        for answer_node in answer_nodes:
            context.extend(answer_node.tokens)
            loss_mask.extend([1] * len(answer_node.tokens))
        return BFSSynteticTask(
            task_index=self.config.task_index,
            context=context,
            loss_mask=loss_mask,
            graph=graph,
            query_node=query_node,
            answer_start_index=answer_start_index,
        )

    @override
    def _generate_eval_set(self) -> list[BFSSynteticTask]:
        eval_set = []
        eval_set.extend(
            self.generate(num_nodes=self.config.max_nodes)
        )
        return eval_set

    @override
    def evaluate(
        self, task: DepoSynteticTask, generation: list[int]
    ) -> dict[str, float]:
        generation = np.array(generation)
        answer_tokens = np.array(task.answer_nodes[0].tokens)
        is_oov_token = generation > 2 * self.config.graph_generator_config.base_vocab_size
        if np.any(is_oov_token):
            truncated_generation = generation[:is_oov_token.argmax() + 1]
        else:
            truncated_generation = generation
        truncated_generation = truncated_generation[:len(answer_tokens)]
        truncated_generation = np.pad(truncated_generation, (0, len(answer_tokens) - len(truncated_generation)), mode='constant', constant_values=-1)
        correct = (truncated_generation == answer_tokens)
        prefix_correct = 0.0 if not correct[0] else (np.argmin(correct) or len(correct)) / len(correct)
        return {
            f"hop_{task.num_hops[0]}/accuracy": np.all(correct),
            f"hop_{task.num_hops[0]}/prefix_accuracy": prefix_correct,
        }
