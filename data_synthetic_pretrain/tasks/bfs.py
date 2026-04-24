from data_synthetic_pretrain.tasks.models import BaseSyntheticTaskConfig, SynteticTask
from data_synthetic_pretrain.tasks.base_task import BaseSynteticTaskGenerator
from typing import override
import numpy as np
from data_synthetic_pretrain.graph.graph import Graph
from data_synthetic_pretrain.graph.models import NodeWord


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
        while num_steps > 0:
            query_node = graph.edges[query_node][0]
            num_steps -= 1
        return query_node

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
        answer_nodes = []
        answer_start_index = len(context) + 2
        for i, num_hops_cur in enumerate(num_hops):
            answer = self.resolve_for_query(graph, query_nodes[i], num_hops_cur)
            answer_nodes.append(answer)
            context.append(self.config.query_token_base + num_hops_cur)
            context += list(query_nodes[i].tokens)
            context += list(answer.tokens)

            loss_mask.extend([0] * (len(query_nodes[i].tokens) + 1))
            loss_mask.extend([1] * len(answer.tokens))
        return DepoSynteticTask(
            task_index=self.config.task_index,
            context=context,
            loss_mask=loss_mask,
            graph=graph,
            query_nodes=query_nodes,
            answer_nodes=answer_nodes,
            num_hops=num_hops,
            answer_start_index=answer_start_index,
        )

    @override
    def _generate_eval_set(self) -> list[DepoSynteticTask]:
        eval_set = []
        num_hops = 1
        while num_hops <= self.config.max_hops:
            eval_set.extend(
                self.generate(num_hops=num_hops, num_nodes=self.config.max_nodes, num_query_nodes=1)
            )
            num_hops *= 2
        for eval_task in eval_set:
            eval_task.context = eval_task.context[:eval_task.answer_start_index]
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
