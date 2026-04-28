from data_synthetic_pretrain.tasks.models import BaseSyntheticTaskConfig, SynteticTask
from data_synthetic_pretrain.tasks.base_task import BaseSynteticTaskGenerator
from typing import override
import numpy as np
from pydantic import field_validator
from data_synthetic_pretrain.graph.graph import Graph
from data_synthetic_pretrain.graph.models import NodeWord
from data_synthetic_pretrain.graph.utils import break_up_sequence_into_words

class ShortestPathGenerationArgs(BaseSyntheticTaskConfig):
    max_nodes: int
    max_distance: int
    num_eval_samples: int = 30
    query_separator_token: int | None = None
    eos_token: int | None = None


class ShortestPathSynteticTask(SynteticTask):
    query_node: NodeWord
    answer_nodes: list[NodeWord]

    @field_validator("query_node", "answer_nodes", mode="before")
    @classmethod
    def _lists_of_nodeword(cls, v):
        if v is None:
            return v
        if isinstance(v, np.ndarray):
            v = v.tolist()
        if not isinstance(v, list):
            return v
        return [NodeWord.model_validate(x) if isinstance(x, dict) else x for x in v]

class ShortestPathTaskGenerator(BaseSynteticTaskGenerator):
    name = "shortest_path"

    def __init__(self, config: ShortestPathGenerationArgs):
        super().__init__(config)

    @classmethod
    def build_from_dict(cls, config: dict) -> BaseSynteticTaskGenerator:
        return cls(ShortestPathGenerationArgs.model_validate(config))

    def bfs(self, graph: Graph, query_node: NodeWord, max_steps: int) -> list[NodeWord]:
        parents_map = graph.bfs(query_node, max_depth=max_steps)
        # Pick the first node in BFS order that sits at depth == max_steps,
        # or fall back to the last reachable node if the graph is smaller.
        target = next(
            (n for n, (_, d) in parents_map.items() if d == max_steps),
            list(parents_map.keys())[-1],
        )
        path = []
        node: NodeWord | None = target
        while node is not None:
            path.append(node)
            node, _ = parents_map[node]
        return path[::-1]
    
    def max_generation_length(self):
        return self.config.max_token_length * self.config.max_distance + 1

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
        max_distance: int | None = None,
    ) -> ShortestPathSynteticTask:
        num_nodes = self._sample_num_nodes() if num_nodes is None else num_nodes
        max_distance = np.random.randint(1, self.config.max_distance + 1) if max_distance is None else max_distance

        graph = self.generate_graph(num_nodes=num_nodes)
        query_node = np.random.choice(graph.nodes, size=1)[0]
        answer_nodes = self.bfs(graph, query_node, max_distance)
        target_node = answer_nodes[-1]

        context = [self.config.task_index] + graph.encode()
        loss_mask = [0] * len(context)
        if self.config.query_separator_token is not None:
            context.append(self.config.query_separator_token)
            loss_mask.append(0)
        context.extend(query_node.tokens)
        loss_mask.extend([0] * len(query_node.tokens))
        context.extend(target_node.tokens)
        loss_mask.extend([0] * len(target_node.tokens))
        answer_start_index = len(context)
        for answer_node in answer_nodes[1:]:
            context.extend(answer_node.tokens)
            loss_mask.extend([1] * len(answer_node.tokens))
        if self.config.eos_token is not None:
            context.append(self.config.eos_token)
            loss_mask.append(1)

        return ShortestPathSynteticTask(
            task_index=self.config.task_index,
            context=context,
            loss_mask=loss_mask,
            graph=graph,
            query_node=query_node,
            answer_nodes=answer_nodes,
            answer_start_index=answer_start_index,
        )

    @override
    def _generate_eval_set(self) -> list[ShortestPathSynteticTask]:
        eval_set = []
        for _ in range(self.config.num_eval_samples):
            eval_set.append(
                self.generate(num_nodes=self.config.max_nodes, distance=self.config.max_distance)
            )
        for eval_task in eval_set:
            eval_task.context = eval_task.context[:eval_task.answer_start_index]
        return eval_set

    @override
    def evaluate(
        self, task: ShortestPathSynteticTask, generation: list[int]
    ) -> dict[str, float]:
        generation = list(generation)
        answer_nodes = task.answer_nodes[1:]
        sequence, remainder = break_up_sequence_into_words(generation, self.config.graph_generator_config.base_vocab_size)
        generated_nodes = [node for node in sequence if isinstance(node, NodeWord)]
        intersection = set(generated_nodes) & set(answer_nodes)
        prefix_acc = 0
        for generated_node, expected_node in zip(generated_nodes, answer_nodes):
            if generated_node == expected_node:
                prefix_acc += 1
            else:
                break
        correct_nodes = [node for node in generated_nodes if node in task.graph.nodes]
        res = {
            "set_accuracy": len(intersection) / len(answer_nodes),
            "prefix_accuracy": prefix_acc / len(answer_nodes),
            "valid_nodes_ratio": (len(correct_nodes) / len(generated_nodes)) if len(generated_nodes) > 0 else 0.0,
        }
        is_correct_path = False
        if len(correct_nodes) == len(generated_nodes):
            is_correct_path = True
            correct_nodes = [task.query_node] + correct_nodes
            for node, next_node in zip(correct_nodes, correct_nodes[1:]):
                if next_node not in task.graph.edges.get(node, []):
                    is_correct_path = False
                    break
            res["is_correct_path"] = is_correct_path
            if is_correct_path:
                res["path_length_to_min_length"] = (len(correct_nodes) - 1) / len(answer_nodes)
        return res
