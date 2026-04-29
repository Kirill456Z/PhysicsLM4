from data_synthetic_pretrain.tasks.models import BaseSyntheticTaskConfig, SynteticTask
from data_synthetic_pretrain.tasks.base_task import BaseSynteticTaskGenerator
from typing import override
import numpy as np
from pydantic import field_validator
from data_synthetic_pretrain.graph.models import NodeWord
import logging
from data_synthetic_pretrain.graph.utils import break_up_sequence_into_words
logger = logging.getLogger(__name__)

class ConCompFactorGenerationArgs(BaseSyntheticTaskConfig):
    max_nodes: int
    num_eval_samples: int = 100
    query_separator_token: int | None = None
    eos_token: int | None = None


class ConCompFactorSynteticTask(SynteticTask):
    answer_nodes: list[NodeWord]
    components: list[list[NodeWord]]

    @field_validator("answer_nodes", "components", mode="before")
    @classmethod
    def _lists_of_nodeword(cls, v):
        if v is None:
            return v
        if isinstance(v, np.ndarray):
            v = v.tolist()
        if not isinstance(v, list):
            return v
        if isinstance(v[0], list):
            res = []
            for item in v:
                res.append([NodeWord.model_validate(x) if isinstance(x, dict) else x for x in item])
        return [NodeWord.model_validate(x) if isinstance(x, dict) else x for x in v]

class ConCompFactorTaskGenerator(BaseSynteticTaskGenerator):
    name = "concomp_factor"

    def __init__(self, config: ConCompFactorGenerationArgs):
        assert not config.graph_generator_config.is_dag, "ConCompFactorTaskGenerator only supports random graphs"
        assert not config.graph_generator_config.is_directed, "ConCompFactorTaskGenerator only supports undirected graphs"
        super().__init__(config)

    @classmethod
    def build_from_dict(cls, config: dict) -> BaseSynteticTaskGenerator:
        return cls(ConCompFactorGenerationArgs.model_validate(config))

    def max_generation_length(self):
        return self.config.graph_generator_config.max_token_length * self.config.max_nodes + 1

    def _sample_num_nodes(self):
        node_choices = list(range(3, self.config.max_nodes + 1))
        power, bias = 1, pow(self.config.max_nodes, 0.5)
        weights = [1.0 / (pow(i, power) + bias + 1e-12) for i in node_choices]
        total = sum(weights)
        weights = [w / total for w in weights]
        return np.random.choice(node_choices, size=1, p=weights)[0]

    @override
    def generate(self, num_nodes: int | None = None) -> ConCompFactorSynteticTask:
        num_nodes = self._sample_num_nodes() if num_nodes is None else num_nodes

        graph = self.generate_graph(num_nodes=num_nodes)
        visited: set = set()
        raw_components = []
        for node in graph.nodes:
            if node not in visited:
                component = list(graph.bfs(node).keys())
                raw_components.append(component)
                visited.update(component)

        components = [sorted(comp, key=lambda x: x.tokens) for comp in raw_components]
        answer_nodes = [comp[0] for comp in components]

        context = [self.config.task_index] + graph.encode()
        loss_mask = [0] * len(context)
        if self.config.query_separator_token is not None:
            context.append(self.config.query_separator_token)
            loss_mask.append(0)
        answer_start_index = len(context)

        for answer_node in answer_nodes:
            context.extend(answer_node.tokens)
            loss_mask.extend([1] * len(answer_node.tokens))
        if self.config.eos_token is not None:
            context.append(self.config.eos_token)
            loss_mask.append(1)

        return ConCompFactorSynteticTask(
            task_index=self.config.task_index,
            context=context,
            loss_mask=loss_mask,
            graph=graph,
            answer_nodes=answer_nodes,
            components=components,
            answer_start_index=answer_start_index,
        )

    @override
    def _generate_eval_set(self) -> list[ConCompFactorSynteticTask]:
        eval_set = []
        for _ in range(self.config.num_eval_samples):
            eval_set.append(self.generate(num_nodes=self.config.max_nodes))
        for eval_task in eval_set:
            eval_task.context = eval_task.context[:eval_task.answer_start_index]
        return eval_set

    @override
    def evaluate(
        self, task: ConCompFactorSynteticTask, generation: list[int]
    ) -> dict[str, float]:
        generation = list(generation)
        has_eos = 0
        if self.config.eos_token is not None and self.config.eos_token in generation:
            generation = generation[:generation.index(self.config.eos_token)]
            has_eos = 1
        sequence, remainder = break_up_sequence_into_words(generation, self.config.graph_generator_config.base_vocab_size)
        valid_nodes = [node for node in sequence if isinstance(node, NodeWord) and node in task.graph.nodes]
        res = {
            "valid_nodes_ratio": len(valid_nodes) / len(sequence),
        }
        node_to_component = {}
        for i, component in enumerate(task.components):
            for node in component:
                if isinstance(node, dict):
                    node = NodeWord.model_validate(node)
                node_to_component[node] = i
        covered_components = [0] * len(task.components)
        for node in valid_nodes:
            component_index = node_to_component[node]
            covered_components[component_index] += 1
        res["component_recall"] = sum([1 for i in covered_components if i > 0]) / len(task.components)
        res["component_precision"] = (sum([1 for i in covered_components if i == 1]) / len(valid_nodes)) if len(valid_nodes) > 0 else 0.0,
        res["accuracy"] = (sequence == task.answer_nodes)
        res["num_generated_nodes"] = len(valid_nodes)
        res["generated_nodes_to_epected_ratio"] = len(valid_nodes) / len(task.answer_nodes)
        prefix_acc = 0
        for generated_node, expected_node in zip(sequence, task.answer_nodes):
            if generated_node == expected_node:
                prefix_acc += 1
            else:
                break
        res["prefix_accuracy"] = prefix_acc / len(task.answer_nodes)
        res["has_eos"] = has_eos
        return res