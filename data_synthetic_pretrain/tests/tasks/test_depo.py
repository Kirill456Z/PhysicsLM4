import pytest
from unittest.mock import patch

from data_synthetic_pretrain.graph.graph import Graph
from data_synthetic_pretrain.graph.models import (
    GraphGeneratorConfig,
    EncodingConfig,
    SpecialToken,
    EncodingFormat,
    NodeWord,
)
from unittest.mock import MagicMock
from data_synthetic_pretrain.tasks.depo import DepoGenerationArgs, DepoRefactored, DepoSynteticTask


class TestDepo:
    @pytest.fixture
    def graph_generator_config(self):
        return GraphGeneratorConfig(
            base_vocab_size=10,
            min_token_length=1,
            max_token_length=4,
            is_directed=True,
            encoding_config=EncodingConfig(
                pair_sep=SpecialToken(token=10),
                node_to_neighbors_sep=SpecialToken(token=11),
                format=EncodingFormat.EDGES_LIST,
            ),
            is_dag=True,
            edge_probability=0.5,
        )

    @pytest.fixture
    def depo_generation_args(self, graph_generator_config):
        return DepoGenerationArgs(
            task_index=99,
            graph_generator_config=graph_generator_config,
            max_hops=3,
            num_queries=2,
            query_token_base=100,
            max_nodes=40,
        )
    
    def test_context_num_queries_is_correct(self, depo_generation_args):
        depo_task_generator = DepoRefactored(depo_generation_args)
        task = depo_task_generator.generate()
        context = task.context
        queries = [token for token in context if token >= depo_generation_args.query_token_base]
        assert len(queries) == depo_generation_args.num_queries
    
    def test_context_task_index_in_context(self, depo_generation_args):
        depo_task_generator = DepoRefactored(depo_generation_args)
        task = depo_task_generator.generate()
        context = task.context
        assert context[0] == depo_generation_args.task_index
    
    def test_loss_mask_is_correct(self, depo_generation_args):
        depo_task_generator = DepoRefactored(depo_generation_args)
        task = depo_task_generator.generate()
        loss_mask = task.loss_mask
        enabled_context = [token for token, mask in zip(task.context, loss_mask) if mask == 1]
        expected_context = []
        for answer_token in task.answer_nodes:
            expected_context += list(answer_token.tokens)
        assert enabled_context == expected_context

    def test_depo_correctness_cyclic_graph(self, depo_generation_args):
        enc_cfg = depo_generation_args.graph_generator_config.encoding_config

        def fake_generate_graph(num_nodes):
            nodes = [NodeWord(tokens=(i,)) for i in range(10)]
            edges = {nodes[i]: [nodes[(i + 1) % 10]] for i in range(10)}
            return Graph(
                nodes=nodes,
                edges=edges,
                adj_list_encoding_config=enc_cfg,
            )

        depo_task_generator = DepoRefactored(depo_generation_args)
        with patch.object(depo_task_generator, "generate_graph", fake_generate_graph):
            task = depo_task_generator.generate()

        for i in range(len(task.answer_nodes)):
            q = task.query_nodes[i].tokens[0]
            h = int(task.num_hops[i])
            a = task.answer_nodes[i].tokens[0]
            assert a == (q + h) % 10
    
    @pytest.fixture
    def depo_task(self):
        return DepoSynteticTask(
            task_index=99,
            context=[],
            loss_mask=[],
            graph=MagicMock(spec=Graph),
            query_nodes=[NodeWord(tokens=(5, 11))],
            answer_nodes=[NodeWord(tokens=(6, 5, 11))],
            num_hops=[1],
            answer_start_index=8,
        )

    @pytest.mark.parametrize("generation, accuracy, prefix_accuracy", [
        ([6, 5, 11], 1.0, 1.0),
        ([1, 5, 12], 0.0, 0.0),
        ([6, 1, 12], 0.0, 0.3333333333333333),
        ([1, 5, 11], 0.0, 0.0),
    ])
    def test_depo_eval_correct_answer(self, depo_generation_args, depo_task, generation, accuracy, prefix_accuracy):
        depo_task_generator = DepoRefactored(depo_generation_args)
        metrics = depo_task_generator.evaluate(depo_task, generation)
        print(metrics)
        assert metrics["hop_1/accuracy"] == accuracy
        assert metrics["hop_1/prefix_accuracy"] == prefix_accuracy
