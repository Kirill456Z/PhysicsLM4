import pytest
from unittest.mock import MagicMock

from data_synthetic_pretrain.graph.graph import Graph
from data_synthetic_pretrain.graph.models import (
    EncodingConfig,
    EncodingFormat,
    GraphGeneratorConfig,
    NodeWord,
    SpecialToken,
)
from data_synthetic_pretrain.tasks.shortest_path import (
    ShortestPathGenerationArgs,
    ShortestPathSynteticTask,
    ShortestPathTaskGenerator,
)


class TestShortestPath:
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
    def shortest_path_generation_args(self, graph_generator_config):
        return ShortestPathGenerationArgs(
            task_index=99,
            graph_generator_config=graph_generator_config,
            max_nodes=10,
            max_distance=4,
            query_separator_token=100,
            eos_token=101,
        )

    @pytest.fixture
    def shortest_path_task(self):
        return ShortestPathSynteticTask(
            task_index=99,
            context=[],
            loss_mask=[],
            graph=MagicMock(spec=Graph, nodes=set([NodeWord(tokens=(5, 11)), NodeWord(tokens=(6, 11)), NodeWord(tokens=(7, 11))])),
            query_node=NodeWord(tokens=(5, 11)),
            answer_nodes=[
                NodeWord(tokens=(5, 11)),
                NodeWord(tokens=(6, 11)),
                NodeWord(tokens=(7, 11)),
            ],
            answer_start_index=8,
        )

    #@pytest.mark.parametrize(
        #"generation, set_accuracy, prefix_accuracy",
        #[
            #([5, 11, 6, 11, 7, 11], 1.0, 1.0),
            #([6, 11, 7, 11, 8, 11], 2 / 3, 0.0),
            #([5, 11, 8, 11, 7, 11], 2 / 3, 1 / 3),
            #([5, 11], 1 / 3, 1 / 3),
            #([], 0.0, 0.0),
        #],
    #)
    #def test_evaluate(
        #self,
        #shortest_path_generation_args,
        #shortest_path_task,
        #generation,
        #set_accuracy,
        #prefix_accuracy,
    #):
        #shortest_path_task_generator = ShortestPathTaskGenerator(
            #shortest_path_generation_args
        #)
        #metrics = shortest_path_task_generator.evaluate(shortest_path_task, generation)

        #assert metrics["set_accuracy"] == set_accuracy
        #assert metrics["prefix_accuracy"] == prefix_accuracy
