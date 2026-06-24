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
from data_synthetic_pretrain.tasks.bfs import (
    BFSGenerationConfig,
    BFSSynteticTask,
    BFSTaskGenerator,
)


class TestBFS:
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
    def bfs_generation_config(self, graph_generator_config):
        return BFSGenerationConfig(
            task_index=99,
            graph_generator_config=graph_generator_config,
            query_token=100,
            max_nodes=10,
            eos_token=101,
            max_nodes_in_output=5,
        )

    #@pytest.fixture
    #def bfs_task(self):
        #return BFSSynteticTask(
            #task_index=99,
            #context=[],
            #loss_mask=[],
            #graph=MagicMock(spec=Graph, nodes=set([NodeWord(tokens=(5, 11)), NodeWord(tokens=(6, 11)), NodeWord(tokens=(7, 11)), NodeWord(tokens=(8, 11))])),
            #query_node=NodeWord(tokens=(5, 11)),
            #answer_sequence=[
                #NodeWord(tokens=(6, 11)),
                #NodeWord(tokens=(7, 11)),
                #NodeWord(tokens=(8, 11)),
            #],
            #answer_start_index=8,
        #)

    #@pytest.mark.parametrize(
        #"generation, set_recall, prefix_accuracy",
        #[
            #([6, 11, 7, 11, 8, 11], 1.0, 1.0),
            #([7, 11, 8, 11, 9, 11], 2 / 3, 0.0),
            #([6, 11, 9, 11, 8, 11], 2 / 3, 1 / 3),
            #([6, 11], 1 / 3, 1 / 3),
            #([], 0.0, 0.0),
        #],
    #)
    #def test_evaluate(
        #self,
        #bfs_generation_config,
        #bfs_task,
        #generation,
        #set_accuracy,
        #prefix_accuracy,
    #):
        #bfs_task_generator = BFSTaskGenerator(bfs_generation_config)
        #metrics = bfs_task_generator.evaluate(bfs_task, generation)

        #assert metrics["set_accuracy"] == set_accuracy
        #assert metrics["prefix_accuracy"] == prefix_accuracy
