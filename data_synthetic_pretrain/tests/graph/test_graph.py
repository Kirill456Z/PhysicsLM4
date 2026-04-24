import pytest
from data_synthetic_pretrain.graph.graph import Graph
from data_synthetic_pretrain.graph.models import NodeWord, EncodingConfig, SpecialToken


class TestGraph:
    @pytest.fixture
    def adj_list_encoding_config(self):
        return EncodingConfig(pair_sep=SpecialToken(token=25), node_to_neighbors_sep=SpecialToken(token=26))

    @pytest.fixture
    def nodes(self):
        return [
            NodeWord(tokens=(1,)),
            NodeWord(tokens=(2,)),
            NodeWord(tokens=(3,)),
            NodeWord(tokens=(4,)),
            NodeWord(tokens=(5,)),
            NodeWord(tokens=(6,)),
            NodeWord(tokens=(7,)),
        ]
    
    @pytest.fixture
    def complete_graph(self, nodes, adj_list_encoding_config):
        return Graph(
            nodes=nodes[:4],
            edges={
                nodes[0]: [nodes[0], nodes[1], nodes[2], nodes[3]],
                nodes[1]: [nodes[0], nodes[1], nodes[2], nodes[3]],
                nodes[2]: [nodes[0], nodes[1], nodes[2], nodes[3]],
                nodes[3]: [nodes[0], nodes[1], nodes[2], nodes[3]],
            },
            adj_list_encoding_config=adj_list_encoding_config,
        )
    
    @pytest.fixture
    def cycle_graph(self, nodes, adj_list_encoding_config):
        return Graph(
            nodes=nodes[:4],
            edges={
                nodes[0]: [nodes[1]],
                nodes[1]: [nodes[2]],
                nodes[2]: [nodes[3]],
                nodes[3]: [nodes[0]],
            },
            adj_list_encoding_config=adj_list_encoding_config,
        )
    
    def test_adjacency_list_multiple_neighbors(self, nodes, complete_graph, adj_list_encoding_config):
        expected = []
        for node in nodes[:4]:
            expected.extend(node.tokens)
            expected.append(adj_list_encoding_config.node_to_neighbors_sep.token)
            for node_to in nodes[:4]:
                expected.extend(node_to.tokens)
            expected.append(adj_list_encoding_config.pair_sep.token)
        expected = expected[:-1]
        result = complete_graph.adjacency_list()
        assert result == expected
    
    def test_edges_list_length(self, nodes, cycle_graph):
        result = cycle_graph.edges_list()
        assert len(result) == len(cycle_graph.edges) * 2
    
    def test_edges_list_correctness(self, nodes, cycle_graph):
        result = cycle_graph.edges_list()
        for from_node, to_node in zip(result[::2], result[1::2]):
            assert NodeWord(tokens=(from_node,)) in cycle_graph.edges
            assert NodeWord(tokens=(to_node,)) in cycle_graph.edges[NodeWord(tokens=(from_node,))]