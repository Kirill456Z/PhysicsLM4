import pytest
from data_synthetic_pretrain.graph.graph_generator import GraphGenerator
from data_synthetic_pretrain.graph.models import GraphGeneratorConfig, NodeWord, SpecialToken, EncodingConfig

class TestGraphGenerator:

    @pytest.fixture
    def adj_list_encoding_config(self):
        return EncodingConfig(pair_sep=SpecialToken(token=25), node_to_neighbors_sep=SpecialToken(token=26))
    
    @pytest.fixture
    def graph_generator_config(self, adj_list_encoding_config):
        return GraphGeneratorConfig(
            base_vocab_size=25,
            min_token_length=1,
            max_token_length=4,
            is_directed=True,
            encoding_config=adj_list_encoding_config,
            is_dag=False,
            edge_probability=0.5
        )
    
    @pytest.fixture
    def graph_generator(self, graph_generator_config):
        return GraphGenerator(config=graph_generator_config)
    
    @pytest.fixture
    def undirected_graph_generator(self, graph_generator_config):
        graph_generator_config.is_directed = False
        return GraphGenerator(config=graph_generator_config)
    
    @pytest.fixture
    def small_vocab_graph_generator(self, graph_generator_config):
        graph_generator_config.base_vocab_size = 10
        graph_generator_config.max_token_length = 3
        return GraphGenerator(config=graph_generator_config)

    def test_generate_node_words_length_in_valid_range(self, graph_generator):
        n_words = 10
        node_words = graph_generator.generate_node_words(n_words)
        assert len(node_words) == n_words
        for node_word in node_words:
            assert len(node_word.tokens) >= graph_generator.config.min_token_length
            assert len(node_word.tokens) <= graph_generator.config.max_token_length
    
    def test_generate_node_words_no_duplicates(self, small_vocab_graph_generator):
        n_words = 500
        node_words = small_vocab_graph_generator.generate_node_words(n_words)
        assert len(node_words) == n_words
        assert len(set(node_words)) == n_words
    
    def test_generate_node_words_tokens_in_valid_range(self, graph_generator):
        n_words = 10
        node_words = graph_generator.generate_node_words(n_words)
        for node_word in node_words:
            for token in node_word.tokens[:-1]:
                assert token in range(1, graph_generator.config.base_vocab_size + 1)
    
    def test_generate_node_words_last_token_is_eow(self, graph_generator):
        n_words = 10
        node_words = graph_generator.generate_node_words(n_words)
        for node_word in node_words:
            assert node_word.tokens[-1] in range(graph_generator.config.base_vocab_size + 1, 2 * graph_generator.config.base_vocab_size + 1)
    
    def test_generate_node_words_generates_min_and_max_length(self, graph_generator):
        n_words = 500
        max_length_encoutered, min_length_encoutered = False, False
        node_words = graph_generator.generate_node_words(n_words)
        for node_word in node_words:
            if len(node_word.tokens) == graph_generator.config.min_token_length:
                min_length_encoutered = True
            if len(node_word.tokens) == graph_generator.config.max_token_length:
                max_length_encoutered = True
        assert min_length_encoutered
        assert max_length_encoutered
    
    def test_generate_random_graph_correct_number_of_nodes(self, graph_generator):
        num_nodes = 10
        graph = graph_generator.generate_random_graph(num_nodes)
        assert len(graph.nodes) == num_nodes
    
    def test_generate_random_graph_test_undirected_edges(self, undirected_graph_generator):
        num_nodes = 10
        graph = undirected_graph_generator.generate_random_graph(num_nodes)
        for node, neighbors in graph.edges.items():
            for neighbor in neighbors:
                assert node in graph.edges[neighbor]
    
    def test_generate_random_graph_test_directed_edges(self, graph_generator):
        num_nodes = 10
        graph = graph_generator.generate_random_graph(num_nodes)
        encountered_directed_only_edge = False
        for node, neighbors in graph.edges.items():
            for neighbor in neighbors:
                if not node in graph.edges[neighbor]:
                    encountered_directed_only_edge = True
        assert encountered_directed_only_edge
    
    def test_generate_dag_one_edge_per_node(self, graph_generator):
        num_nodes = 10
        graph = graph_generator.generate_dag(num_nodes)
        for node, neighbors in graph.edges.items():
            assert len(neighbors) == 1
    
    def test_generate_dag_test_undirected_edges(self, undirected_graph_generator):
        num_nodes = 10
        graph = undirected_graph_generator.generate_dag(num_nodes)
        for node, neighbors in graph.edges.items():
            for neighbor in neighbors:
                assert node in graph.edges[neighbor]
    

class TestGraphGeneratorDecoding:

    @pytest.fixture
    def adj_list_encoding_config(self):
        return EncodingConfig(pair_sep=SpecialToken(token=26), node_to_neighbors_sep=SpecialToken(token=25))
    
    @pytest.fixture
    def graph_generator_config(self, adj_list_encoding_config):
        return GraphGeneratorConfig(
            base_vocab_size=5,
            min_token_length=1,
            max_token_length=4,
            is_directed=True,
            encoding_config=adj_list_encoding_config,
            is_dag=False,
            edge_probability=0.5
        )

    @pytest.fixture
    def graph_generator(self, graph_generator_config):
        return GraphGenerator(config=graph_generator_config)
    
    def test_break_up_into_words_correctness(self, graph_generator):
        expected = [NodeWord(tokens=(1, 6)), NodeWord(tokens=(2, 4, 7)), NodeWord(tokens=(1, 5, 3, 7))]
        edges_list = [1, 6, 2, 4, 7, 1, 5, 3, 7]
        result = graph_generator._break_up_into_words(edges_list)
        assert result == expected
    
    def test_break_up_into_words_with_special_tokens(self, graph_generator):
        expected = [NodeWord(tokens=(1, 6)), SpecialToken(token=25), NodeWord(tokens=(4, 7)), NodeWord(tokens=(1, 5, 3, 7)), SpecialToken(token=26)]
        edges_list = [1, 6, 25, 4, 7, 1, 5, 3, 7, 26]
        result = graph_generator._break_up_into_words(edges_list)
        assert result == expected
    
    def test_break_up_into_words_raises_with_malformed_sequence(self, graph_generator):
        edges_list = [1, 1, 8, 2, 2]
        with pytest.raises(ValueError):
            graph_generator._break_up_into_words(edges_list)
    
    def test_decode_from_edges_list_correctness(self, graph_generator):
        edges_list = [6, 7, 7, 8, 8, 9, 9, 10]
        graph = graph_generator.decode_from_edges_list(edges_list)
        assert len(graph.nodes) == 5
        assert len(graph.edges) == 4
        assert graph.edges[NodeWord(tokens=(7,))] == [NodeWord(tokens=(8,))]
    
    def test_decode_from_adjacency_list_correctness(self, graph_generator):
        adjacency_list = [6, 25, 7, 8, 26, 7, 25, 8, 26, 8, 25, 9, 26, 9, 25, 10]
        graph = graph_generator.decode_from_adjacency_list(adjacency_list)
        assert len(graph.nodes) == 5
        assert len(graph.edges) == 4
        assert graph.edges[NodeWord(tokens=(7,))] == [NodeWord(tokens=(8,))]
        assert graph.edges[NodeWord(tokens=(6,))] == [NodeWord(tokens=(7,)), NodeWord(tokens=(8,))]

class TestGraphGeneratorEncodingDecodingPersistency:
    @pytest.fixture
    def adj_list_encoding_config(self):
        return EncodingConfig(pair_sep=SpecialToken(token=25), node_to_neighbors_sep=SpecialToken(token=26))

    @pytest.fixture
    def graph_generator_config(self, adj_list_encoding_config):
        return GraphGeneratorConfig(
            base_vocab_size=5,
            min_token_length=1,
            max_token_length=4,
            is_directed=True,
            encoding_config=adj_list_encoding_config,
            is_dag=False,
            edge_probability=0.5
        )

    @pytest.fixture
    def graph_generator(self, graph_generator_config):
        return GraphGenerator(config=graph_generator_config)
    
    def test_adjacency_list(self, graph_generator):
        graph = graph_generator.generate_random_graph(50)
        adjacency_list = graph.adjacency_list()
        decoded_graph = graph_generator.decode_from_adjacency_list(adjacency_list)
        assert set(decoded_graph.nodes) == set(graph.nodes)
        for node in decoded_graph.nodes:
            assert set(decoded_graph.edges[node]) == set(graph.edges[node])
    
    def test_edges_list(self, graph_generator):
        graph = graph_generator.generate_random_graph(50)
        edges_list = graph.edges_list()
        decoded_graph = graph_generator.decode_from_edges_list(edges_list)
        assert set(decoded_graph.nodes) == set(graph.nodes)
        for node in decoded_graph.nodes:
            assert set(decoded_graph.edges[node]) == set(graph.edges[node])