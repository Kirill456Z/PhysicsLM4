from re import A
from data_synthetic_pretrain.graph.models import GraphGeneratorConfig
from data_synthetic_pretrain.graph.graph import Graph 
from data_synthetic_pretrain.graph.models import NodeWord, SpecialToken, EncodingFormat
from collections import defaultdict
import numpy as np

class GraphGenerator:
    def __init__(self, config: GraphGeneratorConfig):
        self.config = config
        assert config.base_vocab_size <= 36, "np.base_repr does not support base > 36"
    
    def _generate_node_words(self, n_words: int) -> list[NodeWord]:
        lengths = np.random.randint(self.config.min_token_length, self.config.max_token_length + 1, size=n_words)
        tokens = np.random.randint(1, self.config.base_vocab_size + 1, size=(n_words, np.max(lengths)))
        tokens[range(n_words), lengths - 1] += self.config.base_vocab_size
        return [NodeWord(tokens=tuple(tokens[i, :lengths[i]])) for i in range(n_words)]
    
    def generate(self, num_nodes: int) -> Graph:
        if self.config.is_dag:
            return self.generate_dag(num_nodes)
        else:
            return self.generate_random_graph(num_nodes)
    
    def generate_node_words(self, n_words: int) -> list[NodeWord]:
        result = []
        while len(result) < n_words:
            new_words = self._generate_node_words(2 * n_words)
            unique_words = set(new_words + result)
            result = list(unique_words)
        return result[:n_words]
    
    def generate_random_graph(self, num_nodes: int) -> Graph:
        nodes = self.generate_node_words(num_nodes)
        adj_matrix = np.random.random(size=(num_nodes, num_nodes)) < self.config.edge_probability
        np.fill_diagonal(adj_matrix, False)
        if not self.config.is_directed:
            adj_matrix |= adj_matrix.T
        edges = defaultdict(list)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj_matrix[i, j]:
                    edges[nodes[i]].append(nodes[j])
        return Graph(nodes=nodes, edges=dict(edges), adj_list_encoding_config=self.config.encoding_config)

    def generate_dag(self, num_nodes: int) -> Graph:
        nodes = self.generate_node_words(num_nodes)
        permutation = np.random.permutation(num_nodes)
        edges = defaultdict(list)
        for from_node, to_node in enumerate(permutation):
            edges[nodes[from_node]].append(nodes[to_node])
            if not self.config.is_directed:
                edges[nodes[to_node]].append(nodes[from_node])
        return Graph(nodes=nodes, edges=dict(edges), adj_list_encoding_config=self.config.encoding_config)
    
    def _break_up_into_words(self, tokens: list[int]) -> list[NodeWord | SpecialToken]:
        result = []
        cur_word = []
        for token in tokens:
            if token > 2 * self.config.base_vocab_size:
                result.append(SpecialToken(token=token))
                cur_word = []
            else:
                cur_word.append(token)
                if token > self.config.base_vocab_size:
                    result.append(NodeWord(tokens=tuple(cur_word)))
                    cur_word = []
        if len(cur_word) != 0:
            raise ValueError("Malformed token sequence received for parsing")
        return result

    def decode_from_edges_list(self, edges_list: list[int]) -> Graph:
        word_sequence = self._break_up_into_words(edges_list)
        for word in word_sequence:
            if isinstance(word, SpecialToken):
                raise ValueError("Special tokens are not allowed in edges list")
        if len(word_sequence) % 2 != 0:
            raise ValueError("Number of words in edges list must be even")
        nodes = set()
        edges = defaultdict(list)
        for from_node, to_node in zip(word_sequence[::2], word_sequence[1::2]):
            nodes.add(from_node)
            nodes.add(to_node)
            edges[from_node].append(to_node)
        return Graph(nodes=list(nodes), edges=dict(edges), adj_list_encoding_config=self.config.encoding_config)
    
    def decode_from_adjacency_list(self, adjacency_list: list[int]) -> Graph:
        words_sequence = self._break_up_into_words(adjacency_list)
        pairs = []
        cur_pair = []
        for word in words_sequence:
            if word == self.config.encoding_config.pair_sep:
                pairs.append(cur_pair)
                cur_pair = []
            else:
                cur_pair.append(word)
        pairs.append(cur_pair)
        nodes = set()
        edges = defaultdict(list)
        for pair in pairs:
            from_node = pair[0]
            if self.config.encoding_config.node_to_neighbors_sep.token:
                assert pair[1] == self.config.encoding_config.node_to_neighbors_sep
            for to_node in pair[2:]:
                nodes.add(from_node)
                nodes.add(to_node)
                edges[from_node].append(to_node)
        return Graph(nodes=list(nodes), edges=dict(edges), adj_list_encoding_config=self.config.encoding_config)
    
    def decode(self, representation: list[int]) -> Graph:
        if self.config.encoding_config.format == EncodingFormat.EDGES_LIST:
            return self.decode_from_edges_list(representation)
        elif self.config.encoding_config.format == EncodingFormat.ADJACENCY_LIST:
            return self.decode_from_adjacency_list(representation)
        else:
            raise ValueError(f"Unsupported encoding format: {self.config.encoding_config.format}")