import heapq
from data_synthetic_pretrain.graph.models import GraphGeneratorConfig
from data_synthetic_pretrain.graph.graph import Graph
from data_synthetic_pretrain.graph.models import NodeWord, SpecialToken, EncodingFormat
from collections import defaultdict
import numpy as np
from data_synthetic_pretrain.graph.utils import break_up_sequence_into_words

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
            graph = self.generate_dag(num_nodes)
        else:
            graph = self.generate_random_graph(num_nodes)
        if self.config.max_connectivity_components is not None or self.config.min_concomp_size is not None:
            graph = self.constrain_generation(
                graph,
                self.config.max_connectivity_components,
                self.config.min_concomp_size,
            )
        for node in graph.edges:
            graph.edges[node].sort(key=lambda x: x.tokens)
        return graph

    def _merge_components(
        self, graph: Graph, comp1: tuple, comp2: tuple
    ) -> tuple:
        """Add random edges between two components and return their union."""
        connectivity = np.random.uniform(0, 1, (len(comp1), len(comp2)))
        is_edge = connectivity < self.config.edge_probability
        if not np.any(is_edge):
            is_edge[np.random.choice(len(comp1)), np.random.choice(len(comp2))] = True
        for i in range(len(comp1)):
            for j in range(len(comp2)):
                if is_edge[i, j]:
                    graph.edges.setdefault(comp1[i], []).append(comp2[j])
                    graph.edges.setdefault(comp2[j], []).append(comp1[i])
        return tuple(list(comp1) + list(comp2))

    def constrain_generation(
        self,
        graph: Graph,
        max_connectivity_components: int | None = None,
        min_concomp_size: int | None = None,
    ) -> Graph:
        """Merge connectivity components until constraints are satisfied.

        Repeatedly merges the two smallest components until the number of
        components is <= *max_connectivity_components* and every component has
        at least *min_concomp_size* nodes.  Either argument may be ``None`` to
        skip that constraint.
        """
        visited: set = set()
        components: set[tuple] = set()
        for node in graph.nodes:
            if node not in visited:
                component = tuple(graph.bfs(node).keys())
                components.add(component)
                visited.update(component)

        counter = 0
        heap: list = []
        for c in components:
            heapq.heappush(heap, (len(c), counter, c))
            counter += 1

        while len(heap) >= 2:
            size0, _, _ = heap[0]
            too_many = max_connectivity_components is not None and len(components) > max_connectivity_components
            too_small = min_concomp_size is not None and size0 < min_concomp_size
            if not too_many and not too_small:
                break
            _, _, comp1 = heapq.heappop(heap)
            _, _, comp2 = heapq.heappop(heap)
            components.discard(comp1)
            components.discard(comp2)
            merged = self._merge_components(graph, comp1, comp2)
            components.add(merged)
            heapq.heappush(heap, (len(merged), counter, merged))
            counter += 1
        return graph
    
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
        for i, j in np.argwhere(adj_matrix):
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
    
    def decode_from_edges_list(self, edges_list: list[int]) -> Graph:
        word_sequence, remainder = break_up_sequence_into_words(edges_list, self.config.base_vocab_size)
        if len(remainder) != 0:
            raise ValueError("Malformed token sequence received for parsing")
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
        words_sequence, remainder = break_up_sequence_into_words(adjacency_list, self.config.base_vocab_size)
        if len(remainder) != 0:
            raise ValueError("Malformed token sequence received for parsing")
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