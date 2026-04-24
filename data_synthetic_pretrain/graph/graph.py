from __future__ import annotations

from typing import List
from data_synthetic_pretrain.graph.models import NodeWord, EncodingConfig, EncodingFormat
import numpy as np

class Graph:
    def __init__(self, nodes: list[NodeWord], edges: dict[NodeWord, List[NodeWord]], adj_list_encoding_config: EncodingConfig):
        self.nodes : list[NodeWord] = nodes
        self.edges : dict[NodeWord, List[NodeWord]] = edges
        self.adj_list_encoding_config: EncodingConfig = adj_list_encoding_config
    
    def adjacency_list(self) -> list[int]:
        """Return adjacency list encoding: node_word + node_to_neighbors_sep + neighbor_word + pair_sep."""
        encoded: list[int] = []
        for node in self.nodes:
            encoded.extend(node.tokens)
            if self.adj_list_encoding_config.node_to_neighbors_sep.token:
                encoded.append(self.adj_list_encoding_config.node_to_neighbors_sep.token)
            for neighbor in self.edges[node]:
                encoded.extend(neighbor.tokens)
            if self.adj_list_encoding_config.pair_sep.token:
                encoded.append(self.adj_list_encoding_config.pair_sep.token)
        if self.adj_list_encoding_config.pair_sep.token:
            encoded = encoded[:-1]
        return encoded

    def edges_list(self) -> List[int]:
        """Return flattened edge encoding: src_word + dst_word per edge."""
        encoded: List[int] = []
        edges_flat = []
        for node, neighbors in self.edges.items():
            for neighbor in neighbors:
                edges_flat.append((node, neighbor))
        np.random.shuffle(edges_flat)
        for node, neighbor in edges_flat:
            encoded.extend(node.tokens)
            encoded.extend(neighbor.tokens)
        return encoded

    def encode(self) -> list[int]:
        if self.adj_list_encoding_config.format == EncodingFormat.EDGES_LIST:
            return self.edges_list()
        elif self.adj_list_encoding_config.format == EncodingFormat.ADJACENCY_LIST:
            return self.adjacency_list()
        else:
            raise ValueError(f"Unsupported encoding format: {self.adj_list_encoding_config.format}")

    @property
    def n(self) -> int:
        return len(self.nodes)