from __future__ import annotations

from typing import List
from collections import deque
from data_synthetic_pretrain.graph.models import NodeWord, EncodingConfig, EncodingFormat
import numpy as np
from pydantic import BaseModel

class Graph(BaseModel):
    nodes : list[NodeWord]
    edges : dict[NodeWord, List[NodeWord]]
    adj_list_encoding_config: EncodingConfig
    
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

    def bfs(
        self, start: NodeWord, max_depth: int | None = None
    ) -> dict[NodeWord, tuple[NodeWord | None, int]]:
        """Run BFS from *start*, returning nodes in traversal order.

        Returns a dict ``{node: (parent, depth)}`` for every reachable node.
        Neighbors are processed in lexicographic token order for determinism.
        If *max_depth* is given, nodes beyond that depth are recorded but not
        expanded (their children are not enqueued).
        """
        result: dict[NodeWord, tuple[NodeWord | None, int]] = {start: (None, 0)}
        queue: deque[NodeWord] = deque([start])
        while queue:
            node = queue.popleft()
            _, depth = result[node]
            if max_depth is not None and depth >= max_depth:
                continue
            for neighbor in sorted(self.edges.get(node, []), key=lambda x: x.tokens):
                if neighbor not in result:
                    result[neighbor] = (node, depth + 1)
                    queue.append(neighbor)
        return result

    @property
    def n(self) -> int:
        return len(self.nodes)