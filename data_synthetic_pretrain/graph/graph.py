from __future__ import annotations

from typing import Any, List
from data_synthetic_pretrain.graph.models import NodeWord, EncodingConfig, EncodingFormat
import numpy as np
from pydantic import BaseModel, field_serializer, field_validator

class Graph(BaseModel):
    nodes : list[NodeWord]
    edges : dict[NodeWord, List[NodeWord]]
    adj_list_encoding_config: EncodingConfig

    @field_serializer("edges", when_used="json")
    def _serialize_edges_for_json(
        self, edges: dict[NodeWord, List[NodeWord]]
    ) -> list[dict[str, Any]]:
        # JSON object keys must be strings, so encode edge pairs explicitly.
        return [{"src": src, "neighbors": neighbors} for src, neighbors in edges.items()]

    @field_validator("edges", mode="before")
    @classmethod
    def _deserialize_edges(cls, value: Any) -> Any:
        if isinstance(value, list):
            return {
                NodeWord.model_validate(item["src"]): [
                    NodeWord.model_validate(neighbor) for neighbor in item["neighbors"]
                ]
                for item in value
            }
        if isinstance(value, dict):
            parsed_edges: dict[NodeWord, List[NodeWord]] = {}
            for raw_src, raw_neighbors in value.items():
                if isinstance(raw_src, str) and raw_src.startswith("tokens=(") and raw_src.endswith(")"):
                    tokens_str = raw_src[len("tokens=(") : -1].strip()
                    if tokens_str:
                        tokens = tuple(int(tok.strip()) for tok in tokens_str.split(","))
                    else:
                        tokens = tuple()
                    src = NodeWord(tokens=tokens)
                else:
                    src = NodeWord.model_validate(raw_src)

                parsed_edges[src] = [NodeWord.model_validate(neighbor) for neighbor in raw_neighbors]
            return parsed_edges
        return value
    
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