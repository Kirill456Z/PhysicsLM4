"""ShortestPath synthetic dataset — BFS shortest-path reasoning on directed graphs.

Format (unified with Depo, Brevo, ConComp):
    <TASK_SP> src1 dst1 src2 dst2 ... <SP_QUERY> start_word end_word <SP_ANS> v1 v2 ... vk <EOS>

- Graph is a directed random graph; edges are stored as flat (src, dst) word pairs.
- Query: the start node word followed by the end node word.
- Answer: the sequence of node words forming the BFS shortest path from start to end,
  including both endpoints.  If no path exists the answer is empty (only EOS follows SP_ANS).
- Loss mask: 1 for answer body tokens + EOS, 0 elsewhere.

Token layout is shared with the other tasks — see data_synthetic_pretrain/shared_vocab.py.
"""

from __future__ import annotations

import os
import sys
from collections import deque
from dataclasses import dataclass, fields
from typing import Any, TypedDict

import numpy as np

_PRETRAIN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PRETRAIN_ROOT not in sys.path:
    sys.path.insert(0, _PRETRAIN_ROOT)

from shared_vocab import vocab_layout, generate_words_numpy, split_into_words, unified_vocab_size
from data_synthetic_pretrain.graph import AdjacencyTokens, Graph
from data_synthetic_pretrain.tasks.config import sample_encoding_format
from data_synthetic_pretrain.base_data_generator import (
    BaseDataGenerator,
    CommonSyntheticGenerationArgs,
)


class ShortestPathGeneratorState(TypedDict):
    max_nodes: int
    min_nodes: int
    edge_p: float
    base_vocab_size: int
    min_token_length: int
    max_token_length: int
    encoding_format: str
    seed: int
    sample_count: int


def sp_vocab_size(base_vocab_size: int = 4) -> int:
    """Return the minimum vocab size for ShortestPath with the given base vocab."""
    return unified_vocab_size(base_vocab_size)


# ---------------------------------------------------------------------------
# BFS shortest-path helper (operates on vertex indices)
# ---------------------------------------------------------------------------

def _bfs_path(adj: list[list[int]], start: int, end: int) -> list[int] | None:
    """BFS from *start* to *end*; return list of vertex indices or None if unreachable."""
    if start == end:
        return [start]
    parent: dict[int, int | None] = {start: None}
    queue: deque[int] = deque([start])
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if v not in parent:
                parent[v] = u
                if v == end:
                    path: list[int] = []
                    node: int | None = end
                    while node is not None:
                        path.append(node)
                        node = parent[node]
                    path.reverse()
                    return path
                queue.append(v)
    return None


# ---------------------------------------------------------------------------
# Public generation function
# ---------------------------------------------------------------------------

def generate_sp_sample(
    rng,
    max_nodes: int,
    min_nodes: int,
    edge_p: float,
    base_vocab_size: int = 4,
    min_token_length: int = 4,
    max_token_length: int = 6,
    encoding_format: str = "edges_list",
) -> dict:
    """Generate one ShortestPath sample using the unified vocabulary.

    Args:
        rng: ``numpy.random.Generator`` instance.
        max_nodes: Maximum number of nodes N; actual n sampled from [min_nodes, N].
        min_nodes: Minimum number of nodes.
        edge_p: Bernoulli directed-edge probability.
        base_vocab_size: Shared base vocabulary size B (must match other tasks).
        min_token_length: Minimum tokens per node-name word.
        max_token_length: Maximum tokens per node-name word.

    Returns:
        dict with ``"text"`` (List[int]) and ``"loss_mask"`` (List[int]).
    """
    layout   = vocab_layout(base_vocab_size)
    TASK_SP  = layout["TASK_SP"]
    SP_QUERY = layout["SP_QUERY"]
    SP_ANS   = layout["SP_ANS"]
    EOS      = layout["EOS"]
    adjacency_tokens = AdjacencyTokens(
        node_to_neighbors=layout["ADJ_NODE_TO_NEIGHBORS"],
        pair_sep=layout["ADJ_PAIR_SEP"],
        no_neighbor=layout["ADJ_NO_NEIGHBOR"],
    )

    # Sample graph size
    n = int(rng.integers(min_nodes, max_nodes + 1))

    # Generate n unique multi-token node words
    nodes = generate_words_numpy(rng, n, base_vocab_size, min_token_length, max_token_length)

    # Directed Bernoulli edges (no self-loops)
    edge_matrix = rng.random((n, n)) < edge_p
    np.fill_diagonal(edge_matrix, False)

    adj: list[list[int]] = [[] for _ in range(n)]
    edge_pairs: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(n):
            if edge_matrix[i, j]:
                adj[i].append(j)
                edge_pairs.append((i, j))

    # Shuffle edge order
    if len(edge_pairs) > 1:
        idx = rng.permutation(len(edge_pairs))
        edge_pairs = [edge_pairs[i] for i in idx]

    graph = Graph.from_edges(nodes, edge_pairs, directed=True)
    chosen_encoding = sample_encoding_format(rng, encoding_format)

    # Context: task token + graph encoding
    tokens: list[int] = [TASK_SP]
    if chosen_encoding == "edges_list":
        tokens += graph.edges_list()
    elif chosen_encoding == "adjacency_list":
        tokens += graph.adjacency_list(adjacency_tokens)
    else:
        raise ValueError(f"Unsupported encoding format '{chosen_encoding}'")
    mask: list[int] = [0] * len(tokens)

    # Pick start and end nodes that both appear in the edge list
    nodes_in_edges: set[int] = set()
    for i, j in edge_pairs:
        nodes_in_edges.add(i)
        nodes_in_edges.add(j)
    candidates = sorted(nodes_in_edges) if len(nodes_in_edges) >= 2 else list(range(n))

    if len(candidates) < 2:
        # Degenerate graph — empty answer
        start_idx, end_idx = 0, 0
    else:
        perm = rng.permutation(len(candidates))
        start_idx = candidates[int(perm[0])]
        end_idx   = candidates[int(perm[1])]

    # Query: SP_QUERY + start_word + end_word
    tokens += [SP_QUERY] + nodes[start_idx] + nodes[end_idx]
    mask   += [0] * (1 + len(nodes[start_idx]) + len(nodes[end_idx]))

    # Answer: SP_ANS + path words (mask = 1) + EOS (mask = 1)
    tokens.append(SP_ANS)
    mask.append(0)

    path = _bfs_path(adj, start_idx, end_idx)
    if path is not None:
        for v in path:
            tokens += nodes[v]
            mask   += [1] * len(nodes[v])

    tokens.append(EOS)
    mask.append(1)

    assert len(tokens) == len(mask)
    return {"text": tokens, "loss_mask": mask, "encoding_format": chosen_encoding}


# ---------------------------------------------------------------------------
# Parsing helper (for evaluation)
# ---------------------------------------------------------------------------

def parse_sp_tokens(tokens: list[int], base_vocab_size: int = 4):
    """Parse a ShortestPath token sequence.

    Returns:
        (valid, start_word, end_word, path_words)
    """
    layout   = vocab_layout(base_vocab_size)
    TASK_SP  = layout["TASK_SP"]
    SP_QUERY = layout["SP_QUERY"]
    SP_ANS   = layout["SP_ANS"]
    EOS      = layout["EOS"]

    if not tokens or tokens[0] != TASK_SP or tokens[-1] != EOS:
        return False, None, None, None

    try:
        idx_q   = tokens.index(SP_QUERY)
        idx_ans = tokens.index(SP_ANS)
    except ValueError:
        return False, None, None, None

    query_words = split_into_words(tokens[idx_q + 1:idx_ans], base_vocab_size)
    path_words  = split_into_words(tokens[idx_ans + 1:-1],    base_vocab_size)

    if len(query_words) != 2:
        return False, None, None, None

    return True, tuple(query_words[0]), tuple(query_words[1]), [tuple(w) for w in path_words]


@dataclass
class ShortestPathGenerationArgs(CommonSyntheticGenerationArgs):
    max_nodes: int = 30
    min_nodes: int = 3
    edge_p: float = 0.3
    train_on_all_tokens: bool = False


def _sp_generation_arg_names() -> frozenset[str]:
    return frozenset(f.name for f in fields(ShortestPathGenerationArgs))


_SP_SAMPLE_KEYS = (
    "max_nodes",
    "min_nodes",
    "edge_p",
    "base_vocab_size",
    "min_token_length",
    "max_token_length",
    "encoding_format",
)


def _sp_sample_from_merged(rng, merged: dict):
    return generate_sp_sample(rng, **{k: merged[k] for k in _SP_SAMPLE_KEYS})


class ShortestPathDataGenerator(BaseDataGenerator):
    NAME = "sp"
    ArgsCls = ShortestPathGenerationArgs

    def _build_generator_state_for_loader(
        self,
        merged: dict,
        *,
        seed: int,
        sample_count: int,
    ) -> ShortestPathGeneratorState:
        return ShortestPathGeneratorState(
            max_nodes=merged["max_nodes"],
            min_nodes=merged["min_nodes"],
            edge_p=merged["edge_p"],
            base_vocab_size=merged["base_vocab_size"],
            min_token_length=merged["min_token_length"],
            max_token_length=merged["max_token_length"],
            encoding_format=merged["encoding_format"],
            seed=seed,
            sample_count=sample_count,
        )

    def generate_sample(self, *, rng=None, common_args=None, **overrides: Any):
        overrides = dict(overrides)
        seed = overrides.pop("seed", 42)
        gen = rng if rng is not None else np.random.default_rng(seed)
        allowed = _sp_generation_arg_names()
        merged = self.merged_args_dict(common_args)
        for k, v in overrides.items():
            if k in allowed:
                merged[k] = v
        return _sp_sample_from_merged(gen, merged)

    def evaluate(self, **kwargs):
        return {"task": self.task_name(), "status": "delegated", **kwargs}


def generate_sp_examples(state: ShortestPathGeneratorState):
    def build_rng(seed: int):
        return np.random.default_rng(seed)

    def sample_fn(s: ShortestPathGeneratorState, rng):
        merged = {k: s[k] for k in _SP_SAMPLE_KEYS}
        result = _sp_sample_from_merged(rng, merged)
        return {
            "text": result["text"],
            "loss_mask": result["loss_mask"],
            "encoding_format": result.get("encoding_format", "edges_list"),
        }

    yield from BaseDataGenerator.iterate_stateful_samples(state, build_rng=build_rng, sample_fn=sample_fn)


ShortestPathDataGenerator._iterate_examples_fn = generate_sp_examples
