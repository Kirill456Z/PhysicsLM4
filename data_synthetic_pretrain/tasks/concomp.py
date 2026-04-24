"""ConComp synthetic dataset — Connectivity / BFS reasoning.

Format (unified with Depo and Brevo):
    <TASK_CONCOMP> src1 dst1 src2 dst2 ... <CONCOMP_QUERY> q <CONCOMP_ANS> a1 a2 ... <EOS>

- No vertex-listing preamble; the graph is represented purely by its edge list.
- Node names are multi-token words (same encoding as Depo / Brevo).
- Only vertices that appear in at least one edge are used as query nodes so the model always
  sees the query vertex in the input context.
- Loss mask: 1 for answer-body tokens (after CONCOMP_ANS up to and including EOS), 0 elsewhere.

Token layout is shared with Depo and Brevo — see data_synthetic_pretrain/shared_vocab.py.
"""

import os
import sys
from dataclasses import dataclass, fields
import numpy as np
from typing import Any, TypedDict

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


class ConCompGeneratorState(TypedDict):
    max_vertices: int
    min_vertices: int
    edge_p: float
    queries_count: int
    base_vocab_size: int
    min_token_length: int
    max_token_length: int
    encoding_format: str
    seed: int
    sample_count: int


def concomp_vocab_size(base_vocab_size: int = 4, max_hops: int = 0) -> int:
    """Return the minimum vocab size for ConComp with the given base vocab."""
    return unified_vocab_size(base_vocab_size, max_hops)


# ---------------------------------------------------------------------------
# BFS helper (operates on vertex indices)
# ---------------------------------------------------------------------------

def _bfs_indices(adj: list[list[int]], n: int, start: int, rng) -> list[int]:
    """BFS from *start* in adjacency list *adj*; return visited indices (start excluded).

    Unseen neighbours at each step are shuffled for output variety.
    """
    seen     = {start}
    visited  = set()
    queue    = [start]
    order: list[int] = []

    while queue:
        u = queue.pop(0)
        if u in visited:
            continue
        visited.add(u)
        unseen = [v for v in adj[u] if v not in seen]
        rng.shuffle(unseen)
        for v in unseen:
            seen.add(v)
            order.append(v)
            queue.append(v)

    return order


def sample_graph_instance(
    rng,
    max_vertices: int,
    min_vertices: int,
    edge_p: float,
    base_vocab_size: int,
    min_token_length: int,
    max_token_length: int,
    *,
    directed: bool = True,
) -> tuple[list[list[int]], list[list[int]], list[tuple[int, int]]]:
    """Sample a graph instance and return vertices, adjacency list, and edge pairs.

    This helper is shared across ConComp-family tasks.
    """
    n = int(rng.integers(min_vertices, max_vertices + 1))
    vertices = generate_words_numpy(
        rng, n, base_vocab_size, min_token_length, max_token_length
    )

    if directed:
        random_graph = rng.random((n, n)) < edge_p
        adj: list[list[int]] = [[] for _ in range(n)]
        edge_pairs: list[tuple[int, int]] = []
        for i in range(n):
            for j in range(n):
                if random_graph[i, j]:
                    adj[i].append(j)
                    edge_pairs.append((i, j))
    else:
        random_graph = rng.random((n, n)) < edge_p
        adj = [[] for _ in range(n)]
        edge_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if random_graph[i, j]:
                    adj[i].append(j)
                    adj[j].append(i)
                    edge_pairs.append((i, j))

    edge_pairs_arr = np.array(edge_pairs, dtype=np.int64) if edge_pairs else None
    if edge_pairs_arr is not None and len(edge_pairs_arr) > 1:
        rng.shuffle(edge_pairs_arr)
        edge_pairs = [(int(r[0]), int(r[1])) for r in edge_pairs_arr]

    return vertices, adj, edge_pairs


# ---------------------------------------------------------------------------
# Public generation function
# ---------------------------------------------------------------------------

def generate_concomp_sample(
    rng,
    max_vertices: int,
    min_vertices: int,
    edge_p: float,
    queries_count: int,
    base_vocab_size: int = 4,
    min_token_length: int = 4,
    max_token_length: int = 6,
    encoding_format: str = "edges_list",
) -> dict:
    """Generate one ConComp sample using the unified vocabulary.

    Args:
        rng: ``numpy.random.Generator`` instance.
        max_vertices: Maximum number of vertices N.
        min_vertices: Minimum number of vertices.
        edge_p: Bernoulli edge probability.
        queries_count: Number of query-answer pairs per sample.
        base_vocab_size: Shared base vocabulary size B (must match other tasks).
        min_token_length: Minimum tokens per vertex-name word.
        max_token_length: Maximum tokens per vertex-name word.

    Returns:
        dict with ``"text"`` (List[int]) and ``"loss_mask"`` (List[int]).
    """
    layout         = vocab_layout(base_vocab_size)
    TASK_CONCOMP   = layout["TASK_CONCOMP"]
    CONCOMP_QUERY  = layout["CONCOMP_QUERY"]
    CONCOMP_ANS    = layout["CONCOMP_ANS"]
    EOS            = layout["EOS"]
    adjacency_tokens = AdjacencyTokens(
        node_to_neighbors=layout["ADJ_NODE_TO_NEIGHBORS"],
        pair_sep=layout["ADJ_PAIR_SEP"],
        no_neighbor=layout["ADJ_NO_NEIGHBOR"],
    )

    vertices, adj, edge_pairs = sample_graph_instance(
        rng,
        max_vertices=max_vertices,
        min_vertices=min_vertices,
        edge_p=edge_p,
        base_vocab_size=base_vocab_size,
        min_token_length=min_token_length,
        max_token_length=max_token_length,
        directed=True,
    )
    n = len(vertices)

    graph = Graph.from_edges(vertices, edge_pairs, directed=True)

    chosen_encoding = sample_encoding_format(rng, encoding_format)

    # Context: task type + encoded graph
    tokens = [TASK_CONCOMP]
    if chosen_encoding == "edges_list":
        tokens += graph.edges_list()
    elif chosen_encoding == "adjacency_list":
        tokens += graph.adjacency_list(adjacency_tokens)
    else:
        raise ValueError(f"Unsupported encoding format '{chosen_encoding}'")
    mask = [0] * len(tokens)

    # Only query vertices that appear in the edge list (model has seen them)
    vertices_with_edges: set[int] = set()
    for i, j in edge_pairs:
        vertices_with_edges.add(i)
        vertices_with_edges.add(j)
    available = sorted(vertices_with_edges) if vertices_with_edges else list(range(n))
    rng.shuffle(available)
    query_indices = available[: min(queries_count, len(available))]

    for start_idx in query_indices:
        answer_indices = _bfs_indices(adj, n, start_idx, rng)

        # CONCOMP_QUERY + query word (mask = 0)
        tokens += [CONCOMP_QUERY] + vertices[start_idx]
        mask   += [0] * (1 + len(vertices[start_idx]))

        # CONCOMP_ANS (mask = 0) + answer words (mask = 1)
        tokens.append(CONCOMP_ANS)
        mask.append(0)
        for idx in answer_indices:
            tokens += vertices[idx]
            mask   += [1] * len(vertices[idx])

    # EOS — model is trained to predict it
    tokens.append(EOS)
    mask.append(1)

    assert len(tokens) == len(mask)
    return {"text": tokens, "loss_mask": mask, "encoding_format": chosen_encoding}


# ---------------------------------------------------------------------------
# Parsing helper (for evaluation)
# ---------------------------------------------------------------------------

def parse_concomp_tokens(tokens: list[int], base_vocab_size: int = 4):
    """Parse a ConComp token sequence and verify BFS correctness.

    Returns:
        (valid, query_word, answer_words)
    """
    layout        = vocab_layout(base_vocab_size)
    TASK_CONCOMP  = layout["TASK_CONCOMP"]
    CONCOMP_QUERY = layout["CONCOMP_QUERY"]
    CONCOMP_ANS   = layout["CONCOMP_ANS"]
    EOS           = layout["EOS"]

    if not tokens or tokens[0] != TASK_CONCOMP or tokens[-1] != EOS:
        return False, None, None

    try:
        idx_q   = tokens.index(CONCOMP_QUERY)
        idx_ans = tokens.index(CONCOMP_ANS)
    except ValueError:
        return False, None, None

    edge_words   = split_into_words(tokens[1:idx_q], base_vocab_size)
    query_words  = split_into_words(tokens[idx_q + 1:idx_ans], base_vocab_size)
    answer_words = split_into_words(tokens[idx_ans + 1:-1], base_vocab_size)

    if len(edge_words) % 2 != 0 or len(query_words) != 1:
        return False, None, None

    return True, tuple(query_words[0]), [tuple(w) for w in answer_words]


@dataclass
class ConCompGenerationArgs(CommonSyntheticGenerationArgs):
    max_vertices: int = 50
    min_vertices: int = 3
    edge_p: float = 0.4
    queries_count: int = 1
    train_on_all_tokens: bool = False


def _concomp_generation_arg_names() -> frozenset[str]:
    return frozenset(f.name for f in fields(ConCompGenerationArgs))


_CONCOMP_SAMPLE_KEYS = (
    "max_vertices",
    "min_vertices",
    "edge_p",
    "queries_count",
    "base_vocab_size",
    "min_token_length",
    "max_token_length",
    "encoding_format",
)


def _concomp_sample_from_merged(rng, merged: dict):
    return generate_concomp_sample(rng, **{k: merged[k] for k in _CONCOMP_SAMPLE_KEYS})


class ConCompDataGenerator(BaseDataGenerator):
    NAME = "concomp"
    ArgsCls = ConCompGenerationArgs

    def _build_generator_state_for_loader(
        self,
        merged: dict,
        *,
        seed: int,
        sample_count: int,
    ) -> ConCompGeneratorState:
        return ConCompGeneratorState(
            max_vertices=merged["max_vertices"],
            min_vertices=merged["min_vertices"],
            edge_p=merged["edge_p"],
            queries_count=merged["queries_count"],
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
        allowed = _concomp_generation_arg_names()
        merged = self.merged_args_dict(common_args)
        for k, v in overrides.items():
            if k in allowed:
                merged[k] = v
        return _concomp_sample_from_merged(gen, merged)

    def evaluate(self, **kwargs):
        return {"task": self.task_name(), "status": "delegated", **kwargs}


def generate_concomp_examples(state: ConCompGeneratorState):
    def build_rng(seed: int):
        return np.random.default_rng(seed)

    def sample_fn(s: ConCompGeneratorState, rng):
        merged = {k: s[k] for k in _CONCOMP_SAMPLE_KEYS}
        result = _concomp_sample_from_merged(rng, merged)
        return {
            "text": result["text"],
            "loss_mask": result["loss_mask"],
            "encoding_format": result.get("encoding_format", "edges_list"),
        }

    yield from BaseDataGenerator.iterate_stateful_samples(state, build_rng=build_rng, sample_fn=sample_fn)


ConCompDataGenerator._iterate_examples_fn = generate_concomp_examples
