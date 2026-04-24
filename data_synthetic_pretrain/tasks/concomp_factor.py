"""ConCompFactor synthetic dataset — one node per connected component.

Format:
    <TASK_CONCOMP_FACTOR> src1 dst1 src2 dst2 ... <CONCOMP_FACTOR_ANS> r1 r2 ... <EOS>

- Graph is undirected and represented as an edge list.
- The target is a factorization of connectivity components:
  exactly one representative node from each component.
- Representatives are chosen deterministically as the lexicographically smallest
  node word in each component (shorter word is smaller; for equal length compare
  token-by-token using integer order) to avoid multiple correct textual outputs.
- Loss mask: 1 for answer-body tokens (after CONCOMP_FACTOR_ANS up to and
  including EOS), 0 elsewhere.
"""

import os
import sys
from dataclasses import dataclass, fields
from typing import Any, TypedDict

import numpy as np

_PRETRAIN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PRETRAIN_ROOT not in sys.path:
    sys.path.insert(0, _PRETRAIN_ROOT)

from shared_vocab import vocab_layout, split_into_words, unified_vocab_size
try:
    from .concomp import sample_graph_instance
except ImportError:
    # Allows running this file directly without package context.
    from concomp import sample_graph_instance
from data_synthetic_pretrain.graph import AdjacencyTokens, Graph
from data_synthetic_pretrain.tasks.config import sample_encoding_format
from data_synthetic_pretrain.base_data_generator import (
    BaseDataGenerator,
    CommonSyntheticGenerationArgs,
)


@dataclass
class ConCompFactorGenerationArgs(CommonSyntheticGenerationArgs):
    max_vertices: int = 50
    min_vertices: int = 3
    edge_p: float = 0.4
    train_on_all_tokens: bool = False


def concomp_factor_vocab_size(base_vocab_size: int = 4, max_hops: int = 0) -> int:
    """Return the minimum vocab size for ConCompFactor with the given base vocab."""
    return unified_vocab_size(base_vocab_size, max_hops)


def _connected_components_indices(adj: list[list[int]]) -> list[list[int]]:
    """Return connected components as lists of vertex indices."""
    n = len(adj)
    seen = [False] * n
    components: list[list[int]] = []

    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        component: list[int] = []
        while stack:
            u = stack.pop()
            component.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        components.append(sorted(component))

    return components


def _lexicographic_word_key(word: list[int]) -> tuple[int, tuple[int, ...]]:
    """Sort key: shorter words first, then token-by-token integer order."""
    return (len(word), tuple(word))


def generate_concomp_factor_sample(
    rng,
    max_vertices: int,
    min_vertices: int,
    edge_p: float,
    base_vocab_size: int = 4,
    min_token_length: int = 4,
    max_token_length: int = 6,
    encoding_format: str = "edges_list",
) -> dict:
    """Generate one ConCompFactor sample."""
    layout = vocab_layout(base_vocab_size)
    TASK_CONCOMP_FACTOR = layout["TASK_CONCOMP_FACTOR"]
    CONCOMP_FACTOR_ANS = layout["CONCOMP_FACTOR_ANS"]
    EOS = layout["EOS"]
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
        directed=False,
    )

    graph = Graph.from_edges(vertices, edge_pairs, directed=False)
    chosen_encoding = sample_encoding_format(rng, encoding_format)

    components = _connected_components_indices(adj)
    representative_indices = [
        min(component, key=lambda idx: _lexicographic_word_key(vertices[idx]))
        for component in components
    ]

    tokens = [TASK_CONCOMP_FACTOR]
    if chosen_encoding == "edges_list":
        tokens += graph.edges_list()
    elif chosen_encoding == "adjacency_list":
        tokens += graph.adjacency_list(adjacency_tokens)
    else:
        raise ValueError(f"Unsupported encoding format '{chosen_encoding}'")
    mask = [0] * len(tokens)

    tokens.append(CONCOMP_FACTOR_ANS)
    mask.append(0)

    for idx in representative_indices:
        tokens += vertices[idx]
        mask += [1] * len(vertices[idx])

    tokens.append(EOS)
    mask.append(1)

    assert len(tokens) == len(mask)
    return {"text": tokens, "loss_mask": mask, "encoding_format": chosen_encoding}


def _concomp_factor_generation_arg_names() -> frozenset[str]:
    return frozenset(f.name for f in fields(ConCompFactorGenerationArgs))


_CONCOMP_FACTOR_SAMPLE_KEYS = (
    "max_vertices",
    "min_vertices",
    "edge_p",
    "base_vocab_size",
    "min_token_length",
    "max_token_length",
    "encoding_format",
)


def _concomp_factor_sample_from_merged(rng, merged: dict):
    return generate_concomp_factor_sample(
        rng,
        **{k: merged[k] for k in _CONCOMP_FACTOR_SAMPLE_KEYS},
    )


def parse_concomp_factor_tokens(tokens: list[int], base_vocab_size: int = 4):
    """Parse a ConCompFactor token sequence.

    Returns:
        (valid, edge_words, representative_words)
    """
    layout = vocab_layout(base_vocab_size)
    TASK_CONCOMP_FACTOR = layout["TASK_CONCOMP_FACTOR"]
    CONCOMP_FACTOR_ANS = layout["CONCOMP_FACTOR_ANS"]
    EOS = layout["EOS"]

    if not tokens or tokens[0] != TASK_CONCOMP_FACTOR or tokens[-1] != EOS:
        return False, None, None

    try:
        idx_ans = tokens.index(CONCOMP_FACTOR_ANS)
    except ValueError:
        return False, None, None

    edge_words = split_into_words(tokens[1:idx_ans], base_vocab_size)
    rep_words = split_into_words(tokens[idx_ans + 1 : -1], base_vocab_size)

    if len(edge_words) % 2 != 0:
        return False, None, None

    edge_word_pairs = []
    for i in range(0, len(edge_words), 2):
        edge_word_pairs.append((tuple(edge_words[i]), tuple(edge_words[i + 1])))

    return True, edge_word_pairs, [tuple(w) for w in rep_words]


class ConCompFactorGeneratorState(TypedDict):
    max_vertices: int
    min_vertices: int
    edge_p: float
    base_vocab_size: int
    min_token_length: int
    max_token_length: int
    encoding_format: str
    seed: int
    sample_count: int


class ConCompFactorDataGenerator(BaseDataGenerator):
    NAME = "concomp_factor"
    ArgsCls = ConCompFactorGenerationArgs

    def _build_generator_state_for_loader(
        self,
        merged: dict,
        *,
        seed: int,
        sample_count: int,
    ) -> ConCompFactorGeneratorState:
        return ConCompFactorGeneratorState(
            max_vertices=merged["max_vertices"],
            min_vertices=merged["min_vertices"],
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
        allowed = _concomp_factor_generation_arg_names()
        merged = self.merged_args_dict(common_args)
        for k, v in overrides.items():
            if k in allowed:
                merged[k] = v
        return _concomp_factor_sample_from_merged(gen, merged)

    def evaluate(self, **kwargs):
        return {"task": self.task_name(), "status": "delegated", **kwargs}


def generate_concomp_factor_examples(state: ConCompFactorGeneratorState):
    def build_rng(seed: int):
        return np.random.default_rng(seed)

    def sample_fn(s: ConCompFactorGeneratorState, rng):
        merged = {k: s[k] for k in _CONCOMP_FACTOR_SAMPLE_KEYS}
        result = _concomp_factor_sample_from_merged(rng, merged)
        return {
            "text": result["text"],
            "loss_mask": result["loss_mask"],
            "encoding_format": result.get("encoding_format", "edges_list"),
        }

    yield from BaseDataGenerator.iterate_stateful_samples(state, build_rng=build_rng, sample_fn=sample_fn)


ConCompFactorDataGenerator._iterate_examples_fn = generate_concomp_factor_examples
