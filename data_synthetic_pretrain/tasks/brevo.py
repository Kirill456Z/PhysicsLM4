# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Author: Zeyuan Allen-Zhu
#
# Task Brevo: Mental reasoning breadth.
#
# Each instance is formatted as:
#   <TASK_BREVO> x1 y1 x2 y2 ... xm ym <BREVO_QUERY> q <BREVO_ANS> a1 a2 ... ap <EOS>
#
# 2m words define m directed edges xi → yi.  Upon a query vertex q the model outputs
# all vertices recursively reachable from q, in topological order (leaves first).
#
# Token layout is shared with Depo and ConComp — see data_synthetic_pretrain/shared_vocab.py.
# Node names are always multi-token words (Brevo2 only; Brevo1 is retired).

import os
import sys
import random
from dataclasses import dataclass, fields
from collections import defaultdict, deque
from typing import Any, Optional, TypedDict

_PRETRAIN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PRETRAIN_ROOT not in sys.path:
    sys.path.insert(0, _PRETRAIN_ROOT)

from shared_vocab import vocab_layout, generate_words_stdlib, split_into_words
from data_synthetic_pretrain.graph import AdjacencyTokens, Graph
from data_synthetic_pretrain.tasks.config import sample_encoding_format
from data_synthetic_pretrain.base_data_generator import (
    BaseDataGenerator,
    CommonSyntheticGenerationArgs,
)


class BrevoGeneratorState(TypedDict):
    max_nodes: int
    base_vocab_size: int
    min_token_length: int
    max_token_length: int
    encoding_format: str
    seed: int
    sample_count: int


class TopoSortDepthStats:
    def __init__(self, n, vocab_size=125, max_in=4):
        self.n = n
        self.vocab_size = vocab_size
        self.max_in = max_in

    def generate_dag(self, rng):
        nodes = rng.sample(range(1, self.vocab_size + 1), self.n)
        dag = defaultdict(list)
        out_degree = defaultdict(int)
        leaves = rng.randint(1, (len(nodes) - 1) // 4 + 1)
        for i in range(leaves, len(nodes)):
            tgt = nodes[i]
            possible_parents = [src for src in nodes[:i] if out_degree[src] < 4]
            if not possible_parents:
                continue
            num_parents = rng.randint(1, min(len(possible_parents), 4))
            parents = rng.sample(possible_parents, num_parents)
            for parent in parents:
                dag[tgt].append(parent)
                out_degree[parent] += 1
        return nodes, dag

    def subtree_from_query(self, dag, query):
        visited = set()
        stack = [query]
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                for parent in dag.get(node, []):
                    if parent not in visited:
                        stack.append(parent)
        filtered = defaultdict(list)
        for node in visited:
            for parent in dag.get(node, []):
                if parent in visited:
                    filtered[node].append(parent)
        for node in visited:
            _ = filtered[node]
        return filtered

    def topological_sort(self, dag, rng):
        indegree = {node: 0 for node in dag}
        for node in dag:
            for parent in dag[node]:
                indegree[parent] += 1
        queue = [node for node in dag if indegree[node] == 0]
        order = []
        while queue:
            node = queue.pop(rng.randint(0, len(queue) - 1))
            order.append(node)
            for parent in dag[node]:
                indegree[parent] -= 1
                if indegree[parent] == 0:
                    queue.append(parent)
        order.reverse()
        return order

    def compute_graph_depth(self, dag, query):
        distance = {query: 0}
        queue = deque([query])
        while queue:
            node = queue.popleft()
            for parent in dag[node]:
                if parent not in distance:
                    distance[parent] = distance[node] + 1
                    queue.append(parent)
        leaves = [node for node in dag if len(dag[node]) == 0]
        if not leaves:
            return 0
        return min(distance.get(leaf, float("inf")) for leaf in leaves if leaf in distance)

    def generate_sample(self, rng):
        nodes, dag = self.generate_dag(rng)
        start_index = max(len(nodes) * 3 // 4, len(nodes) - 1)
        candidate_nodes = nodes[start_index:]
        nonzero_degree_nodes = [node for node in candidate_nodes if len(dag[node]) > 0]
        query = rng.choice(nonzero_degree_nodes)
        subdag = self.subtree_from_query(dag, query)
        topo = self.topological_sort(subdag, rng)
        depth = self.compute_graph_depth(subdag, query)
        return dag, topo, depth

    def generate_tokens(
        self,
        rng,
        base_vocab_size=4,
        min_tlen=4,
        max_tlen=6,
        encoding_format: str = "edges_list",
    ):
        """Generate a tokenised Brevo sample.

        Returns:
            tokens: flat list of token IDs (unified vocabulary)
            labels: parallel list of 0/1 training labels (1 = answer tokens)
            depth:  BFS depth of the query sub-DAG
        """
        layout = vocab_layout(base_vocab_size)
        TASK_BREVO   = layout["TASK_BREVO"]
        BREVO_QUERY  = layout["BREVO_QUERY"]
        BREVO_ANS    = layout["BREVO_ANS"]
        EOS          = layout["EOS"]
        adjacency_tokens = AdjacencyTokens(
            node_to_neighbors=layout["ADJ_NODE_TO_NEIGHBORS"],
            pair_sep=layout["ADJ_PAIR_SEP"],
            no_neighbor=layout["ADJ_NO_NEIGHBOR"],
        )

        dag, topo, depth = self.generate_sample(rng)
        query = topo[-1]

        all_node_ids = sorted(set(dag.keys()) | {p for ps in dag.values() for p in ps})
        word_list = generate_words_stdlib(
            rng, len(all_node_ids), base_vocab_size, min_tlen, max_tlen
        )
        word_map = {nid: word_list[i] for i, nid in enumerate(all_node_ids)}

        edges = [(p, c) for c, ps in dag.items() for p in ps]
        rng.shuffle(edges)

        graph_nodes = [tuple(word_map[nid]) for nid in all_node_ids]
        node_to_idx = {nid: i for i, nid in enumerate(all_node_ids)}
        graph_edges = [(node_to_idx[p], node_to_idx[c]) for (p, c) in edges]
        graph = Graph(nodes=graph_nodes, edges=graph_edges, directed=True)
        chosen_encoding = sample_encoding_format(rng, encoding_format)

        # Context (label = 0): task type + encoded graph + query marker + query word + ans marker
        context = [TASK_BREVO]
        if chosen_encoding == "edges_list":
            context += graph.edges_list()
        elif chosen_encoding == "adjacency_list":
            context += graph.adjacency_list(adjacency_tokens)
        else:
            raise ValueError(f"Unsupported encoding format '{chosen_encoding}'")
        context += [BREVO_QUERY] + word_map[query] + [BREVO_ANS]

        # Answer (label = 1): topological-order words + EOS
        answer = []
        for node in topo:
            answer += word_map[node]
        answer.append(EOS)

        tokens = context + answer
        labels = [0] * len(context) + [1] * len(answer)
        assert len(tokens) == len(labels)
        return tokens, labels, depth, chosen_encoding

    @staticmethod
    def parse_tokens_multi(tokens, base_vocab_size=4):
        """Parse and validate a multi-token Brevo sequence.

        Returns:
            (valid: bool, query_word: tuple|None, answer_words: list[tuple]|None)
        """
        layout = vocab_layout(base_vocab_size)
        TASK_BREVO  = layout["TASK_BREVO"]
        BREVO_QUERY = layout["BREVO_QUERY"]
        BREVO_ANS   = layout["BREVO_ANS"]
        EOS         = layout["EOS"]

        if not tokens or tokens[0] != TASK_BREVO or tokens[-1] != EOS:
            return False, None, None
        try:
            idx_query = tokens.index(BREVO_QUERY)
            idx_ans   = tokens.index(BREVO_ANS)
        except ValueError:
            return False, None, None

        edge_words   = split_into_words(tokens[1:idx_query], base_vocab_size)
        query_words  = split_into_words(tokens[idx_query + 1:idx_ans], base_vocab_size)
        answer_words = split_into_words(tokens[idx_ans + 1:-1], base_vocab_size)

        if len(edge_words) % 2 != 0 or len(query_words) != 1:
            return False, None, None

        query_word = tuple(query_words[0])
        all_words  = set(tuple(w) for w in edge_words)
        all_words.add(query_word)
        word_to_id = {w: i for i, w in enumerate(sorted(all_words))}

        dag = defaultdict(list)
        for i in range(0, len(edge_words), 2):
            p_id = word_to_id[tuple(edge_words[i])]
            c_id = word_to_id[tuple(edge_words[i + 1])]
            dag[c_id].append(p_id)

        query_id = word_to_id[query_word]
        topo_ids = [word_to_id[tuple(w)] for w in answer_words]

        # BFS reachability from query
        reachable = set()
        stack = [query_id]
        while stack:
            node = stack.pop()
            reachable.add(node)
            for parent in dag.get(node, []):
                if parent not in reachable:
                    stack.append(parent)

        if set(topo_ids) != reachable:
            return False, query_word, [tuple(w) for w in answer_words]

        # Validate topological order
        seen = set()
        for node in topo_ids:
            for parent in dag.get(node, []):
                if parent not in seen:
                    return False, query_word, [tuple(w) for w in answer_words]
            seen.add(node)

        return True, query_word, [tuple(w) for w in answer_words]


def topsort_data(
    N,
    rng=None,
    base_vocab_size=4,
    min_tlen=4,
    max_tlen=6,
    encoding_format: str = "edges_list",
):
    """Generate one Brevo sample (convenience wrapper).

    Args:
        N: Maximum graph size; n is sampled from [3, N] with a power-law distribution.
        rng: Optional random.Random instance (defaults to Random(42)).
        base_vocab_size: Shared base vocabulary size.
        min_tlen / max_tlen: Word token-length bounds.

    Returns:
        dict with keys 0 (tokens), 'label' (loss mask), 'depth' (graph depth).
    """
    if rng is None:
        rng = random.Random(42)
    choices = list(range(3, N + 1))
    power, bias = 1, pow(N, 0.5)
    p = [1.0 / (pow(i, power) + bias + 1e-12) for i in choices]
    total = sum(p)
    p = [x / total for x in p]
    n = rng.choices(choices, weights=p)[0]

    topo = TopoSortDepthStats(n, vocab_size=N)
    tokens, labels, depth, chosen_encoding = topo.generate_tokens(
        rng,
        base_vocab_size=base_vocab_size,
        min_tlen=min_tlen,
        max_tlen=max_tlen,
        encoding_format=encoding_format,
    )
    return {0: tokens, "label": labels, "depth": depth, "encoding_format": chosen_encoding}


if __name__ == "__main__":
    _rng = random.Random(42)
    print(topsort_data(N=90, rng=_rng))
    print(topsort_data(N=60, rng=_rng))
    print(topsort_data(N=30, rng=_rng))


@dataclass
class BrevoGenerationArgs(CommonSyntheticGenerationArgs):
    max_nodes: int = 90
    train_on_all_tokens: bool = False


def _brevo_generation_arg_names() -> frozenset[str]:
    return frozenset(f.name for f in fields(BrevoGenerationArgs))


_BREVO_RNG_PARAM_KEYS = (
    "max_nodes",
    "base_vocab_size",
    "min_token_length",
    "max_token_length",
    "encoding_format",
)


def _topsort_from_merged(rng: random.Random, merged: dict):
    return topsort_data(
        N=merged["max_nodes"],
        rng=rng,
        base_vocab_size=merged["base_vocab_size"],
        min_tlen=merged["min_token_length"],
        max_tlen=merged["max_token_length"],
        encoding_format=merged["encoding_format"],
    )


class BrevoDataGenerator(BaseDataGenerator):
    NAME = "brevo"
    ArgsCls = BrevoGenerationArgs

    def __init__(self, seed: int = 42, generation_args: BrevoGenerationArgs | None = None):
        super().__init__(generation_args or BrevoGenerationArgs())
        self.rng = random.Random(seed)

    def _build_generator_state_for_loader(
        self,
        merged: dict,
        *,
        seed: int,
        sample_count: int,
    ) -> BrevoGeneratorState:
        return BrevoGeneratorState(
            max_nodes=merged["max_nodes"],
            base_vocab_size=merged["base_vocab_size"],
            min_token_length=merged["min_token_length"],
            max_token_length=merged["max_token_length"],
            encoding_format=merged["encoding_format"],
            seed=seed,
            sample_count=sample_count,
        )

    def init_generation_state(
        self,
        *,
        seed: int = 42,
        sample_count: int = 0,
        common_args: CommonSyntheticGenerationArgs | None = None,
        **overrides,
    ) -> BrevoGeneratorState:
        args = self.merged_args_dict(common_args)
        args.update(overrides)
        return BrevoGeneratorState(
            max_nodes=args["max_nodes"],
            base_vocab_size=args["base_vocab_size"],
            min_token_length=args["min_token_length"],
            max_token_length=args["max_token_length"],
            encoding_format=args["encoding_format"],
            seed=seed,
            sample_count=sample_count,
        )

    def generate(
        self,
        generation_state: Optional[BrevoGeneratorState] = None,
        *,
        common_args: CommonSyntheticGenerationArgs | None = None,
        **overrides: Any,
    ):
        """Generate one Brevo sample.

        Defaults come from ``self.generation_args`` (constructor), merged with
        ``common_args`` and optional ``overrides`` (``BrevoGenerationArgs`` fields only).
        """
        allowed = _brevo_generation_arg_names()
        if generation_state is not None:
            rng = random.Random(f"{generation_state['seed']}:{generation_state['sample_count']}")
            merged = {k: generation_state[k] for k in _BREVO_RNG_PARAM_KEYS}
            data = _topsort_from_merged(rng, merged)
            generation_state["sample_count"] += 1
            return data

        merged = self.merged_args_dict(common_args)
        for k, v in overrides.items():
            if k in allowed:
                merged[k] = v
        return _topsort_from_merged(self.rng, merged)

    def generate_batch(
        self,
        *,
        generation_state: BrevoGeneratorState,
        batch_size: int,
    ):
        return [self.generate(generation_state=generation_state) for _ in range(batch_size)]

    def evaluate(self, **kwargs):
        return {"task": self.task_name(), "status": "delegated", **kwargs}


def generate_brevo_examples(state: BrevoGeneratorState):
    def build_rng(seed: int):
        return random.Random(seed)

    def sample_fn(s: BrevoGeneratorState, rng: random.Random):
        distribution = list(range(3, s["max_nodes"] + 1))
        power, bias = 1, pow(s["max_nodes"], 0.5)
        raw_p = [1.0 / (pow(i, power) + bias + 1e-12) for i in distribution]
        total = sum(raw_p)
        weights = [x / total for x in raw_p]
        n = rng.choices(distribution, weights=weights)[0]
        topo_obj = TopoSortDepthStats(n, vocab_size=s["max_nodes"])
        tokens, labels, _depth, encoding = topo_obj.generate_tokens(
            rng,
            base_vocab_size=s["base_vocab_size"],
            min_tlen=s["min_token_length"],
            max_tlen=s["max_token_length"],
            encoding_format=s["encoding_format"],
        )
        return {"text": tokens, "loss_mask": labels, "encoding_format": encoding}

    yield from BaseDataGenerator.iterate_stateful_samples(state, build_rng=build_rng, sample_fn=sample_fn)


BrevoDataGenerator._iterate_examples_fn = generate_brevo_examples
