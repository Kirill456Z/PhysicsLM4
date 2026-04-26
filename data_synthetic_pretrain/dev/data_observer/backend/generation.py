from __future__ import annotations
from collections import defaultdict
import numpy as np
from typing import Any

from data_synthetic_pretrain.tasks import SYNTHETIC_TASKS
from data_synthetic_pretrain.tasks.depo import DepoSynteticTask
from data_synthetic_pretrain.dataloader.dataloader import SyntheticDataLoader
from data_synthetic_pretrain.dataloader.data_generation_args import SyntheticTasksFormattingArgs
from data_synthetic_pretrain.graph.graph import Graph
from data_synthetic_pretrain.graph.models import NodeWord, EncodingConfig

from config import ObserverConfig


def serialize_graph(graph: Graph) -> dict[str, Any]:
    node_index = {node: i for i, node in enumerate(graph.nodes)}
    edges = [
        {"from": node_index[src], "to": node_index[dst]}
        for src, neighbors in graph.edges.items()
        for dst in neighbors
    ]
    return {
        "nodes": [{"tokens": list(n.tokens)} for n in graph.nodes],
        "edges": edges,
        "n_nodes": graph.n,
    }


def _make_loader(config: ObserverConfig, generator) -> SyntheticDataLoader:
    return SyntheticDataLoader(
        generators=[generator],
        weights=[1.0],
        formatting_args=SyntheticTasksFormattingArgs(
            batch_size=config.formatting.batch_size,
            seq_len=config.formatting.seq_len,
            pad_token=config.formatting.pad_token,
            no_train_label_token=config.formatting.no_train_label_token,
            n_workers=1,
            prefetch_size=1,
        ),
    )


def _sample_to_dict(sample, formatted: np.ndarray, task_name: str, gen_args: dict) -> dict:
    context_padded = formatted[:, 0].tolist()
    labels = formatted[:, 1].tolist()

    base_vocab_size = gen_args.get("graph_generator_config", {}).get("base_vocab_size", 4)

    result: dict[str, Any] = {
        "graph": serialize_graph(sample.graph),
        "task_index": sample.task_index,
        "task_name": task_name,
        "base_vocab_size": base_vocab_size,
        "context": sample.context,
        "loss_mask": sample.loss_mask,
        "context_padded": context_padded,
        "labels": labels,
        "task_specific": None,
        "answer_start_index": int(sample.answer_start_index),
    }

    if isinstance(sample, DepoSynteticTask):
        asi = int(sample.answer_start_index)
        result["task_specific"] = {
            "type": "depo",
            "query_nodes": [list(n.tokens) for n in sample.query_nodes],
            "answer_nodes": [list(n.tokens) for n in sample.answer_nodes],
            "num_hops": [int(h) for h in sample.num_hops],
            "answer_start_index": asi,
        }

    return result


def generate_sample(config: ObserverConfig, task_name: str) -> dict:
    task_cfg = next((t for t in config.tasks if t.task_name == task_name), None)
    if task_cfg is None:
        raise ValueError(f"Task '{task_name}' not found in config")
    if task_name not in SYNTHETIC_TASKS:
        raise ValueError(
            f"Task '{task_name}' not implemented (available: {list(SYNTHETIC_TASKS.keys())})"
        )

    generator = SYNTHETIC_TASKS[task_name].build_from_dict(task_cfg.generation_args)
    loader = _make_loader(config, generator)
    sample = generator.generate()
    formatted = loader._format_sample(sample)
    return _sample_to_dict(sample, formatted, task_name, task_cfg.generation_args)


def generate_batch(config: ObserverConfig, task_name: str, batch_size: int) -> list[dict]:
    task_cfg = next((t for t in config.tasks if t.task_name == task_name), None)
    if task_cfg is None:
        raise ValueError(f"Task '{task_name}' not found in config")
    if task_name not in SYNTHETIC_TASKS:
        raise ValueError(f"Task '{task_name}' not implemented")

    generator = SYNTHETIC_TASKS[task_name].build_from_dict(task_cfg.generation_args)
    loader = _make_loader(config, generator)
    results = []
    for _ in range(batch_size):
        sample = generator.generate()
        formatted = loader._format_sample(sample)
        results.append(_sample_to_dict(sample, formatted, task_name, task_cfg.generation_args))
    return results


def _rebuild_graph(graph_nodes: list[list[int]], graph_edges: list[dict[str, int]], gen_args: dict[str, Any]) -> Graph:
    ec_raw = gen_args.get("graph_generator_config", {}).get("encoding_config", {})
    encoding_config = EncodingConfig.model_validate(ec_raw)

    nodes = [NodeWord(tokens=tuple(t)) for t in graph_nodes]
    node_by_idx = {i: n for i, n in enumerate(nodes)}
    edges: dict[NodeWord, list[NodeWord]] = defaultdict(list)
    for edge in graph_edges:
        edges[node_by_idx[edge["from"]]].append(node_by_idx[edge["to"]])
    return Graph(nodes=nodes, edges=dict(edges), adj_list_encoding_config=encoding_config)


def _to_python(obj: Any) -> Any:
    """Recursively convert numpy scalars / arrays to Python natives."""
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def evaluate_generation(
    config: ObserverConfig,
    task_name: str,
    generation: list[int],
    task_index: int,
    context: list[int],
    loss_mask: list[int],
    answer_start_index: int,
    query_nodes: list[list[int]],
    answer_nodes: list[list[int]],
    num_hops: list[int],
    graph_nodes: list[list[int]],
    graph_edges: list[dict[str, int]],
) -> dict[str, Any]:
    task_cfg = next((t for t in config.tasks if t.task_name == task_name), None)
    if task_cfg is None:
        raise ValueError(f"Task '{task_name}' not found in config")
    if task_name not in SYNTHETIC_TASKS:
        raise ValueError(f"Task '{task_name}' not implemented")

    generator = SYNTHETIC_TASKS[task_name].build_from_dict(task_cfg.generation_args)
    graph = _rebuild_graph(graph_nodes, graph_edges, task_cfg.generation_args)

    task = DepoSynteticTask(
        task_index=task_index,
        context=context,
        loss_mask=loss_mask,
        graph=graph,
        query_nodes=[NodeWord(tokens=tuple(t)) for t in query_nodes],
        answer_nodes=[NodeWord(tokens=tuple(t)) for t in answer_nodes],
        num_hops=num_hops,
        answer_start_index=answer_start_index,
    )

    raw_metrics = generator.evaluate(task, generation)
    return _to_python(raw_metrics)
