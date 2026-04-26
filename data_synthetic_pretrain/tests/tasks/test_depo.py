import io
import os
import pytest
from unittest.mock import patch

from data_synthetic_pretrain.graph.graph import Graph
from data_synthetic_pretrain.graph.models import (
    GraphGeneratorConfig,
    EncodingConfig,
    SpecialToken,
    EncodingFormat,
    NodeWord,
)
from unittest.mock import MagicMock
from data_synthetic_pretrain.tasks.depo import DepoGenerationArgs, DepoRefactored, DepoSynteticTask
from data_synthetic_pretrain.tasks.models import SynteticTask


class TestDepo:
    @pytest.fixture
    def graph_generator_config(self):
        return GraphGeneratorConfig(
            base_vocab_size=10,
            min_token_length=1,
            max_token_length=4,
            is_directed=True,
            encoding_config=EncodingConfig(
                pair_sep=SpecialToken(token=10),
                node_to_neighbors_sep=SpecialToken(token=11),
                format=EncodingFormat.EDGES_LIST,
            ),
            is_dag=True,
            edge_probability=0.5,
        )

    @pytest.fixture
    def depo_generation_args(self, graph_generator_config):
        return DepoGenerationArgs(
            task_index=99,
            graph_generator_config=graph_generator_config,
            max_hops=3,
            num_queries=2,
            query_token_base=100,
            max_nodes=40,
        )
    
    def test_context_num_queries_is_correct(self, depo_generation_args):
        depo_task_generator = DepoRefactored(depo_generation_args)
        task = depo_task_generator.generate()
        context = task.context
        queries = [token for token in context if token >= depo_generation_args.query_token_base]
        assert len(queries) == depo_generation_args.num_queries
    
    def test_context_task_index_in_context(self, depo_generation_args):
        depo_task_generator = DepoRefactored(depo_generation_args)
        task = depo_task_generator.generate()
        context = task.context
        assert context[0] == depo_generation_args.task_index
    
    def test_loss_mask_is_correct(self, depo_generation_args):
        depo_task_generator = DepoRefactored(depo_generation_args)
        task = depo_task_generator.generate()
        loss_mask = task.loss_mask
        enabled_context = [token for token, mask in zip(task.context, loss_mask) if mask == 1]
        expected_context = []
        for answer_token in task.answer_nodes:
            expected_context += list(answer_token.tokens)
        assert enabled_context == expected_context

    def test_depo_correctness_cyclic_graph(self, depo_generation_args):
        enc_cfg = depo_generation_args.graph_generator_config.encoding_config

        def fake_generate_graph(num_nodes):
            nodes = [NodeWord(tokens=(i,)) for i in range(10)]
            edges = {nodes[i]: [nodes[(i + 1) % 10]] for i in range(10)}
            return Graph(
                nodes=nodes,
                edges=edges,
                adj_list_encoding_config=enc_cfg,
            )

        depo_task_generator = DepoRefactored(depo_generation_args)
        with patch.object(depo_task_generator, "generate_graph", fake_generate_graph):
            task = depo_task_generator.generate()

        for i in range(len(task.answer_nodes)):
            q = task.query_nodes[i].tokens[0]
            h = int(task.num_hops[i])
            a = task.answer_nodes[i].tokens[0]
            assert a == (q + h) % 10
    
    @pytest.fixture
    def depo_task(self):
        return DepoSynteticTask(
            task_index=99,
            context=[],
            loss_mask=[],
            graph=MagicMock(spec=Graph),
            query_nodes=[NodeWord(tokens=(5, 11))],
            answer_nodes=[NodeWord(tokens=(6, 5, 11))],
            num_hops=[1],
            answer_start_index=8,
        )

    @pytest.mark.parametrize("generation, accuracy, prefix_accuracy", [
        ([6, 5, 11], 1.0, 1.0),
        ([1, 5, 12], 0.0, 0.0),
        ([6, 1, 12], 0.0, 0.3333333333333333),
        ([1, 5, 11], 0.0, 0.0),
        ([1, 2], 0.0, 0.0),
        ([6], 0.0, 0.33333333333333333),
        ([6, 5, 11, 12, 13], 1.0, 1.0),
        ([], 0.0, 0.0),
    ])
    def test_depo_eval_correct_answer(self, depo_generation_args, depo_task, generation, accuracy, prefix_accuracy):
        depo_task_generator = DepoRefactored(depo_generation_args)
        metrics = depo_task_generator.evaluate(depo_task, generation)
        print(metrics)
        assert metrics["hop_1/accuracy"] == accuracy
        assert metrics["hop_1/prefix_accuracy"] == prefix_accuracy


def _vpath_class(state: dict):
    """In-memory path for eval_dump_dir + hash; ``exists`` reads ``state['exists']``."""

    class VirtualPath:
        def __init__(self, *parts: str):
            self._parts = parts

        def __truediv__(self, other: str) -> "VirtualPath":
            return VirtualPath(*self._parts, str(other))

        def __fspath__(self) -> str:
            if not self._parts:
                return os.sep
            return os.path.join(os.sep, *self._parts)

        def exists(self) -> bool:
            return state["exists"]

        def mkdir(self, parents: bool = True, exist_ok: bool = False) -> None:
            pass

        def write_text(self, data: str, encoding: str = "utf-8") -> None:
            state["config_text"] = data

    return VirtualPath


class _StringIOStaysUsableAfterContext(io.StringIO):
    """``with open(..., "w")`` calls ``close()``; plain ``StringIO`` then rejects ``getvalue()``."""

    def close(self) -> None:
        pass


def _open_factory(state: dict, real_open=open):
    def fake_open(file, mode: str = "r", *args, **kwargs):
        p = os.fspath(file)
        p_norm = p.replace("\\", "/")
        if not p_norm.endswith("eval.jsonl"):
            return real_open(file, mode, *args, **kwargs)
        if "w" in mode:
            buf = _StringIOStaysUsableAfterContext()
            state["eval_write_buf"] = buf
            return buf
        if "r" in mode:
            return io.StringIO(state.get("eval_read_text", ""))
        return real_open(file, mode, *args, **kwargs)

    return fake_open


def _assert_tasks_equivalent(a: SynteticTask, b: SynteticTask) -> None:
    assert isinstance(a, SynteticTask)
    assert isinstance(b, SynteticTask)

    assert a.task_index == b.task_index
    assert a.context == b.context
    assert a.loss_mask == b.loss_mask
    assert a.answer_start_index == b.answer_start_index

    assert [n.tokens for n in a.graph.nodes] == [n.tokens for n in b.graph.nodes]
    assert all(isinstance(n, NodeWord) for n in a.graph.nodes)
    assert all(isinstance(n, NodeWord) for n in b.graph.nodes)

    def _edges_signature(task: SynteticTask) -> dict[tuple[int, ...], list[tuple[int, ...]]]:
        sig: dict[tuple[int, ...], list[tuple[int, ...]]] = {}
        for src, dsts in task.graph.edges.items():
            assert isinstance(src, NodeWord)
            assert isinstance(dsts, list)
            assert all(isinstance(d, NodeWord) for d in dsts)
            sig[src.tokens] = [d.tokens for d in dsts]
        return sig

    assert _edges_signature(a) == _edges_signature(b)

    if hasattr(a, "query_nodes") or hasattr(b, "query_nodes"):
        assert [n.tokens for n in a.query_nodes] == [n.tokens for n in b.query_nodes]
        assert all(isinstance(n, NodeWord) for n in a.query_nodes)
        assert all(isinstance(n, NodeWord) for n in b.query_nodes)

    if hasattr(a, "answer_nodes") or hasattr(b, "answer_nodes"):
        assert [n.tokens for n in a.answer_nodes] == [n.tokens for n in b.answer_nodes]
        assert all(isinstance(n, NodeWord) for n in a.answer_nodes)
        assert all(isinstance(n, NodeWord) for n in b.answer_nodes)

    if hasattr(a, "num_hops") or hasattr(b, "num_hops"):
        assert [int(x) for x in a.num_hops] == [int(x) for x in b.num_hops]


class TestDepoEvalSerialization:
    @pytest.fixture
    def graph_generator_config(self):
        return GraphGeneratorConfig(
            base_vocab_size=10,
            min_token_length=1,
            max_token_length=4,
            is_directed=True,
            encoding_config=EncodingConfig(
                pair_sep=SpecialToken(token=10),
                node_to_neighbors_sep=SpecialToken(token=11),
                format=EncodingFormat.EDGES_LIST,
            ),
            is_dag=True,
            edge_probability=0.5,
        )

    @pytest.fixture
    def depo_generation_args(self, graph_generator_config):
        return DepoGenerationArgs(
            task_index=99,
            graph_generator_config=graph_generator_config,
            max_hops=3,
            num_queries=2,
            query_token_base=100,
            max_nodes=40,
            eval_dump_dir="mem",
        )

    @pytest.fixture
    def two_depo_eval_tasks(self, depo_generation_args):
        enc = depo_generation_args.graph_generator_config.encoding_config
        nodes = [NodeWord(tokens=(i,)) for i in range(5)]
        edges = {nodes[i]: [nodes[(i + 1) % 5]] for i in range(5)}
        g = Graph(
            nodes=nodes,
            edges=edges,
            adj_list_encoding_config=enc,
        )
        t0 = DepoSynteticTask(
            task_index=99,
            context=[1, 2, 3],
            loss_mask=[0, 0, 0],
            graph=g,
            query_nodes=[NodeWord(tokens=(0,))],
            answer_nodes=[NodeWord(tokens=(2,))],
            num_hops=[1],
            answer_start_index=0,
        )
        t1 = DepoSynteticTask(
            task_index=99,
            context=[4, 5, 6, 7],
            loss_mask=[0, 1, 1, 0],
            graph=g,
            query_nodes=[NodeWord(tokens=(1,)), NodeWord(tokens=(2,))],
            answer_nodes=[NodeWord(tokens=(3,)), NodeWord(tokens=(4,))],
            num_hops=[2, 1],
            answer_start_index=2,
        )
        return [t0, t1]

    def test_eval_set_roundtrip_from_constructor_mocks(
        self, depo_generation_args, two_depo_eval_tasks
    ):
        state: dict = {"exists": False, "config_text": None, "eval_read_text": None}

        VPath = _vpath_class(state)
        fake_open = _open_factory(state)

        with (
            patch("data_synthetic_pretrain.tasks.base_task.Path", VPath),
            patch("builtins.open", fake_open),
            patch.object(
                DepoRefactored,
                "_generate_eval_set",
                return_value=two_depo_eval_tasks,
            ),
        ):
            gen_first = DepoRefactored(depo_generation_args)

        assert gen_first.get_eval_set() is two_depo_eval_tasks
        assert state.get("config_text") is not None
        wbuf = state.get("eval_write_buf")
        assert wbuf is not None
        captured = wbuf.getvalue()
        written_lines = [ln for ln in captured.splitlines() if ln.strip()]
        assert len(written_lines) == len(two_depo_eval_tasks)
        for line, expected in zip(written_lines, two_depo_eval_tasks):
            _assert_tasks_equivalent(SynteticTask.model_validate_json(line), expected)

        state["eval_read_text"] = captured
        state["exists"] = True

        with (
            patch("data_synthetic_pretrain.tasks.base_task.Path", VPath),
            patch("builtins.open", fake_open),
        ):
            gen_second = DepoRefactored(depo_generation_args)

        assert len(gen_second.get_eval_set()) == len(two_depo_eval_tasks)
        for before, after in zip(
            gen_first.get_eval_set(), gen_second.get_eval_set()
        ):
            _assert_tasks_equivalent(before, after)
