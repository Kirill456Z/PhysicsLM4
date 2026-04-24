import pytest
from data_synthetic_pretrain.tasks.models import SynteticTask
from data_synthetic_pretrain.graph.graph import Graph
from data_synthetic_pretrain.graph.models import EncodingConfig, NodeWord, SpecialToken
from unittest.mock import MagicMock
from data_synthetic_pretrain.dataloader.data_generation_args import (
    SyntheticTasksFormattingArgs,
)
from data_synthetic_pretrain.tasks.base_task import BaseSynteticTaskGenerator
from data_synthetic_pretrain.dataloader.dataloader import SyntheticDataLoader
import numpy as np


def _minimal_graph() -> Graph:
    cfg = EncodingConfig(
        pair_sep=SpecialToken(token=25),
        node_to_neighbors_sep=SpecialToken(token=26),
    )
    n = NodeWord(tokens=(1,))
    return Graph(nodes=[n], edges={n: []}, adj_list_encoding_config=cfg)


class PicklableTaskStub:
    """Module-level class so instances are picklable under multiprocessing spawn."""

    def __init__(self, context, loss_mask, graph, task_index=0):
        self._context = list(context)
        self._loss_mask = list(loss_mask)
        self._graph = graph
        self._task_index = task_index

    def generate(self):
        return SynteticTask(
            context=list(self._context),
            loss_mask=list(self._loss_mask),
            task_index=self._task_index,
            graph=self._graph,
        )


class TestSyntheticDataLoader:
    @pytest.fixture
    def syntehtic_task(self):
        return SynteticTask(
            context=[1, 2, 3, 4, 5],
            loss_mask=[1, 1, 1, 1, 1],
            task_index=0,
            graph=_minimal_graph(),
        )

    @pytest.fixture
    def formatting_args(self):
        return SyntheticTasksFormattingArgs(
            batch_size=32,
            seq_len=2048,
            pad_token=0,
        )

    @pytest.fixture
    def syntehtic_task_generator(self):
        return MagicMock(spec=BaseSynteticTaskGenerator)

    @pytest.fixture
    def dataloader(self, syntehtic_task_generator, formatting_args):
        return SyntheticDataLoader(
            generators=[syntehtic_task_generator],
            weights=[1.0],
            formatting_args=formatting_args,
        )

    def test_format_sample(self, syntehtic_task, dataloader):
        formatted_sample = dataloader._format_sample(syntehtic_task)
        assert formatted_sample.shape == (2048, 2)
        context = formatted_sample[:, 0]
        labels = formatted_sample[:, 1]
        ignore = dataloader.formatting_args.no_train_label_token
        assert np.all(context[: len(syntehtic_task.context)] == np.array(syntehtic_task.context))
        assert np.all(context[len(syntehtic_task.context) :] == 0)
        # labels: roll(context) with position 0 and all non-train positions set to ignore index
        n = len(syntehtic_task.context)
        padded = np.zeros(2048, dtype=context.dtype)
        padded[:n] = syntehtic_task.context
        m = np.zeros(2048, dtype=int)
        m[:n] = syntehtic_task.loss_mask
        m = np.roll(m, 1)
        m[0] = 0
        expected = np.roll(padded, 1)
        expected[0] = dataloader.formatting_args.pad_token
        expected[m == 0] = ignore
        assert np.array_equal(labels, expected)

    def test_sync_iter_batch_shape(self, syntehtic_task, dataloader):
        dataloader.generators[0].generate.return_value = syntehtic_task
        batch, state = next(iter(dataloader))
        assert batch.shape == (
            dataloader.formatting_args.batch_size,
            dataloader.formatting_args.seq_len,
            2,
        )
        assert state.sampled_batches == 1
        assert dataloader.state.sampled_batches == 1
        it = iter(dataloader)
        _, state2 = next(it)
        _, state3 = next(it)
        assert state2.sampled_batches == 2
        assert state3.sampled_batches == 3

    def test_async_batched_batches_shape(self):
        fmt = SyntheticTasksFormattingArgs(
            batch_size=4,
            seq_len=8,
            pad_token=0,
            prefetch_size=8,
            n_workers=2,
        )
        stub = PicklableTaskStub([1, 2, 3], [1, 1, 1], _minimal_graph())
        loader = SyntheticDataLoader(generators=[stub], weights=[1.0], formatting_args=fmt)
        with loader.produce_async_batches() as it:
            batch, state = next(it)
        assert batch.shape == (4, 8, 2)
        assert state.sampled_batches == 1
        assert loader.state.sampled_batches == 1

    def test_n_workers_must_be_positive(self):
        fmt = SyntheticTasksFormattingArgs(n_workers=0)
        with pytest.raises(ValueError, match="n_workers"):
            SyntheticDataLoader(
                generators=[PicklableTaskStub([1], [1], _minimal_graph())],
                weights=[1.0],
                formatting_args=fmt,
            )
