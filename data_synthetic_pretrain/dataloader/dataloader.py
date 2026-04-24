from __future__ import annotations

import contextlib
from multiprocessing import Event, Process, Queue
from multiprocessing.synchronize import Event as EventClass
from queue import Empty, Full
from typing import Iterator, List

import numpy as np
from pydantic import BaseModel

from data_synthetic_pretrain.dataloader.data_generation_args import (
    SyntheticTasksFormattingArgs,
    SyntheticTasksGenerationArgs,
)
from data_synthetic_pretrain.tasks import SYNTHETIC_TASKS
from data_synthetic_pretrain.tasks.base_task import BaseSynteticTaskGenerator
from data_synthetic_pretrain.tasks.models import SynteticTask


class DataloaderState(BaseModel):
    sampled_batches: int = 0


def _batched_consume_buffer(
    producers: list[Process],
    queue: Queue,
    batch_size: int,
) -> Iterator[np.ndarray]:
    def any_alive() -> bool:
        return any(p.exitcode is None for p in producers)

    while any_alive():
        batch: list[np.ndarray] = []
        while len(batch) < batch_size:
            if not any_alive():
                break
            try:
                batch.append(queue.get(timeout=0.1))
            except Empty:
                pass
        if len(batch) == batch_size:
            yield np.stack(batch, axis=0)
        elif not any_alive():
            break

    raise RuntimeError(
        "Data loader quit unexpectedly, real error has been raised previously"
    )


class SyntheticDataLoader:
    def __init__(
        self,
        generators: List[BaseSynteticTaskGenerator],
        weights: list[float],
        formatting_args: SyntheticTasksFormattingArgs,
    ):
        if len(generators) != len(weights):
            raise ValueError("Number of generators and weights must be the same")
        if formatting_args.n_workers < 1:
            raise ValueError("n_workers must be at least 1")
        self.generators = generators
        weights_arr = np.array(weights) / np.sum(weights)
        self.weights = weights_arr
        self.formatting_args = formatting_args
        self.state = DataloaderState()

    def _format_sample(self, sample: SynteticTask) -> np.ndarray:
        context = np.array(sample.context)
        loss_mask = np.array(sample.loss_mask)
        if context.shape[0] > self.formatting_args.seq_len:
            context = context[: self.formatting_args.seq_len]
            loss_mask = loss_mask[: self.formatting_args.seq_len]
        else:
            context = np.pad(
                context,
                (0, self.formatting_args.seq_len - context.shape[0]),
                mode="constant",
                constant_values=self.formatting_args.pad_token,
            )
            loss_mask = np.pad(
                loss_mask,
                (0, self.formatting_args.seq_len - loss_mask.shape[0]),
                mode="constant",
                constant_values=0,
            )
        return np.stack([context, loss_mask], axis=1)

    def _sample_and_format(self) -> np.ndarray:
        generator = np.random.choice(self.generators, p=self.weights)
        sample = generator.generate()
        return self._format_sample(sample)

    def _producer_main_loop(self, queue: Queue, stop_event: EventClass) -> None:
        while not stop_event.is_set():
            sample = self._sample_and_format()
            while not stop_event.is_set():
                try:
                    queue.put(sample, timeout=0.1)
                    break
                except Full:
                    pass

    @staticmethod
    def _producer_process_entry(
        loader: SyntheticDataLoader,
        queue: Queue,
        stop_event: EventClass,
    ) -> None:
        loader._producer_main_loop(queue, stop_event)

    @contextlib.contextmanager
    def produce_async_batches(self):
        prefetch_size = self.formatting_args.prefetch_size
        batch_size = self.formatting_args.batch_size
        n_workers = self.formatting_args.n_workers
        queue: Queue = Queue(maxsize=prefetch_size)
        stop_event = Event()
        producers: list[Process] = []
        for _ in range(n_workers):
            p = Process(
                target=SyntheticDataLoader._producer_process_entry,
                args=(self, queue, stop_event),
            )
            p.start()
            producers.append(p)
        raw = _batched_consume_buffer(producers, queue, batch_size)

        def batches_with_state():
            try:
                for batch_arr in raw:
                    self.state.sampled_batches += 1
                    yield (batch_arr, self.state.model_copy())
            finally:
                raw.close()

        consumer = batches_with_state()
        try:
            yield consumer
        finally:
            stop_event.set()
            consumer.close()
            for p in producers:
                p.join(timeout=0.2)
                if p.exitcode is None:
                    p.kill()

    def __iter__(self) -> Iterator[tuple[np.ndarray, DataloaderState]]:
        """Synchronous infinite iterator over ``(batch, state)`` (no prefetch queue)."""
        while True:
            rows = [
                self._sample_and_format()
                for _ in range(self.formatting_args.batch_size)
            ]
            batch = np.stack(rows, axis=0)
            self.state.sampled_batches += 1
            yield (batch, self.state.model_copy())


def build_dataloader(
    synthetic_tasks_generation_args: SyntheticTasksGenerationArgs,
    synthetic_tasks_formatting_args: SyntheticTasksFormattingArgs,
):
    generators = []
    weights = []
    for generation_args in synthetic_tasks_generation_args.synthetic_tasks:
        task_generator = SYNTHETIC_TASKS[generation_args.task_name].build_from_dict(
            generation_args.generation_args
        )
        generators.append(task_generator)
        weights.append(generation_args.weight or 1.0)
    dataloader = SyntheticDataLoader(
        generators=generators,
        weights=weights,
        formatting_args=synthetic_tasks_formatting_args,
    )
    return dataloader.produce_async_batches()
