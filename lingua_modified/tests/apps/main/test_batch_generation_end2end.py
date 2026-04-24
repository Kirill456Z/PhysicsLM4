"""End-to-end: one synthetic training batch from ``depo_debug.yaml`` config, saved for inspection."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from apps.main.train_args import TrainArgs, prepare_train_args, validate_train_args
from data_synthetic_pretrain.dataloader.dataloader import build_dataloader

DEPO_DEBUG_YAML = (
    Path(__file__).resolve().parents[3]
    / "apps"
    / "main"
    / "configs"
    / "depo_debug.yaml"
)


def _merge_train_args_like_train_py(config_path: Path) -> TrainArgs:
    """Same OmegaConf merge as ``apps.main.train.main`` (file overrides defaults only)."""
    file_cfg = OmegaConf.load(config_path)
    default_cfg = OmegaConf.structured(TrainArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg)
    return OmegaConf.to_object(cfg)


class TestBatchGenerationE2E:
    def test_batch_generation_end2end(self, tmp_path: Path) -> None:
        args = _merge_train_args_like_train_py(DEPO_DEBUG_YAML)
        # Smaller, faster e2e; structure matches train (``prepare_train_args`` syncs into formatting).
        args.data.batch_size = 2
        args.data.n_workers = 1
        args.data.prefetch_size = 4
        prepare_train_args(args)
        validate_train_args(args, args.model.vocab_size)

        with build_dataloader(
            args.synthetic_tasks_generation_args,
            args.synthetic_tasks_formatting_args,
        ) as data_iter:
            batch_np, dl_state = next(data_iter)

        bsz, seqlen, pair = batch_np.shape
        assert bsz == args.data.batch_size
        assert seqlen == args.data.seq_len
        assert pair == 2
        assert np.issubdtype(batch_np.dtype, np.integer)

        out_npz = tmp_path / "e2e_training_batch.npz"
        if path_override := os.environ.get("E2E_BATCH_NPZ_PATH"):
            out_npz = Path(path_override)
            out_npz.parent.mkdir(parents=True, exist_ok=True)

        meta_path = out_npz.with_suffix(".meta.json")
        np.savez_compressed(
            out_npz,
            batch=batch_np,
            sampled_batches=int(dl_state["sampled_batches"]),
        )
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "batch_shape": [bsz, seqlen, pair],
                    "batch_dtype": str(batch_np.dtype),
                    "seq_len": args.data.seq_len,
                    "batch_size": args.data.batch_size,
                    "pad_token": args.data.pad_token,
                    "no_train_label_token": args.data.no_train_label_token,
                    "input_ids_slice_dim": 0,
                    "labels_slice_dim": 1,
                },
                f,
                indent=2,
            )
        assert out_npz.is_file()
        assert meta_path.is_file()

        loaded = np.load(out_npz)
        try:
            assert np.array_equal(loaded["batch"], batch_np)
            assert int(loaded["sampled_batches"]) == 1
        finally:
            loaded.close()
