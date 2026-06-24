"""
Canon Layer Snapshot Script

Loads a trained model from a checkpoint, runs a single batch of auto-generated
Depo data, and saves canon layer weights and inputs to disk.

Output directory structure:
    /scratch/zemlians/physics4lm/canon_snapshots/{checkpoint_version}/
        {layer_idx}{a,b,c,d}_weights.pt   - canon conv1d weights
        {layer_idx}{a,b,c,d}_inputs.pt    - inputs to the canon layer

Usage:
    python -m lingua.log_from_trained \
        --config apps/main/configs/depo_debug.yaml \
        --checkpoint-version 1.2.3
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
from omegaconf import OmegaConf

from apps.main.transformer import LMTransformer, LMTransformerArgs
from lingua.args import dataclass_from_dict
from lingua.checkpoint import (
    CONSOLIDATE_FOLDER,
    CONSOLIDATE_NAME,
    RE_FOLDER,
    consolidate_checkpoints,
)
from lingua.data import DataArgs
from data_synthetic_pretrain.base_data_generator import BaseDataGenerator
from data_synthetic_pretrain.tasks.depo import DepoDataGenerator, DepoGenerationArgs
from lingua.tokenizer import MockTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SNAPSHOT_BASE = "/scratch/zemlians/physics4lm/canon_snapshots"
CKPT_BASE = "/scratch/zemlians/physics4lm/depo_checkpoints"


# ---------------------------------------------------------------------------
# Model loading (mirrors eval_depo.py, no distributed required)
# ---------------------------------------------------------------------------

def load_model_from_consolidated(consolidate_path: str):
    """Load model from a consolidated checkpoint (.pth)."""
    consolidate_path = Path(consolidate_path)

    with open(consolidate_path / "params.json", "r") as f:
        params = json.load(f)

    model_params = params.get("model", params)
    model_args = dataclass_from_dict(LMTransformerArgs, model_params)

    model = LMTransformer(model_args)

    checkpoint = torch.load(
        consolidate_path / CONSOLIDATE_NAME,
        map_location="cpu",
        weights_only=True,
    )
    model_state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(model_state_dict, strict=True)
    model = model.cuda()
    model.eval()

    return model, model_args, params


def find_latest_step_folder(ckpt_dir: str) -> Optional[Path]:
    """Find the latest step folder inside a checkpoint directory."""
    ckpt_path = Path(ckpt_dir)
    if not ckpt_path.exists():
        return None
    folders = [
        p for p in ckpt_path.iterdir()
        if p.is_dir() and re.match(RE_FOLDER, p.name)
    ]
    if not folders:
        return None
    folders.sort(key=lambda p: int(re.findall(r"\d+", p.name)[-1]))
    return folders[-1]


def resolve_checkpoint(checkpoint_version: str) -> Path:
    """
    Resolve a checkpoint version to a consolidated checkpoint path.
    Consolidates the DCP checkpoint if needed.
    """
    ckpt_dir = Path(CKPT_BASE) / f"v{checkpoint_version}" / "checkpoints"
    logger.info(f"Looking for checkpoints in {ckpt_dir}")

    step_folder = find_latest_step_folder(str(ckpt_dir))
    if step_folder is None:
        raise FileNotFoundError(f"No step folders found in {ckpt_dir}")
    logger.info(f"Using latest step folder: {step_folder.name}")

    # Check if already consolidated
    consolidate_path = step_folder / CONSOLIDATE_FOLDER
    if consolidate_path.exists() and (consolidate_path / CONSOLIDATE_NAME).exists():
        logger.info(f"Found existing consolidated checkpoint at {consolidate_path}")
        return consolidate_path

    # Check if it's a direct consolidated folder (has params.json + .pth)
    if (step_folder / "params.json").exists() and next(step_folder.glob("*.pth"), None):
        logger.info(f"Step folder is already consolidated: {step_folder}")
        return step_folder

    # Need to consolidate DCP -> .pth
    logger.info(f"Consolidating DCP checkpoint at {step_folder} ...")
    consolidate_path = consolidate_checkpoints(str(step_folder))
    logger.info(f"Consolidated to {consolidate_path}")
    return Path(consolidate_path)


# ---------------------------------------------------------------------------
# Canon input capture via forward-method wrapping
# ---------------------------------------------------------------------------

def register_canon_hooks(model) -> Dict[str, list]:
    """
    Wrap forward methods of all canon layers so that we capture the input
    tensor (x) that is passed to each canon layer during a forward pass.

    Returns a dict that will be populated with captured inputs after the
    forward pass, keyed like "0a", "0b", "3c", etc.
    """
    captured_inputs: Dict[str, torch.Tensor] = {}
    originals = []  # keep references so we can restore later

    for layer_idx, layer in enumerate(model.layers):
        canon_locations = [
            ("a", getattr(layer, "canonA", None)),
            ("b", getattr(layer.attention, "canonB", None) if hasattr(layer, "attention") else None),
            ("c", getattr(layer, "canonC", None)),
            ("d", getattr(layer.feed_forward, "canonD", None) if hasattr(layer, "feed_forward") else None),
        ]
        for suffix, canon in canon_locations:
            if canon is None:
                continue
            name = f"{layer_idx}{suffix}"
            orig_forward = canon.forward

            # Use default-arg binding to close over the correct name/orig_forward
            def make_wrapper(n, orig):
                def capturing_forward(x, **kwargs):
                    captured_inputs[n] = x.detach().cpu()
                    return orig(x, **kwargs)
                return capturing_forward

            canon.forward = make_wrapper(name, orig_forward)
            originals.append((canon, orig_forward))

    return captured_inputs, originals


def restore_canon_forwards(originals):
    for canon, orig in originals:
        canon.forward = orig


# ---------------------------------------------------------------------------
# Collect canon weights
# ---------------------------------------------------------------------------

def collect_canon_weights(model) -> Dict[str, torch.Tensor]:
    """Extract canon layer weights from the model."""
    weights: Dict[str, torch.Tensor] = {}
    for layer_idx, layer in enumerate(model.layers):
        canon_locations = [
            ("a", getattr(layer, "canonA", None)),
            ("b", getattr(layer.attention, "canonB", None) if hasattr(layer, "attention") else None),
            ("c", getattr(layer, "canonC", None)),
            ("d", getattr(layer.feed_forward, "canonD", None) if hasattr(layer, "feed_forward") else None),
        ]
        for suffix, canon in canon_locations:
            if canon is None:
                continue
            name = f"{layer_idx}{suffix}"
            weights[name] = canon.weight.detach().cpu()
            if canon.bias is not None:
                weights[f"{name}_bias"] = canon.bias.detach().cpu()
    return weights


# ---------------------------------------------------------------------------
# Generate a single training batch
# ---------------------------------------------------------------------------

def generate_single_batch(full_config: dict):
    """
    Generate a single training batch using the Depo on-the-fly generator,
    matching the same pipeline used during training.
    """
    depo_cfg = full_config.get("depo_generation_args", {})
    data_cfg = full_config.get("data", {})

    data_args = dataclass_from_dict(DataArgs, data_cfg, strict=False)
    # Force synchronous loading for a single batch
    data_args.load_async = False

    gen_args = dataclass_from_dict(DepoGenerationArgs, depo_cfg, strict=False)
    generator = DepoDataGenerator(generation_args=gen_args)
    state = generator.init_dataloader_state(
        data_args=data_args,
        seed=full_config.get("seed", 42),
        rank=0,
        world_size=1,
        common_args=None,
    )

    with BaseDataGenerator.build_on_the_fly_dataloader(
        state,
        DepoDataGenerator._iterate_examples_fn,
        pad_token_id=251,
        enable_packing=False,
    ) as data_loader:
        loaded_batch, _ = next(data_loader)

    if len(loaded_batch) == 3:
        batch_np, mask_np, _ = loaded_batch
    else:
        batch_np, mask_np = loaded_batch

    batch = torch.tensor(batch_np, dtype=torch.long)
    mask = torch.tensor(mask_np, dtype=torch.long)
    return batch, mask


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Save canon layer snapshots from a trained checkpoint")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the training config YAML (e.g. apps/main/configs/depo_debug.yaml)")
    parser.add_argument("--checkpoint-version", type=str, required=True,
                        help="Checkpoint version string (e.g. 1.2.3)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory (default: SNAPSHOT_BASE/{version})")
    args = parser.parse_args()

    # --- resolve output dir ---
    output_dir = Path(args.output_dir) if args.output_dir else Path(SNAPSHOT_BASE) / args.checkpoint_version
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # --- load training config ---
    file_cfg = OmegaConf.load(args.config)
    full_config = OmegaConf.to_object(file_cfg)

    # --- resolve & load model ---
    consolidate_path = resolve_checkpoint(args.checkpoint_version)
    logger.info("Loading model …")
    model, model_args, _ = load_model_from_consolidated(str(consolidate_path))
    logger.info(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

    # --- collect weights ---
    logger.info("Collecting canon weights …")
    weights = collect_canon_weights(model)
    for name, w in weights.items():
        out_path = output_dir / f"{name}_weights.pt"
        torch.save(w, out_path)
        logger.info(f"  Saved {out_path}  shape={tuple(w.shape)}")

    # --- generate a single batch ---
    logger.info("Generating a single training batch …")
    batch, mask = generate_single_batch(full_config)
    logger.info(f"  batch shape: {tuple(batch.shape)},  mask shape: {tuple(mask.shape)}")

    # --- register hooks & run forward ---
    logger.info("Registering canon input hooks …")
    captured_inputs, originals = register_canon_hooks(model)

    input_ids = batch[:, :, 0].cuda()
    with torch.no_grad():
        _ = model(input_ids)

    restore_canon_forwards(originals)

    # --- save captured inputs ---
    logger.info("Saving captured canon inputs …")
    for name, tensor in captured_inputs.items():
        out_path = output_dir / f"{name}_inputs.pt"
        torch.save(tensor, out_path)
        logger.info(f"  Saved {out_path}  shape={tuple(tensor.shape)}")

    # --- summary ---
    logger.info("=" * 60)
    logger.info("Snapshot complete!")
    logger.info(f"  Checkpoint version : {args.checkpoint_version}")
    logger.info(f"  Output directory   : {output_dir}")
    logger.info(f"  Weight files       : {len(weights)}")
    logger.info(f"  Input files        : {len(captured_inputs)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
