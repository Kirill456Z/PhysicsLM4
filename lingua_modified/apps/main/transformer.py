# Copyright (c) Meta Platforms, Inc. and affiliates.

# This file is adapted from the original implementation in:
# https://github.com/facebookresearch/lingua/
# Released under the BSD 3-Clause License by:
# Mathurin Videau*, Badr Youbi Idrissi*, Daniel Haziza, Luca Wehrstedt, Jade Copet, Olivier Teytaud, and David Lopez-Paz.
#
# Modifications made by Zeyuan Allen-Zhu include:
# - added z-loss support
# - default to sdpa attention.
#
# These modifications are licensed under the Apache 2.0 license, as stated in the root LICENSE file.
#

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask, BlockMask

from torch.distributed._tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
    PrepareModuleInput,
    parallelize_module,
)

from xformers.ops import fmha, AttentionBias
from lingua.transformer import (
    BaseTransformer,
    BaseTransformerArgs,
    RMSNorm,
    TiedLinear,
    cross_entropy,
)


def create_causal_mask(seqlen, attn_impl, sliding_window):
    if sliding_window is not None and attn_impl == "fmha":
        return fmha.attn_bias.LocalAttentionFromBottomRightMask(
            window_left=sliding_window - 1, window_right=0
        )
    elif attn_impl == "fmha":
        return fmha.attn_bias.LowerTriangularMask()
    elif attn_impl == "sdpa":
        return "causal"
    elif attn_impl == "flex_attention":
        return create_block_mask(causal_mask, None, None, seqlen, seqlen)
    else:
        raise NotImplementedError(
            f"Attention {attn_impl} with {sliding_window} sliding window not implemented"
        )


from apps.main.count_flops import attention_flops_per_token, get_num_flop_per_token  # noqa: F401


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


@dataclass
class LMTransformerArgs(BaseTransformerArgs):

    seed: int = 42

    vocab_size: int = -1
    weight_tying: bool = False
    attn_impl: str = "sdpa"  

    z_loss: bool = False

    sliding_window: Optional[int] = None

    # Fuse final linear with cross-entropy to avoid materializing full logits (saves ~6GB for 50k vocab).
    # Requires flash-linear-attention (fla). z_loss is disabled when fused.
    fuse_cross_entropy: bool = False


def _to_plain_for_fla(t: torch.Tensor) -> torch.Tensor:
    """Convert DTensor to plain tensor for FLA (which does not support DTensor)."""
    if isinstance(t, DTensor):
        return t.full_tensor()
    return t


def _get_fused_cross_entropy_loss():
    """Lazy import to avoid requiring fla when fuse_cross_entropy is False."""
    try:
        from fla.modules import FusedLinearCrossEntropyLoss
        return FusedLinearCrossEntropyLoss(ignore_index=-100)
    except ImportError as e:
        raise ImportError(
            "fuse_cross_entropy=True requires flash-linear-attention. "
            "Install with: pip install flash-linear-attention"
        ) from e


class LMTransformer(BaseTransformer):
    def __init__(self, args: LMTransformerArgs):
        super().__init__(args)
        self.weight_tying = args.weight_tying
        self.sliding_window = args.sliding_window
        self.attn_impl = args.attn_impl
        self.z_loss = args.z_loss
        self.fuse_cross_entropy = args.fuse_cross_entropy
        self.layer_norm_type = args.layer_norm_type
        print(f"Using attention implementation: {self.attn_impl}")

        assert args.vocab_size > 0

        self.tok_embeddings = torch.nn.Embedding(args.vocab_size, args.dim)

        self.norm = RMSNorm(args.dim, eps=args.norm_eps) if args.layer_norm_type != "none" else None
        # Peri-LN: optional initial embedding normalization (y0 = Norm(x0))
        self.emb_norm = RMSNorm(args.dim, eps=args.norm_eps) if args.layer_norm_type == "peri" else None

        if args.weight_tying:
            self.output = TiedLinear(self.tok_embeddings)
        else:
            self.output = nn.Linear(
                args.dim,
                args.vocab_size,
                bias=False,
            )

        self._fused_ce_loss = None
        if args.fuse_cross_entropy:
            if args.z_loss:
                import logging
                logging.getLogger().warning(
                    "fuse_cross_entropy=True disables z_loss (FusedLinearCrossEntropyLoss does not support it)"
                )
            self._fused_ce_loss = _get_fused_cross_entropy_loss()

    def forward(
        self,
        token_values: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, torch.Tensor, str]] = None,
        attn_impl: Optional[str] = None,
    ):
        bsz, seqlen = token_values.shape
        if attn_impl is None:
            attn_impl = self.attn_impl

        h = self.tok_embeddings(token_values)
        if self.emb_norm is not None:
            h = self.emb_norm(h)

        mask = (
            mask
            if mask is not None
            else create_causal_mask(seqlen, attn_impl, self.sliding_window)
        )

        h = super().forward(h, tok_idx=tok_idx, mask=mask, attn_impl=attn_impl)

        h_norm = self.norm(h) if self.norm is not None else h
        if target is not None and self._fused_ce_loss is not None:
            # Fused path: avoids materializing logits (saves ~6GB for 50k vocab).
            # Convert DTensor to plain: FLA's FusedLinearCrossEntropyLoss does not support DTensor.
            weight = (
                self.output.tied_module.weight
                if self.weight_tying
                else self.output.weight
            )
            h_norm_plain = _to_plain_for_fla(h_norm)
            weight_plain = _to_plain_for_fla(weight)
            # Align dtypes: full_tensor() can return float32 while activations are bf16
            weight_plain = weight_plain.to(h_norm_plain.dtype)
            return self._fused_ce_loss(h_norm_plain, target, weight_plain, None)
        logits = self.output(h_norm)
        if target is not None:
            return cross_entropy(logits, target, z_loss=self.z_loss)
        return logits

    def reset_parameters(self, init_std=None):
        # Either use fixed base std or sqrt model dim
        super().reset_parameters()
        init_std = init_std or (self.dim ** (-0.5))
        if self.norm is not None:
            self.norm.reset_parameters()
        if self.emb_norm is not None:
            self.emb_norm.reset_parameters()
        nn.init.trunc_normal_(
            self.tok_embeddings.weight,
            mean=0.0,
            std=init_std,
            a=-3 * init_std,
            b=3 * init_std,
        )
        if not self.weight_tying:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )


# Optional policy for activation checkpointing. With None, we stick to the default (defined distributed.py: default_no_recompute_ops)
def get_no_recompute_ops():
    return None


# Optional and only used for fully shard options (fsdp) is choose. Highly recommanded for large models
def build_fsdp_grouping_plan(model_args: LMTransformerArgs):
    group_plan: Tuple[int, bool] = []

    # Grouping and output seperately
    group_plan.append(("tok_embeddings", False))
    if hasattr(model_args, "layer_norm_type") and model_args.layer_norm_type == "peri":
        group_plan.append(("emb_norm", False))

    # Grouping by layers
    for i in range(model_args.n_layers):
        group_plan.append((f"layers.{i}", False))

    group_plan.append(("output", True))

    return group_plan


# Optional and only used for model/tensor parallelism when tp_size > 1
def tp_parallelize(model, tp_mesh, model_args: LMTransformerArgs, distributed_args):
    assert model_args.dim % distributed_args.tp_size == 0
    assert model_args.vocab_size % distributed_args.tp_size == 0
    assert model_args.n_heads % distributed_args.tp_size == 0
    assert (model_args.n_kv_heads or 0) % distributed_args.tp_size == 0
    assert model_args.n_heads % (model_args.n_kv_heads or 1) == 0

    # Embedding layer tp
    main_plan = {}
    main_plan["tok_embeddings"] = ColwiseParallel(
        input_layouts=Replicate(), output_layouts=Shard(1)
    )
    if model.norm is not None:
        main_plan["norm"] = SequenceParallel()
    if model.emb_norm is not None:
        main_plan["emb_norm"] = SequenceParallel()
    main_plan["output"] = ColwiseParallel(
        input_layouts=Shard(1), output_layouts=Replicate()
    )

    parallelize_module(
        model,
        tp_mesh,
        main_plan,
    )

    # Attention layers tp
    for layer in model.layers:
        layer_plan = {}

        layer_plan["attention"] = PrepareModuleInput(
            input_layouts=(Shard(1), None),
            desired_input_layouts=(Replicate(), None),
        )
        if layer.attention_norm is not None:
            layer_plan["attention_norm"] = SequenceParallel()
        layer_plan["attention.wq"] = ColwiseParallel()
        layer_plan["attention.wk"] = ColwiseParallel()
        layer_plan["attention.wv"] = ColwiseParallel()
        layer_plan["attention.wo"] = RowwiseParallel(output_layouts=Shard(1))

        # Feedforward layers tp
        layer_plan["feed_forward"] = PrepareModuleInput(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
        )
        if layer.ffn_norm is not None:
            layer_plan["ffn_norm"] = SequenceParallel()
        layer_plan["feed_forward.w1"] = ColwiseParallel()
        layer_plan["feed_forward.w3"] = ColwiseParallel()
        layer_plan["feed_forward.w2"] = RowwiseParallel(output_layouts=Shard(1))

        parallelize_module(
            layer,
            tp_mesh,
            layer_plan,
        )

        # Adjusting the number of heads and kv heads according to the tp size
        attn_layer = layer.attention
        attn_layer.n_heads = attn_layer.n_heads // distributed_args.tp_size
        attn_layer.n_kv_heads = attn_layer.n_kv_heads // distributed_args.tp_size
