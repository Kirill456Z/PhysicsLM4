"""
Analytical FLOPs counting for the LM transformer.

All functions operate on plain numeric arguments to avoid circular imports
with transformer.py.  Import this module wherever FLOPs estimates are needed.
"""


def attention_flops_per_token(n_layers: int, seq_len: int, dim: int, causal: bool) -> float:
    """Flash-attention FLOPs estimate per token (fwd+bwd).

    Formula from:
    https://github.com/Dao-AILab/flash-attention/blob/main/benchmarks/benchmark_flash_attention.py#L27-L30
    """
    return 3.5 * (4 * n_layers * seq_len * dim // (2 if causal else 1))


def get_num_flop_per_token(
    num_non_embed_params: int,
    n_layers: int,
    dim: int,
    seq_len: int,
) -> float:
    """Total (fwd+bwd) FLOPs per token via the standard 6N + attention formula.

    Args:
        num_non_embed_params: parameter count excluding token-embedding and output-head weights.
        n_layers: number of transformer layers.
        dim: model hidden dimension.
        seq_len: training sequence length.
    """
    return 6 * num_non_embed_params + attention_flops_per_token(n_layers, seq_len, dim, True)


def compute_total_analytical_flops(
    num_non_embed_params: int,
    n_layers: int,
    dim: int,
    seq_len: int,
    total_tokens: int,
) -> int:
    """Cumulative FLOPs processed since the start of training.

    Args:
        num_non_embed_params: non-embedding parameter count (model_params - vocab_size * dim).
        n_layers: number of transformer layers.
        dim: model hidden dimension.
        seq_len: training sequence length.
        total_tokens: total number of tokens seen across all ranks up to the current step.
    """
    return int(get_num_flop_per_token(num_non_embed_params, n_layers, dim, seq_len) * total_tokens)
