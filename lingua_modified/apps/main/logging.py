"""Training diagnostics metrics for Canon vs baseline comparison.

Frequency tiers (configured via LoggingArgs):
  easy   – default every 50 steps:   grad norms, UWR, canon kernel stats, instability
  medium – default every 500 steps:  activation/residual stats (probe steps)
  heavy  – default every 2000 steps: covariance spectra, sharpness (probe steps)

All collector functions return flat dicts {metric_name: scalar_value}
ready to merge into wandb metrics.
"""

import math

import torch
from torch.distributed._tensor import DTensor


# ── Helpers ──────────────────────────────────────────────────────────────────


def _to_plain(t: torch.Tensor) -> torch.Tensor:
    """Unwrap DTensor to a plain tensor for norm computation."""
    if isinstance(t, DTensor):
        return t.full_tensor()
    return t


def _resolve_attr(obj, dotted_path):
    """Resolve a dot-separated attribute path, returning None if any part is missing."""
    for part in dotted_path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


# Component paths relative to a TransformerBlock
_LINEAR_COMPONENTS = [
    ("attention.wq", "wq"),
    ("attention.wk", "wk"),
    ("attention.wv", "wv"),
    ("attention.wo", "wo"),
    ("feed_forward.w1", "w1"),
    ("feed_forward.w2", "w2"),
    ("feed_forward.w3", "w3"),
]

_CANON_COMPONENTS = [
    ("canonA", "canonA"),
    ("attention.canonB", "canonB"),
    ("canonC", "canonC"),
    ("feed_forward.canonD", "canonD"),
]

_ALL_COMPONENTS = _LINEAR_COMPONENTS + _CANON_COMPONENTS


# ═══════════════════════════════════════════════════════════════════════════════
#  EASY TIER – grad norms, weight ratios, canon kernel analysis, instability
# ═══════════════════════════════════════════════════════════════════════════════


# ── Instability counters ────────────────────────────────────────────────────


def extract_instability_counters(loss_value: float, grad_norm_value: float) -> dict:
    """Check loss and gradient norm for NaN / Inf."""
    return {
        "instability/loss_nan": int(math.isnan(loss_value)),
        "instability/loss_inf": int(math.isinf(loss_value)),
        "instability/grad_nan": int(math.isnan(grad_norm_value)),
        "instability/grad_inf": int(math.isinf(grad_norm_value)),
    }


# ── Layerwise gradient norms ────────────────────────────────────────────────


def extract_layerwise_grad_norms(model) -> dict:
    """
    Per-layer, per-component L1 and L2 gradient norms.

    Covers attention (wq/wk/wv/wo), FFN (w1/w2/w3), canon (A/B/C/D),
    and the shared embedding weight.

    Keys: grad_norm/{component}_l2/layer_{i}
          grad_norm/{component}_l1/layer_{i}
          grad_norm/embedding_l2
          grad_norm/embedding_l1
    """
    metrics = {}
    with torch.no_grad():
        for layer_idx, layer in enumerate(model.layers):
            for attr_path, name in _ALL_COMPONENTS:
                module = _resolve_attr(layer, attr_path)
                if (
                    module is not None
                    and hasattr(module, "weight")
                    and module.weight.grad is not None
                ):
                    grad = _to_plain(module.weight.grad)
                    metrics[f"grad_norm/{name}_l2/layer_{layer_idx}"] = torch.norm(grad, p=2).item()
                    metrics[f"grad_norm/{name}_l1/layer_{layer_idx}"] = torch.norm(grad, p=1).item()

        # Embedding (shared with lm_head when weight_tying=True)
        if hasattr(model, "tok_embeddings") and model.tok_embeddings.weight.grad is not None:
            grad = _to_plain(model.tok_embeddings.weight.grad)
            metrics["grad_norm/embedding_l2"] = torch.norm(grad, p=2).item()
            metrics["grad_norm/embedding_l1"] = torch.norm(grad, p=1).item()

    return metrics


# ── Grad-to-weight ratio (UWR proxy) ────────────────────────────────────────


def extract_grad_to_weight_ratios(model) -> dict:
    """
    Per-layer |∇W|₂ / |W|₂ — optimizer-independent update-to-weight ratio proxy.

    Keys: uwr/{component}/layer_{i}
          uwr/embedding
    """
    metrics = {}
    with torch.no_grad():
        for layer_idx, layer in enumerate(model.layers):
            for attr_path, name in _ALL_COMPONENTS:
                module = _resolve_attr(layer, attr_path)
                if (
                    module is not None
                    and hasattr(module, "weight")
                    and module.weight.grad is not None
                ):
                    grad = _to_plain(module.weight.grad)
                    weight = _to_plain(module.weight.data)
                    w_norm = torch.norm(weight, p=2).item()
                    if w_norm > 0:
                        metrics[f"uwr/{name}/layer_{layer_idx}"] = (
                            torch.norm(grad, p=2).item() / w_norm
                        )

        if hasattr(model, "tok_embeddings") and model.tok_embeddings.weight.grad is not None:
            grad = _to_plain(model.tok_embeddings.weight.grad)
            weight = _to_plain(model.tok_embeddings.weight.data)
            w_norm = torch.norm(weight, p=2).item()
            if w_norm > 0:
                metrics["uwr/embedding"] = torch.norm(grad, p=2).item() / w_norm

    return metrics


# ── Per-layer contribution to global gradient norm ──────────────────────────


def extract_per_layer_grad_contribution(model) -> dict:
    """
    Fraction of total gradient norm² from each transformer layer.

    Useful for diagnosing depth-wise gradient imbalance (cf. NormFormer).

    Keys: grad_contrib/layer_{i}  (values sum to ≈1.0 with non_layer)
          grad_contrib/non_layer
    """
    metrics = {}
    layer_norms_sq = []
    layer_param_ids = set()

    with torch.no_grad():
        for layer in model.layers:
            norm_sq = 0.0
            for p in layer.parameters():
                if p.grad is not None:
                    g = _to_plain(p.grad)
                    norm_sq += g.float().pow(2).sum().item()
                layer_param_ids.add(id(p))
            layer_norms_sq.append(norm_sq)

        non_layer_norm_sq = 0.0
        for p in model.parameters():
            if id(p) not in layer_param_ids and p.grad is not None:
                g = _to_plain(p.grad)
                non_layer_norm_sq += g.float().pow(2).sum().item()

        total = sum(layer_norms_sq) + non_layer_norm_sq
        if total > 0:
            for layer_idx, ns in enumerate(layer_norms_sq):
                metrics[f"grad_contrib/layer_{layer_idx}"] = ns / total
            metrics["grad_contrib/non_layer"] = non_layer_norm_sq / total

    return metrics


# ── Canon weight norms (per shift position) ─────────────────────────────────


def extract_canon_weight_norms(model) -> dict:
    """
    L2 norm of canon conv weights for each shift position.

    Canon weights have shape (hidden_size, 1, kernel_size) — depthwise grouped conv.

    Keys: canon_weight/{name}/layer_{i}/shift_{k}
    """
    metrics = {}
    with torch.no_grad():
        for layer_idx, layer in enumerate(model.layers):
            for attr_path, name in _CANON_COMPONENTS:
                module = _resolve_attr(layer, attr_path)
                if module is not None and hasattr(module, "weight"):
                    weight = _to_plain(module.weight)   # (H, 1, K)
                    w2d = weight.squeeze(1)              # (H, K)
                    for k in range(w2d.shape[1]):
                        metrics[f"canon_weight/{name}/layer_{layer_idx}/shift_{k}"] = (
                            torch.norm(w2d[:, k]).item()
                        )
    return metrics


# ── Canon kernel analysis ───────────────────────────────────────────────────


def extract_canon_kernel_analysis(model) -> dict:
    """
    Canon-specific diagnostics for "scale-only vs genuine mixing":

    diagonal_dominance — |1 + K[t=0]| / Σ_{t≠0}|K[t]|  (residual folded in)
        Higher → behaves more like per-channel rescaling.
    mixing_energy — mean over channels of Σ_{t≠0}|K[t]|
        Direct neighbour-tap magnitude.
    effective_scale — mean over channels of √(Σ_t K[t]²)
        Expected gain on iid inputs (learned kernel only, no residual).

    Keys: canon_analysis/{metric}/{name}/layer_{i}
    """
    metrics = {}
    with torch.no_grad():
        for layer_idx, layer in enumerate(model.layers):
            for attr_path, name in _CANON_COMPONENTS:
                module = _resolve_attr(layer, attr_path)
                if module is None or not hasattr(module, "weight"):
                    continue

                weight = _to_plain(module.weight)    # (H, 1, K)
                w = weight.squeeze(1).float()         # (H, K)

                # Convention: kernel positions are [oldest … t=-1, t=0]
                # t=0 (current token) is the last position.

                # Diagonal dominance (residual folded in: effective t=0 = 1 + K[t=0])
                t0 = w[:, -1]
                neighbour = w[:, :-1]
                eff_t0_abs = (1.0 + t0).abs()                      # (H,)
                neighbour_abs_sum = neighbour.abs().sum(dim=1)      # (H,)
                diag_dom = (eff_t0_abs / (neighbour_abs_sum + 1e-8)).mean().item()
                metrics[f"canon_analysis/diagonal_dominance/{name}/layer_{layer_idx}"] = diag_dom

                # Mixing energy
                mix_energy = neighbour_abs_sum.mean().item()
                metrics[f"canon_analysis/mixing_energy/{name}/layer_{layer_idx}"] = mix_energy

                # Effective scale (learned kernel only)
                eff_scale = w.pow(2).sum(dim=1).sqrt().mean().item()
                metrics[f"canon_analysis/effective_scale/{name}/layer_{layer_idx}"] = eff_scale

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
#  Top-level collectors (called from the training loop)
# ═══════════════════════════════════════════════════════════════════════════════


def collect_easy_grad_metrics(model) -> dict:
    """
    Easy-tier metrics that require gradients.

    Call after loss.backward() and BEFORE optimizer.zero_grad().
    """
    metrics = {}
    metrics.update(extract_layerwise_grad_norms(model))
    metrics.update(extract_grad_to_weight_ratios(model))
    metrics.update(extract_per_layer_grad_contribution(model))
    return metrics


def collect_easy_weight_metrics(model) -> dict:
    """
    Easy-tier metrics computed from weights only (no gradient dependency).

    Can be called at any point in the training loop.
    """
    metrics = {}
    metrics.update(extract_canon_weight_norms(model))
    metrics.update(extract_canon_kernel_analysis(model))
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
#  MEDIUM TIER – activation stats, residual dynamics, canon alignment
#  Default frequency: every 500 steps (diagnostics_medium_freq).
#
#  Collected via a separate no-grad half-batch forward pass with torch.compile
#  temporarily disabled (TorchCompileDisabler) so that forward hooks can fire.
#  All ranks must call the collector (FSDP all-gather), but only the master
#  rank registers hooks and returns non-empty metrics.
#
#  Metrics produced (per transformer layer unless noted):
#
#  Weight-based (no forward pass):
#    rmsnorm_gamma/{mean,std,min,max}/{attn_norm,ffn_norm}/layer_*
#        Learned RMSNorm scale (γ) statistics. Tracks whether the model learns
#        depth-dependent rescaling (cf. NormFormer γ analysis).
#
#  Residual stream:
#    residual_rms/pre_attn/layer_*   – RMS of hidden state entering the layer
#    residual_rms/pre_ffn/layer_*    – RMS after attention residual add
#        Depth profile of activation magnitude; growth/shrinkage across layers.
#
#  Activation magnitudes (internal):
#    activation_rms/{attn_norm_out,attn_input,attn_out}/layer_*
#    activation_rms/{ffn_norm_out,ffn_input,ffn_out}/layer_*
#        RMS at every sub-layer boundary. attn_norm_out vs attn_input differ
#        only when canon A is active (shows canon's rescaling effect).
#
#  Branch contribution ratios:
#    branch_ratio/attn/layer_*  – rms(attn_output) / rms(residual_input)
#    branch_ratio/ffn/layer_*   – rms(ffn_output)  / rms(residual_input)
#        Relative strength of each branch vs the residual stream.
#        Large early-layer ratios signal gradient-magnitude mismatch.
#
#  Canon alignment (only when canon A / C are active):
#    canon_align/rms_ratio/{canonA,canonC}/layer_*
#        rms(post_canon) / rms(pre_canon). Close to 1.0 → mostly rescaling.
#    canon_align/cos_sim/{canonA,canonC}/layer_*
#        Cosine similarity between pre- and post-canon activations (flattened
#        over sequence×dim per sample, averaged over batch). Close to 1.0 →
#        canon preserves direction (NormFormer-like rescaling). Significantly
#        below 1.0 → genuine sequence mixing via the conv1d.
#
#  Temporal derivative (sequence smoothness):
#    temporal_deriv/residual/layer_*
#        rms(h_i − h_{i−1}) of the residual stream entering each layer.
#    temporal_deriv/post_canonA/layer_*
#    temporal_deriv/post_canonC/layer_*
#        Same metric after canon conv1d. If canon reduces this value it is
#        smoothing the sequence; if it increases it, it adds high-frequency
#        structure.
#
#  Attention entropy:
#    attn_entropy/mean/layer_*
#        Mean Shannon entropy of the attention distribution, averaged over
#        all heads, batch samples and query positions. Computed by replaying
#        the full Q·K pipeline (wq/wk → q/k-norm → Canon B → RoPE → GQA →
#        scaled logits → causal mask → softmax). Higher values indicate
#        more uniform attention; lower values indicate sharper/peaked heads.
#        Sequence is capped at 1024 tokens to bound the O(S²) logit matmul.
#    attn_entropy/std_across_heads/layer_*
#        Std-dev of per-head mean entropy within each layer. High values
#        indicate heterogeneous head behaviour (some peaked, some diffuse).
#
#  Outlier features (residual stream entering each block):
#    outlier_features/kurtosis/layer_*
#        Kurtosis of neuron activation RMS: m_4/m_2^2 over s_j = RMS of
#        neuron j across positions. Min 1 (uniform), max d (single outlier).
#    outlier_features/max_median_ratio/layer_*
#        Mean over positions of max_j|X_{α,j}| / median_j|X_{α,j}|.
#        Min 1 (uniform), unbounded when dominant outlier exists.
# ═══════════════════════════════════════════════════════════════════════════════


# ── Outlier feature metrics (residual stream) ──────────────────────────────


def compute_outlier_kurtosis(x: torch.Tensor) -> float:
    """Kurtosis of neuron activation RMS over residual stream X (n×d).

    s_j = sqrt((1/n) sum_α X_{α,j}^2), Kurt = m_4/m_2^2 where m_k = (1/d) sum_j s_j^k.
    Min 1 (uniform), max d (single outlier).
    """
    x = x.detach().float()
    x_flat = x.reshape(-1, x.shape[-1])
    s = x_flat.pow(2).mean(dim=0).sqrt()
    m_4 = s.pow(4).mean().item()
    m_2 = s.pow(2).mean().item()
    return m_4 / (m_2 * m_2 + 1e-12)


def compute_outlier_max_median_ratio(x: torch.Tensor) -> float:
    """Max-Median Ratio: mean_α (max_j |X_{α,j}| / median_j |X_{α,j}|).

    Min 1 (uniform), unbounded when dominant outlier exists.
    """
    x = x.detach().float().abs()
    x_flat = x.reshape(-1, x.shape[-1])
    max_per_pos = x_flat.max(dim=1)[0]
    median_per_pos = x_flat.median(dim=1)[0]
    ratio = max_per_pos / (median_per_pos + 1e-12)
    return ratio.mean().item()


# ── RMSNorm γ statistics (weight-based, no forward pass) ────────────────────


def extract_rmsnorm_stats(model) -> dict:
    """
    RMSNorm learned scale (γ) statistics per layer — mean, std, min, max.

    No forward pass needed.

    Keys: rmsnorm_gamma/{stat}/{norm_name}/layer_{i}
    """
    metrics = {}
    with torch.no_grad():
        for layer_idx, layer in enumerate(model.layers):
            for norm_name, norm_attr in [
                ("attn_norm", "attention_norm"),
                ("ffn_norm", "ffn_norm"),
            ]:
                norm = getattr(layer, norm_attr, None)
                if norm is not None and hasattr(norm, "weight"):
                    w = _to_plain(norm.weight).float()
                    metrics[f"rmsnorm_gamma/mean/{norm_name}/layer_{layer_idx}"] = w.mean().item()
                    metrics[f"rmsnorm_gamma/std/{norm_name}/layer_{layer_idx}"] = w.std().item()
                    metrics[f"rmsnorm_gamma/min/{norm_name}/layer_{layer_idx}"] = w.min().item()
                    metrics[f"rmsnorm_gamma/max/{norm_name}/layer_{layer_idx}"] = w.max().item()
    return metrics


# ── Activation statistics collector (forward-hook based) ─────────────────────


class _MediumMetricsCollector:
    """Collects activation statistics via temporary forward hooks.

    Hook placement mirrors TransformerBlock.forward:
        x → [attention_norm] → xx → [canonA?] → xx' → [attention] → attn_out
            h = x + attn_out
        h → [ffn_norm] → hh → [canonC?] → hh' → [feed_forward] → ffn_out
            out = h + ffn_out
    """

    _MAX_SEQ_FOR_ENTROPY = 1024  # cap sequence length to bound O(S²) logit matmul

    def __init__(self, collect_heavy=False):
        self.metrics = {}
        self._hooks = []
        self._temp = {}
        self._attn_inputs = {}   # {layer_idx: (x, freq_cis)}  for entropy
        self._collect_heavy = collect_heavy
        self._ffn_inputs = {}    # {layer_idx: x}  for Canon D alignment (heavy)

    # ── stat helpers (all detach, cast to float) ──

    @staticmethod
    def _rms(x):
        """RMS over hidden dim, averaged over batch and sequence."""
        return x.detach().float().pow(2).mean(-1).sqrt().mean().item()

    @staticmethod
    def _temporal_deriv_rms(x):
        """RMS of consecutive-position differences (smoothness measure)."""
        x = x.detach().float()
        if x.shape[1] <= 1:
            return 0.0
        diff = x[:, 1:] - x[:, :-1]
        return diff.pow(2).mean(-1).sqrt().mean().item()

    @staticmethod
    def _cos_sim_batch_mean(a, b):
        """Cosine similarity averaged over batch (flatten seq×dim per sample)."""
        a_flat = a.detach().float().reshape(a.shape[0], -1)
        b_flat = b.detach().float().reshape(b.shape[0], -1)
        return torch.nn.functional.cosine_similarity(a_flat, b_flat, dim=1).mean().item()

    # ── hook registration ──

    def register_hooks(self, model):
        """Register temporary forward hooks on all transformer layers."""
        for L, layer in enumerate(model.layers):
            has_canon_a = layer.canonA is not None
            has_canon_c = layer.canonC is not None

            # 1) Block input — residual stream entering this layer
            def _block_pre(mod, args, L=L):
                x = args[0]
                self.metrics[f"residual_rms/pre_attn/layer_{L}"] = self._rms(x)
                self.metrics[f"temporal_deriv/residual/layer_{L}"] = self._temporal_deriv_rms(x)
                self.metrics[f"outlier_features/kurtosis/layer_{L}"] = compute_outlier_kurtosis(x)
                self.metrics[f"outlier_features/max_median_ratio/layer_{L}"] = (
                    compute_outlier_max_median_ratio(x)
                )

            self._hooks.append(layer.register_forward_pre_hook(_block_pre))

            # 2) attention_norm output — pre-canon-A activation (skip when layer_norm_type="none")
            if layer.attention_norm is not None:
                def _attn_norm_out(mod, inp, out, L=L, hca=has_canon_a):
                    self.metrics[f"activation_rms/attn_norm_out/layer_{L}"] = self._rms(out)
                    if hca:
                        self._temp[f"pre_cA/{L}"] = out.detach()

                self._hooks.append(layer.attention_norm.register_forward_hook(_attn_norm_out))

            # 3) Attention pre-forward — post-canon-A (= actual attention input)
            #    Also stores (x, freq_cis) for attention entropy computation.
            def _attn_pre(mod, args, L=L, hca=has_canon_a):
                x_in = args[0]
                self.metrics[f"activation_rms/attn_input/layer_{L}"] = self._rms(x_in)
                key = f"pre_cA/{L}"
                if hca and key in self._temp:
                    pre = self._temp.pop(key)
                    pre_rms = self._rms(pre)
                    post_rms = self._rms(x_in)
                    self.metrics[f"canon_align/rms_ratio/canonA/layer_{L}"] = (
                        post_rms / (pre_rms + 1e-8)
                    )
                    self.metrics[f"canon_align/cos_sim/canonA/layer_{L}"] = (
                        self._cos_sim_batch_mean(pre, x_in)
                    )
                    self.metrics[f"temporal_deriv/post_canonA/layer_{L}"] = (
                        self._temporal_deriv_rms(x_in)
                    )
                # Store for entropy: args = (x, freq_cis, tok_idx, mask, attn_impl, kv_cache)
                freq_cis = args[1] if len(args) > 1 else None
                self._attn_inputs[L] = (x_in.detach(), freq_cis)

            self._hooks.append(layer.attention.register_forward_pre_hook(_attn_pre))

            # 4) Attention output — branch contribution
            def _attn_out(mod, inp, out, L=L):
                attn_rms = self._rms(out)
                self.metrics[f"activation_rms/attn_out/layer_{L}"] = attn_rms
                pre_rms = self.metrics.get(f"residual_rms/pre_attn/layer_{L}", 1e-8)
                self.metrics[f"branch_ratio/attn/layer_{L}"] = attn_rms / (pre_rms + 1e-8)

            self._hooks.append(layer.attention.register_forward_hook(_attn_out))

            # 5) ffn_norm pre-forward — captures h = x + attn_out (residual for FFN)
            if layer.ffn_norm is not None:
                def _ffn_norm_pre(mod, args, L=L):
                    h = args[0]
                    self.metrics[f"residual_rms/pre_ffn/layer_{L}"] = self._rms(h)

                self._hooks.append(layer.ffn_norm.register_forward_pre_hook(_ffn_norm_pre))

                # 6) ffn_norm output — pre-canon-C activation
                def _ffn_norm_out(mod, inp, out, L=L, hcc=has_canon_c):
                    self.metrics[f"activation_rms/ffn_norm_out/layer_{L}"] = self._rms(out)
                    if hcc:
                        self._temp[f"pre_cC/{L}"] = out.detach()

                self._hooks.append(layer.ffn_norm.register_forward_hook(_ffn_norm_out))

            # 7) feed_forward pre-forward — post-canon-C (= actual FFN input)
            #    Also stores x for Canon D alignment (heavy tier).
            def _ffn_pre(mod, args, L=L, hcc=has_canon_c):
                x_in = args[0]
                self.metrics[f"activation_rms/ffn_input/layer_{L}"] = self._rms(x_in)
                key = f"pre_cC/{L}"
                if hcc and key in self._temp:
                    pre = self._temp.pop(key)
                    pre_rms = self._rms(pre)
                    post_rms = self._rms(x_in)
                    self.metrics[f"canon_align/rms_ratio/canonC/layer_{L}"] = (
                        post_rms / (pre_rms + 1e-8)
                    )
                    self.metrics[f"canon_align/cos_sim/canonC/layer_{L}"] = (
                        self._cos_sim_batch_mean(pre, x_in)
                    )
                    self.metrics[f"temporal_deriv/post_canonC/layer_{L}"] = (
                        self._temporal_deriv_rms(x_in)
                    )
                # Store FFN input for Canon D alignment (heavy tier)
                if self._collect_heavy:
                    self._ffn_inputs[L] = x_in.detach()

            self._hooks.append(layer.feed_forward.register_forward_pre_hook(_ffn_pre))

            # 8) feed_forward output — FFN branch contribution
            def _ffn_out(mod, inp, out, L=L):
                ffn_rms = self._rms(out)
                self.metrics[f"activation_rms/ffn_out/layer_{L}"] = ffn_rms
                pre_rms = self.metrics.get(f"residual_rms/pre_ffn/layer_{L}", 1e-8)
                self.metrics[f"branch_ratio/ffn/layer_{L}"] = ffn_rms / (pre_rms + 1e-8)

            self._hooks.append(layer.feed_forward.register_forward_hook(_ffn_out))

    # ── attention entropy (called after forward pass) ──

    def compute_attention_entropy(self, model):
        """Compute per-layer attention entropy from stored (x, freq_cis) pairs.

        Replicates the Q·K pipeline (wq/wk → q/k-norm → Canon B → RoPE →
        GQA repeat → scaled dot-product logits → causal mask → softmax →
        entropy) so that the result matches the actual attention distribution.

        Must be called after the forward pass and before remove_hooks().
        """
        from lingua.transformer import apply_rotary_emb, repeat_kv
        from lingua.canon_helper import apply_canon

        for L, (x, freq_cis) in self._attn_inputs.items():
            attn = model.layers[L].attention
            bsz, seq_len, _ = x.shape

            # Cap sequence length to bound O(S²) cost
            S = min(seq_len, self._MAX_SEQ_FOR_ENTROPY)
            x = x[:, :S]
            if freq_cis is not None:
                fc = freq_cis[:S]
            else:
                fc = None

            # Q / K projections
            xq = attn.wq(x)
            xk = attn.wk(x)

            # Q/K norms
            if attn.q_norm is not None:
                xq = attn.q_norm(xq)
            if attn.k_norm is not None:
                xk = attn.k_norm(xk)

            # Canon B (operates on concatenated [q, k, v])
            if attn.canonB is not None:
                xv = attn.wv(x)
                qkv = apply_canon(
                    "canonB",
                    attn.canonB,
                    torch.cat([xq, xk, xv], dim=-1),
                    None,
                    None,
                )
                xq, xk, _ = qkv.split(
                    [
                        attn.n_heads * attn.head_dim,
                        attn.n_kv_heads * attn.head_dim,
                        attn.n_kv_heads * attn.head_dim,
                    ],
                    dim=-1,
                )

            # Reshape to (B, S, H, D)
            hd = attn.head_dim
            xq = xq.view(bsz, S, attn.n_heads, hd)
            xk = xk.view(bsz, S, attn.n_kv_heads, hd)

            # RoPE
            if fc is not None:
                if attn.rope_dim is None:
                    xq, xk = apply_rotary_emb(xq, xk, 1, fc)
                else:
                    rd = attn.rope_dim
                    xq_rot, xk_rot = apply_rotary_emb(
                        xq[..., :rd], xk[..., :rd], 1, fc
                    )
                    xq = torch.cat((xq_rot, xq[..., rd:]), dim=-1)
                    xk = torch.cat((xk_rot, xk[..., rd:]), dim=-1)

            # GQA: expand K heads
            xk = repeat_kv(xk, attn.heads_per_group, dim=2)

            # (B, H, S, D)
            xq = xq.transpose(1, 2)
            xk = xk.transpose(1, 2)

            # Scaled logits  (B, H, S, S)
            logits = torch.matmul(xq.float(), xk.float().transpose(-2, -1)) / math.sqrt(hd)

            # Causal mask
            causal = torch.triu(
                torch.ones(S, S, device=logits.device, dtype=torch.bool), diagonal=1
            )
            logits.masked_fill_(causal, float("-inf"))

            # Softmax → entropy  H(p) = -Σ p·log(p)
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log(probs + 1e-10)
            entropy = -(probs * log_probs).sum(dim=-1)  # (B, H, S)

            # Per-head mean (over batch and queries)
            per_head = entropy.mean(dim=(0, 2))  # (H,)
            self.metrics[f"attn_entropy/mean/layer_{L}"] = per_head.mean().item()
            self.metrics[f"attn_entropy/std_across_heads/layer_{L}"] = per_head.std().item()

        self._attn_inputs.clear()

    # ── Canon B / D alignment (heavy tier, called after forward pass) ──

    def compute_canon_bd_alignment(self, model):
        """Compute rms_ratio and cos_sim for Canon B and Canon D.

        Canon B operates inside Attention.forward on concatenated [q, k, v]
        after the linear projections + Q/K norms.  Canon D operates inside
        FeedForward.forward on concatenated [w1(x), w3(x)].

        Because these happen on local variables (no hookable submodule
        boundary), we replay the relevant sub-pipelines from the stored
        inputs captured during the forward pass.

        Must be called after the forward pass and before remove_hooks().
        Only produces metrics when the respective canon layer is present.
        """
        from lingua.canon_helper import apply_canon

        # ── Canon B ──
        for L, (x, _freq_cis) in self._attn_inputs.items():
            attn = model.layers[L].attention
            if attn.canonB is None:
                continue

            xq = attn.wq(x)
            xk = attn.wk(x)
            xv = attn.wv(x)

            if attn.q_norm is not None:
                xq = attn.q_norm(xq)
            if attn.k_norm is not None:
                xk = attn.k_norm(xk)

            pre = torch.cat([xq, xk, xv], dim=-1)
            post = apply_canon("canonB", attn.canonB, pre, None, None)

            pre_rms = self._rms(pre)
            post_rms = self._rms(post)
            self.metrics[f"canon_align/rms_ratio/canonB/layer_{L}"] = (
                post_rms / (pre_rms + 1e-8)
            )
            self.metrics[f"canon_align/cos_sim/canonB/layer_{L}"] = (
                self._cos_sim_batch_mean(pre, post)
            )

        # ── Canon D ──
        for L, x in self._ffn_inputs.items():
            ff = model.layers[L].feed_forward
            if ff.canonD is None:
                continue

            x1 = ff.w1(x)
            x3 = ff.w3(x)
            pre = torch.cat([x1, x3], dim=-1)
            post = apply_canon("canonD", ff.canonD, pre, None, None)

            pre_rms = self._rms(pre)
            post_rms = self._rms(post)
            self.metrics[f"canon_align/rms_ratio/canonD/layer_{L}"] = (
                post_rms / (pre_rms + 1e-8)
            )
            self.metrics[f"canon_align/cos_sim/canonD/layer_{L}"] = (
                self._cos_sim_batch_mean(pre, post)
            )

        # Don't clear _attn_inputs here — entropy may need them too.
        # They are cleared in remove_hooks().
        self._ffn_inputs.clear()

    # ── cleanup ──

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._temp.clear()
        self._attn_inputs.clear()
        self._ffn_inputs.clear()

    def get_metrics(self):
        result = dict(self.metrics)
        self.metrics.clear()
        return result


# ── Top-level medium collector ──────────────────────────────────────────────


def _run_forward_with_collector(
    model, input_ids, bsz, seqlen, is_master, collect_heavy=False,
) -> dict:
    """Shared forward pass for medium (and optionally heavy) metric collection.

    All ranks MUST call this (for FSDP all-gather correctness).
    Only the master rank registers hooks and returns non-empty metrics.
    """
    from lingua.probe import TorchCompileDisabler

    collector = (
        _MediumMetricsCollector(collect_heavy=collect_heavy) if is_master else None
    )
    compile_disabler = TorchCompileDisabler(model)

    try:
        compile_disabler.__enter__()
        if collector is not None:
            collector.register_hooks(model)

        was_training = model.training
        model.eval()
        with torch.no_grad():
            small_bsz = max(1, bsz // 2)
            _ = model(input_ids[:small_bsz, :seqlen])

            if collector is not None:
                # Heavy-tier: Canon B / D alignment (runs before entropy
                # because entropy clears _attn_inputs)
                if collect_heavy:
                    collector.compute_canon_bd_alignment(model)

                # Medium-tier: attention entropy (clears _attn_inputs)
                collector.compute_attention_entropy(model)

        if was_training:
            model.train()

    finally:
        if collector is not None:
            collector.remove_hooks()
        compile_disabler.__exit__(None, None, None)

    return collector.get_metrics() if collector is not None else {}


def collect_medium_forward_metrics(model, input_ids, bsz, seqlen, is_master=True) -> dict:
    """
    Run a separate no-grad forward pass with hooks to capture activation stats.

    All ranks MUST call this (for FSDP all-gather correctness).
    Only the master rank registers hooks and returns non-empty metrics.

    Args:
        model: LMTransformer (may be FSDP-wrapped + compiled)
        input_ids: current batch input_ids (a small subset is used)
        bsz: current batch size
        seqlen: current sequence length
        is_master: whether this rank should collect metrics
    """
    return _run_forward_with_collector(
        model, input_ids, bsz, seqlen, is_master, collect_heavy=False,
    )


def collect_medium_metrics(model, input_ids, bsz, seqlen, is_master=True) -> dict:
    """
    Collect all medium-tier metrics (forward-pass dependent + weight-based).

    All ranks MUST call this function. Only master gets non-empty results.
    """
    metrics = collect_medium_forward_metrics(model, input_ids, bsz, seqlen, is_master)
    if is_master:
        metrics.update(extract_rmsnorm_stats(model))
    return metrics


# ── Top-level heavy collector ───────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
# Heavy-tier metrics (default every 2000 steps)
#
#  Canon B / D alignment (only when Canon B / D are active):
#    canon_align/rms_ratio/canonB/layer_*
#        rms(post_canonB) / rms(pre_canonB) where pre/post refers to the
#        concatenated [q, k, v] tensor (after wq/wk/wv + Q/K norms, before
#        reshape+RoPE).  Computed by replaying the sub-pipeline from the
#        stored attention input.
#    canon_align/cos_sim/canonB/layer_*
#        Cosine similarity between pre- and post-Canon-B activations.
#    canon_align/rms_ratio/canonD/layer_*
#        rms(post_canonD) / rms(pre_canonD) where pre/post refers to the
#        concatenated [w1(x), w3(x)] tensor (after w1/w3, before SiLU gate).
#    canon_align/cos_sim/canonD/layer_*
#        Cosine similarity between pre- and post-Canon-D activations.
#
#  These are expensive because they re-run wq/wk/wv (Canon B) or w1/w3
#  (Canon D) projections from stored inputs.  Kept at heavy frequency to
#  limit overhead.
# ═══════════════════════════════════════════════════════════════════════════════


def collect_heavy_metrics(
    model, input_ids, bsz, seqlen, is_master=True,
) -> dict:
    """Collect heavy-tier metrics (Canon B/D alignment + all medium metrics).

    Runs a single forward pass with ``collect_heavy=True`` so that the
    collector stores extra tensors needed for the Canon B / D replay.
    Also includes all medium-tier metrics (activation stats, entropy,
    RMSNorm γ) since heavy steps are a superset of medium steps.

    All ranks MUST call this function.  Only master gets non-empty results.
    """
    metrics = _run_forward_with_collector(
        model, input_ids, bsz, seqlen, is_master, collect_heavy=True,
    )
    if is_master:
        metrics.update(extract_rmsnorm_stats(model))
    return metrics
