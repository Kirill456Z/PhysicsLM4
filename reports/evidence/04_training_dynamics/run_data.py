"""
F4: Canon changes training dynamics — gradient uniformity, representation scaling,
    learned weights, outlier features, cosine similarity.

Fetches training time-series for Canon-ABCD and Llama baseline on text modeling.
Prints key per-layer metrics at the end of training, then saves layerwise plots.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb

WANDB_API_KEY = "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"
ENTITY = "kirill456z"
PROJECT = "physics4llm"
OUT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(OUT_DIR, exist_ok=True)

CANON_RUN = "canon_text_0.1.129"   # display: text_ks_4_default_init_with_residual_trainable_bs_128_seq_len_2048_0.1.129
LLAMA_RUN = "canon_text_0.1.130"   # display: text_llama_bs_128_seq_len_2048_0.1.130
N_LAYERS = 8
CANON_TYPES = ["A", "B", "C", "D"]
KERNEL_SIZE = 4

# Metric key builders
def grad_key(layer):      return f"grad_contrib/layer_{layer}"
def kurtosis_key(layer):  return f"outlier_features/kurtosis/layer_{layer}"
def rms_key(ctype, layer):return f"canon_align/rms_ratio/canon{ctype}/layer_{layer}"
def cos_key(ctype, layer):return f"canon_align/cos_sim/canon{ctype}/layer_{layer}"
def weight_key(ctype, layer, shift): return f"canon_weight/canon{ctype}/layer_{layer}/shift_{shift}"


def fetch_run_history(run_name, keys, samples=1000):
    api = wandb.Api(api_key=WANDB_API_KEY)
    run = api.run(f"{ENTITY}/{PROJECT}/{run_name}")
    hist = run.history(samples=samples, keys=keys)
    if "_step" not in hist.columns and not hist.empty:
        hist = hist.reset_index().rename(columns={"index": "_step"})
    return hist.sort_values("_step").reset_index(drop=True)


def last_step_values(hist, keys):
    """Return a dict of {key: value} using the trailing-50-step average."""
    result = {}
    for k in keys:
        if k in hist.columns:
            s = hist[k].dropna()
            n = min(50, len(s))
            result[k] = float(s.iloc[-n:].mean()) if n > 0 else float("nan")
        else:
            result[k] = float("nan")
    return result


def print_gradient_table(canon_hist, llama_hist):
    keys = [grad_key(l) for l in range(N_LAYERS)]
    canon_vals = last_step_values(canon_hist, keys)
    llama_vals = last_step_values(llama_hist, keys)
    print(f"\n{'='*60}")
    print("Gradient contribution per layer (end of training, trailing avg)")
    print(f"{'Layer':>6}  {'Llama':>10}  {'Canon':>10}  {'Ratio L/C':>12}")
    print(f"{'='*60}")
    llama_arr = np.array([llama_vals[k] for k in keys])
    canon_arr = np.array([canon_vals[k] for k in keys])
    for l in range(N_LAYERS):
        ratio = llama_arr[l] / canon_arr[l] if canon_arr[l] != 0 else float("nan")
        print(f"  L{l:>3}  {llama_arr[l]:>10.4f}  {canon_arr[l]:>10.4f}  {ratio:>12.2f}")
    deep_shallow_llama = llama_arr[-1] / llama_arr[0] if llama_arr[0] != 0 else float("nan")
    deep_shallow_canon = canon_arr[-1] / canon_arr[0] if canon_arr[0] != 0 else float("nan")
    print(f"\n  Deep/shallow ratio  Llama={deep_shallow_llama:.2f}×  Canon={deep_shallow_canon:.2f}×")


def print_kurtosis_table(canon_hist, llama_hist):
    keys = [kurtosis_key(l) for l in range(N_LAYERS)]
    canon_vals = last_step_values(canon_hist, keys)
    llama_vals = last_step_values(llama_hist, keys)
    print(f"\n{'='*60}")
    print("Activation kurtosis per layer (end of training)")
    print(f"{'Layer':>6}  {'Llama':>10}  {'Canon':>10}")
    print(f"{'='*60}")
    for l in range(N_LAYERS):
        lk, ck = llama_vals[kurtosis_key(l)], canon_vals[kurtosis_key(l)]
        print(f"  L{l:>3}  {lk:>10.2f}  {ck:>10.2f}")


def print_canon_weights(canon_hist):
    print(f"\n{'='*60}")
    print("Canon weights at end of training (averaged over layers 1–7)")
    print("Expect: shift_3 > shift_2 > shift_1 > shift_0")
    print(f"{'='*60}")
    for ctype in CANON_TYPES:
        shift_means = []
        for shift in range(KERNEL_SIZE):
            vals = []
            for l in range(1, N_LAYERS):
                k = weight_key(ctype, l, shift)
                if k in canon_hist.columns:
                    s = canon_hist[k].dropna()
                    if not s.empty:
                        vals.append(float(s.iloc[-50:].mean()))
            shift_means.append(np.nanmean(vals) if vals else float("nan"))
        print(f"  Canon-{ctype}:  " + "  ".join(f"shift_{i}={v:.4f}" for i, v in enumerate(shift_means)))


def make_gradient_plot(canon_hist, llama_hist):
    keys = [grad_key(l) for l in range(N_LAYERS)]
    def get_mid(hist):
        n = len(hist)
        mid_start = max(0, n // 2 - 25)
        vals = []
        for k in keys:
            if k in hist.columns:
                s = hist[k].dropna()
                if len(s) > mid_start + 5:
                    vals.append(float(s.iloc[mid_start:mid_start+50].mean()))
                else:
                    vals.append(float("nan"))
            else:
                vals.append(float("nan"))
        return np.array(vals)

    canon_mid = get_mid(canon_hist)
    llama_mid  = get_mid(llama_hist)
    layers = np.arange(N_LAYERS)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(layers, llama_mid, "o-", color="#2166ac", label="Llama", linewidth=2)
    ax.plot(layers, canon_mid, "s-", color="#d6604d", label="Canon-ABCD", linewidth=2)
    ax.set_xlabel("Layer index"); ax.set_ylabel("Gradient contribution")
    ax.set_title("Gradient contribution per layer (mid-training)\nCanon produces a much flatter profile than Llama")
    ax.legend(); ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xticks(layers)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "gradient_uniformity.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"Saved: {path}")


def make_kurtosis_plot(canon_hist, llama_hist):
    keys_c = [kurtosis_key(l) for l in range(N_LAYERS)]
    canon_vals = np.array([last_step_values(canon_hist, keys_c)[k] for k in keys_c])
    llama_vals = np.array([last_step_values(llama_hist, keys_c)[k] for k in keys_c])
    layers = np.arange(N_LAYERS)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(layers, llama_vals, "o-", color="#2166ac", label="Llama", linewidth=2)
    ax.plot(layers, canon_vals, "s-", color="#d6604d", label="Canon-ABCD", linewidth=2)
    ax.set_xlabel("Layer index"); ax.set_ylabel("Kurtosis")
    ax.set_title("Activation kurtosis per layer\nCanon suppresses outlier features (except layer 0)")
    ax.legend(); ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xticks(layers)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "kurtosis.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"Saved: {path}")


def make_canon_weights_plot(canon_hist):
    layers = np.arange(N_LAYERS)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i, ctype in enumerate(CANON_TYPES):
        ax = axes[i // 2][i % 2]
        for shift in range(KERNEL_SIZE):
            vals = []
            for l in range(N_LAYERS):
                k = weight_key(ctype, l, shift)
                if k in canon_hist.columns:
                    s = canon_hist[k].dropna()
                    vals.append(float(s.iloc[-50:].mean()) if not s.empty else float("nan"))
                else:
                    vals.append(float("nan"))
            ax.plot(layers, vals, "o-", color=colors[shift], label=f"shift {shift}", linewidth=2)
        ax.set_title(f"Canon-{ctype}: learned weights by layer")
        ax.set_xlabel("Layer"); ax.set_ylabel("Weight value")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_xticks(layers)
    fig.suptitle("Canon learned weights (end of training)\nFarther shifts (larger index) consistently receive larger weights",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "canon_weights_layerwise.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"Saved: {path}")


def make_rms_ratio_plot(canon_hist):
    layers = np.arange(N_LAYERS)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, ctype in enumerate(["A", "C"]):   # A and C are the strongest
        vals = []
        for l in range(N_LAYERS):
            k = rms_key(ctype, l)
            if k in canon_hist.columns:
                s = canon_hist[k].dropna()
                vals.append(float(s.iloc[-50:].mean()) if not s.empty else float("nan"))
            else:
                vals.append(float("nan"))
        ax.plot(layers, vals, "o-", color=colors[i], label=f"Canon-{ctype}", linewidth=2)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="identity (ratio=1)")
    ax.set_xlabel("Layer"); ax.set_ylabel("RMS ratio (output / input)")
    ax.set_title("Canon RMS ratio: output RMS / input RMS\nCanon amplifies representations; effect grows with depth")
    ax.legend(); ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xticks(layers)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "rms_ratio.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"Saved: {path}")


def make_cos_sim_plot(canon_hist):
    layers = np.arange(N_LAYERS)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(figsize=(8, 4))
    # Only A and C have cos_sim metrics (from the metric inspection)
    for i, ctype in enumerate(["A", "C"]):
        vals = []
        for l in range(N_LAYERS):
            k = cos_key(ctype, l)
            if k in canon_hist.columns:
                s = canon_hist[k].dropna()
                vals.append(float(s.iloc[-50:].mean()) if not s.empty else float("nan"))
            else:
                vals.append(float("nan"))
        ax.plot(layers, vals, "o-", color=colors[i], label=f"Canon-{ctype}", linewidth=2)
    ax.set_xlabel("Layer"); ax.set_ylabel("Cosine similarity (output vs input)")
    ax.set_title("Canon cosine similarity\nDeeper layers approach identity; A and C mix more in early layers")
    ax.legend(); ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xticks(layers); ax.set_ylim(0.5, 1.05)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "cosine_similarity.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"Saved: {path}")


# ── Exp 5: Temporal evolution plots ───────────────────────────────────────────

def plot_gradient_ratio_over_time(canon_hist, llama_hist):
    """
    Plot deep/shallow gradient ratio (L7/L0) over training steps for Canon and Llama.
    Shows whether Canon's gradient uniformity is present from step 1 or builds up.
    """
    def ratio_series(hist):
        deep_k  = grad_key(N_LAYERS - 1)
        shall_k = grad_key(0)
        if deep_k not in hist.columns or shall_k not in hist.columns:
            return pd.Series(dtype=float), pd.Index([])
        df = hist[["_step", deep_k, shall_k]].dropna()
        ratio = df[deep_k] / df[shall_k].replace(0, float("nan"))
        return ratio.rolling(15, min_periods=1).mean(), df["_step"]

    c_ratio, c_steps = ratio_series(canon_hist)
    l_ratio, l_steps = ratio_series(llama_hist)

    fig, ax = plt.subplots(figsize=(9, 4))
    if len(c_steps): ax.plot(c_steps, c_ratio, color="#d6604d", lw=2, label="Canon-ABCD")
    if len(l_steps): ax.plot(l_steps, l_ratio, color="#2166ac", lw=2, label="Llama")
    ax.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.6, label="ratio = 1 (uniform)")
    ax.set_xlabel("Training step"); ax.set_ylabel("Gradient ratio  L7 / L0")
    ax.set_title("Exp 5: Deep/shallow gradient ratio over training (text modeling)\n"
                 "Does Canon's gradient uniformity appear immediately or build up?")
    ax.legend(); ax.grid(True, alpha=0.3, ls="--")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "gradient_ratio_over_time.png")
    fig.savefig(path, dpi=150); plt.close(fig); print(f"Saved: {path}")


def plot_canon_weights_over_time(canon_hist):
    """
    For Canon types A and C (the active positions), plot shift weights over training
    steps at layers 0, 3, 7.  Shows whether the farther-shift dominance pattern
    (shift_2 > shift_1 > shift_0) is present from the start or learned gradually.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    probe_layers = [0, 3, 7]

    for row, ct in enumerate(["A", "C"]):
        for col, layer in enumerate(probe_layers):
            ax = axes[row, col]
            for shift in range(KERNEL_SIZE):
                k = weight_key(ct, layer, shift)
                if k not in canon_hist.columns:
                    continue
                s = canon_hist[["_step", k]].dropna()
                smooth = s[k].rolling(15, min_periods=1).mean()
                ax.plot(s["_step"], smooth, color=colors[shift],
                        lw=1.8, label=f"shift {shift}")
            ax.set_title(f"Canon-{ct}  layer {layer}")
            ax.set_xlabel("Step"); ax.set_ylabel("Weight value")
            ax.legend(fontsize=7); ax.grid(True, alpha=0.3, ls="--")

    fig.suptitle("Exp 5: Canon weight evolution over training (text modeling)\n"
                 "Does shift_2 dominance appear early (inductive bias) or late (learned)?",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "canon_weights_over_time.png")
    fig.savefig(path, dpi=150); plt.close(fig); print(f"Saved: {path}")


def print_temporal_summary(canon_hist, llama_hist):
    """Print gradient ratio at early, mid, and end of training."""
    print(f"\n{'='*60}")
    print("Exp 5: Gradient deep/shallow ratio at training checkpoints")
    print(f"{'='*60}")
    deep_k  = grad_key(N_LAYERS - 1)
    shall_k = grad_key(0)
    for label, hist in [("Llama", llama_hist), ("Canon", canon_hist)]:
        if deep_k not in hist.columns or shall_k not in hist.columns:
            print(f"  {label}: no grad data"); continue
        df = hist[["_step", deep_k, shall_k]].dropna()
        n = len(df)
        for phase, idx in [("early (10%)", slice(0, max(1, n//10))),
                            ("mid   (50%)", slice(max(0, n//2-5), max(1, n//2+5))),
                            ("end   (95%)", slice(max(0, n-n//10), n))]:
            sub = df.iloc[idx]
            ratio = (sub[deep_k] / sub[shall_k].replace(0, float("nan"))).mean()
            step  = int(sub["_step"].mean())
            print(f"  {label}  {phase}  step≈{step:>7}  ratio={ratio:.2f}×")

    print(f"\n  Canon weight onset at layer 0 (type A):")
    for shift in range(KERNEL_SIZE):
        k = weight_key("A", 0, shift)
        if k not in canon_hist.columns: continue
        s = canon_hist[["_step", k]].dropna()
        n = len(s)
        early_val = s[k].iloc[:max(1, n//10)].mean()
        late_val  = s[k].iloc[-max(1, n//10):].mean()
        print(f"    shift {shift}: early={early_val:.3f}  late={late_val:.3f}  "
              f"change={late_val-early_val:+.3f}")


if __name__ == "__main__":
    # Build full key list
    _grad_keys    = [grad_key(l) for l in range(N_LAYERS)]
    _kurtosis_keys = [kurtosis_key(l) for l in range(N_LAYERS)]
    _rms_keys     = [rms_key(ct, l) for ct in ["A", "C"] for l in range(N_LAYERS)]
    _cos_keys     = [cos_key(ct, l) for ct in ["A", "C"] for l in range(N_LAYERS)]
    _weight_keys  = [weight_key(ct, l, s) for ct in CANON_TYPES for l in range(N_LAYERS) for s in range(KERNEL_SIZE)]

    all_canon_keys = _grad_keys + _kurtosis_keys + _rms_keys + _cos_keys + _weight_keys + ["loss/out"]
    all_llama_keys = _grad_keys + _kurtosis_keys + ["loss/out"]

    print(f"Fetching Canon run ({CANON_RUN})...")
    canon_hist = fetch_run_history(CANON_RUN, keys=all_canon_keys, samples=1000)
    print(f"  rows={len(canon_hist)}  step_range={canon_hist['_step'].min():.0f}–{canon_hist['_step'].max():.0f}")

    print(f"\nFetching Llama run ({LLAMA_RUN})...")
    llama_hist = fetch_run_history(LLAMA_RUN, keys=all_llama_keys, samples=1000)
    print(f"  rows={len(llama_hist)}  step_range={llama_hist['_step'].min():.0f}–{llama_hist['_step'].max():.0f}")

    # Existing end-of-training analysis
    print_gradient_table(canon_hist, llama_hist)
    print_kurtosis_table(canon_hist, llama_hist)
    print_canon_weights(canon_hist)

    make_gradient_plot(canon_hist, llama_hist)
    make_kurtosis_plot(canon_hist, llama_hist)
    make_canon_weights_plot(canon_hist)
    make_rms_ratio_plot(canon_hist)
    make_cos_sim_plot(canon_hist)

    # Exp 5: temporal evolution
    print("\n=== Exp 5: Temporal evolution plots ===")
    print_temporal_summary(canon_hist, llama_hist)
    plot_gradient_ratio_over_time(canon_hist, llama_hist)
    plot_canon_weights_over_time(canon_hist)

    print("\nDone.")
