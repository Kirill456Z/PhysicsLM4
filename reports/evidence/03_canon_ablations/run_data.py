"""
F3: Canon ablations — kernel size, initialization, layer depth, normalization type.

Fetches training loss curves for each ablation group.
Prints final-loss table per ablation, then saves a 2×2 panel figure.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from utils import fetch_training_history

WANDB_API_KEY = "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"
ENTITY = "kirill456z"
PROJECT = "physics4llm"
OUT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Run definitions per ablation
# ---------------------------------------------------------------------------
KERNEL_SIZE_RUNS = {
    "k=1 (no cross-token)": ["canon_text_0.1.117"],   # display: text_ks_1_default_init...0.1.117
    "Llama baseline":        ["text_llama_bs_128_seq_len_2048_0.1.136"],
    "k=2":                   ["text_ks_2_const_var_sqr_init_with_residual_trainable_bs_128_seq_len_2048_0.1.164"],
    "k=4":                   ["text_ks_4_const_var_sqr_init_with_residual_trainable_bs_128_seq_len_2048_0.1.147"],
    "k=6":                   ["text_ks_6_const_var_sqr_init_with_residual_trainable_bs_128_seq_len_2048_0.1.148"],
}

INIT_RUNS = {
    "Llama baseline":  ["text_llama_bs_128_seq_len_2048_0.1.136"],
    "zeros init":      ["text_ks_4_zeros_init_with_residual_trainable_bs_128_seq_len_2048_0.1.162"],
    "default init":    ["canon_text_0.1.129"],   # display: text_ks_4_default_init...0.1.129
    "const_var (1/√k)":["text_ks_4_const_var_init_with_residual_trainable_bs_128_seq_len_2048_0.1.163"],
    "const_var_sqrt (1/k)": ["text_ks_4_const_var_sqr_init_with_residual_trainable_bs_128_seq_len_2048_0.1.147"],
}

LAYER_DEPTH_RUNS = {
    "Llama baseline":  ["text_llama_bs_128_seq_len_2048_0.1.206"],
    "First layer only":["text_ks_4_default_init_with_residual_trainable_bs_128_seq_len_2048_0.1.195"],  # display: ...first_layer_only...
    "Last layer only": ["text_ks_4_default_init_with_residual_trainable_bs_128_seq_len_2048_0.1.197"],  # display: ...last_layer_only...
    "All layers":      ["text_ks_4_default_init_with_residual_trainable_bs_128_seq_len_2048_0.1.207"],
}

NORM_TYPE_RUNS = {
    "Llama Pre-LN":   ["text_llama_bs_128_seq_len_2048_0.1.206"],
    "Canon Pre-LN":   ["text_ks_4_default_init_with_residual_trainable_bs_128_seq_len_2048_0.1.207"],
    "Llama Post-LN":  ["text_llama_bs_128_seq_len_2048_0.1.198"],   # display: ...post_norm...
    "Canon Post-LN":  ["text_ks_4_default_init_with_residual_trainable_bs_128_seq_len_2048_0.1.196"],  # display: ...post_ln...
    "Llama Peri-LN":  ["text_llama_bs_128_seq_len_2048_0.1.212"],   # display: ...peri_ln...
    "Canon Peri-LN":  ["text_ks_4_default_init_with_residual_trainable_bs_128_seq_len_2048_0.1.213"],  # display: ...peri_ln...
}


def _final_loss(hist, trailing=100):
    s = hist["loss/out"].dropna() if "loss/out" in hist.columns else None
    if s is None or s.empty:
        return float("nan")
    n = min(trailing, len(s))
    return float(s.iloc[-n:].mean())


def print_ablation_table(name, histories):
    print(f"\n--- {name} ---")
    for group, runs in histories.items():
        vals = [_final_loss(hist) for hist in runs.values() if not hist.empty]
        mean_val = np.nanmean(vals) if vals else float("nan")
        print(f"  {group:30s}  loss={mean_val:.4f}")


def plot_ablation(ax, histories, title, palette=None):
    default_colors = [p["color"] for p in plt.rcParams["axes.prop_cycle"]]
    colors = palette or default_colors
    for i, (group, runs) in enumerate(histories.items()):
        color = colors[i % len(colors)]
        for rid, hist in runs.items():
            if hist.empty or "loss/out" not in hist.columns:
                continue
            s = hist[["_step", "loss/out"]].dropna()
            s = s.copy()
            s["loss/out"] = s["loss/out"].ewm(alpha=0.2).mean()
            s = s[s["loss/out"] < 1.0]
            ax.plot(s["_step"], s["loss/out"], color=color, linewidth=1.8, label=group)
    ax.set_xlabel("Step")
    ax.set_ylabel("loss/out")
    ax.set_yscale("log")
    ax.set_ylim(top=1.0)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, linestyle="--")


if __name__ == "__main__":
    fetch_kw = dict(
        metric_keys=["loss/out"],
        api_key=WANDB_API_KEY, entity=ENTITY, project=PROJECT, samples=400,
    )

    print("Fetching kernel size runs...")
    ks_hist = fetch_training_history(KERNEL_SIZE_RUNS, **fetch_kw)
    print("Fetching init runs...")
    init_hist = fetch_training_history(INIT_RUNS, **fetch_kw)
    print("Fetching layer depth runs...")
    depth_hist = fetch_training_history(LAYER_DEPTH_RUNS, **fetch_kw)
    print("Fetching norm type runs...")
    norm_hist = fetch_training_history(NORM_TYPE_RUNS, **fetch_kw)

    # Print tables
    print_ablation_table("Kernel size", ks_hist)
    print_ablation_table("Initialization", init_hist)
    print_ablation_table("Layer depth", depth_hist)
    print_ablation_table("Normalization type", norm_hist)

    # 2×2 panel figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    plot_ablation(axes[0, 0], ks_hist, "Kernel size: k=1 < baseline < k=2 < k=4 < k=6")
    plot_ablation(axes[0, 1], init_hist, "Initialization: all canon variants beat Llama; 1/k is best")
    plot_ablation(axes[1, 0], depth_hist, "Layer depth: all layers > first only > baseline > last only")

    # Norm type: group canon vs llama by norm type with distinct linestyles
    ax = axes[1, 1]
    norm_colors = {"Pre-LN": "#2166ac", "Post-LN": "#d6604d", "Peri-LN": "#1a9641"}
    norm_styles = {"Llama": "--", "Canon": "-"}
    for group, runs in norm_hist.items():
        # Parse "Llama Pre-LN" → arch=Llama, norm=Pre-LN
        parts = group.split(" ", 1)
        arch, norm = parts[0], parts[1] if len(parts) > 1 else ""
        color = norm_colors.get(norm, "gray")
        ls = norm_styles.get(arch, "-")
        for rid, hist in runs.items():
            if hist.empty or "loss/out" not in hist.columns:
                continue
            s = hist[["_step", "loss/out"]].dropna().copy()
            s["loss/out"] = s["loss/out"].ewm(alpha=0.98).mean()
            s = s[s["loss/out"] < 1.0]
            ax.plot(s["_step"], s["loss/out"], color=color, linestyle=ls,
                    linewidth=1.8, label=group)
    ax.set_xlabel("Step"); ax.set_ylabel("loss/out")
    ax.set_yscale("log")
    ax.set_ylim(top=1.0)
    ax.set_title("Normalization type: Canon beats Llama in all cases\n(solid=Canon, dashed=Llama; color=norm type)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3, linestyle="--")

    fig.suptitle("Canon ablations", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "canon_ablations_2x2.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {path}")

    # Individual high-res plots for inclusion in LaTeX
    for runs, title, fname in [
        (ks_hist,   "Kernel size ablation", "kernel_size"),
        (init_hist, "Initialization ablation", "initialization"),
        (depth_hist,"Layer depth ablation", "layer_depth"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4))
        plot_ablation(ax, runs, title)
        fig.tight_layout()
        p = os.path.join(OUT_DIR, f"{fname}.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"Saved: {p}")

    print("\nDone.")
