"""
F1: Synthetic benchmarks have high seed-to-seed variance; text modeling is stable.

Fetches training loss curves for:
  - Text modeling (stable): 2 Llama + 2 Canon seeds
  - Depo 8L (noisy): 3 Canon + 1 Llama seeds
  - Depo 12L (grokking): 2 Canon seeds
  - Brevo (noisy): 3 Llama seeds
Prints final-loss summary and seed variance, then saves plots.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from utils import fetch_training_history, plot_variance_curves

WANDB_API_KEY = "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"
ENTITY = "kirill456z"
PROJECT = "physics4llm"
OUT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Run definitions
# ---------------------------------------------------------------------------
TEXT_RUNS = {
    "Llama (text)": [
        "text_llama_bs_128_seq_len_2048_0.1.136",
        "text_ks_4_const_var_sqr_init_with_residual_trainable_bs_128_seq_len_2048_0.1.140",
    ],
    "Canon (text)": [
        "text_ks_4_const_var_sqr_init_with_residual_trainable_bs_128_seq_len_2048_0.1.147",
        "canon_text_0.1.130",   # display: text_llama_bs_128_seq_len_2048_0.1.130
    ],
}

DEPO_8L_RUNS = {
    "Canon 8L (depo)": [
        "depo_ks_4_default_init_with_residual_trainable_0.1.120",
        "depo_ks_4_default_init_with_residual_trainable_0.1.121",
        "depo_ks_4_default_init_with_residual_trainable_0.1.122",
    ],
    "Llama 8L (depo)": [
        "llama_0.1.126",   # display: depo_llama_8_hops_100_nodes_bs_256_seq_len_768_0.1.126
    ],
}

DEPO_12L_RUNS = {
    "Canon 12L (depo)": [
        "depo_ks_4_default_init_with_residual_trainable_8_hops_100_nodes_bs_128_seq_len_1024_0.1.155",
        "depo_ks_4_default_init_with_residual_trainable_8_hops_100_nodes_bs_128_seq_len_1024_0.1.156",
    ],
}

BREVO_RUNS = {
    "Llama (brevo)": [
        "brevo_llama_110_nodes_single_bs_256_seq_len_1024_0.1.217",
        "brevo_llama_110_nodes_single_bs_256_seq_len_1024_0.1.218",
        "brevo_llama_110_nodes_single_bs_256_seq_len_1024_0.1.219",
    ],
}


def print_summary(histories: dict, metric: str = "loss/out", trailing: int = 50):
    print(f"\n{'='*60}")
    print(f"Summary: {metric}  (trailing-{trailing}-step avg)")
    print(f"{'='*60}")
    for group, runs in histories.items():
        vals = []
        for rid, hist in runs.items():
            if hist.empty or metric not in hist.columns:
                continue
            s = hist[metric].dropna()
            if s.empty:
                continue
            vals.append(float(s.iloc[-trailing:].mean()))
        if vals:
            print(f"  {group:40s}  mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  n={len(vals)}")
            for i, v in enumerate(vals):
                print(f"    seed {i}: {v:.4f}")
        else:
            print(f"  {group:40s}  NO DATA")


def fetch_all():
    kw = dict(api_key=WANDB_API_KEY, entity=ENTITY, project=PROJECT, samples=300)
    print("\n--- Fetching text runs ---")
    text_hist = fetch_training_history(TEXT_RUNS, metric_keys=["loss/out"], **kw)
    print("\n--- Fetching Depo 8L runs ---")
    depo8_hist = fetch_training_history(DEPO_8L_RUNS, metric_keys=["loss/out"], **kw)
    print("\n--- Fetching Depo 12L runs ---")
    depo12_hist = fetch_training_history(DEPO_12L_RUNS, metric_keys=["loss/out"], **kw)
    print("\n--- Fetching Brevo runs ---")
    brevo_hist = fetch_training_history(BREVO_RUNS, metric_keys=["loss/out"], **kw)
    return text_hist, depo8_hist, depo12_hist, brevo_hist


def make_plots(text_hist, depo8_hist, depo12_hist, brevo_hist):
    palette_text = ["#2166ac", "#d6604d"]   # blue=Llama, red=Canon
    palette_single = ["#d6604d"]
    palette_brevo = ["#4dac26"]

    # --- Plot 1: Text modeling (stable) ---
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_variance_curves(
        text_hist, "loss/out", ax=ax,
        title="Text modeling — same model, different seeds (stable)",
        smoothing=10, alpha_individual=0.3, palette=palette_text,
    )
    ax.set_ylim(0.70, 1.0)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "text_stable.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")

    # --- Plot 2: Depo 8L (noisy) ---
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_variance_curves(
        depo8_hist, "loss/out", ax=ax,
        title="Depo 8L/512D — high seed-to-seed variance",
        smoothing=5, alpha_individual=0.4,
    )
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "depo_8l_noisy.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")

    # --- Plot 3: Depo 12L (grokking) ---
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_variance_curves(
        depo12_hist, "loss/out", ax=ax,
        title="Depo 12L/768D — grokking: one seed stagnates, one drops sharply",
        smoothing=5, alpha_individual=0.7, show_individual=True,
    )
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "depo_12l_grokking.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")

    # --- Plot 4: Brevo (noisy) ---
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_variance_curves(
        brevo_hist, "loss/out", ax=ax,
        title="Brevo 110 nodes — three seeds diverge (same instability as Depo)",
        smoothing=5, alpha_individual=0.7, palette=palette_brevo,
    )
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "brevo_noisy.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")

    # --- Plot 5: Side-by-side comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_variance_curves(
        text_hist, "loss/out", ax=axes[0],
        title="Text modeling (stable)",
        smoothing=10, alpha_individual=0.3, palette=palette_text,
    )
    axes[0].set_ylim(0.70, 1.0)

    combined_synth = {**depo8_hist, **brevo_hist}
    plot_variance_curves(
        combined_synth, "loss/out", ax=axes[1],
        title="Synthetic tasks: Depo 8L + Brevo (high variance)",
        smoothing=5, alpha_individual=0.4,
    )
    fig.suptitle("Seed variance: text modeling vs synthetic benchmarks", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "comparison_stable_vs_noisy.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    text_hist, depo8_hist, depo12_hist, brevo_hist = fetch_all()
    print_summary(text_hist)
    print_summary(depo8_hist)
    print_summary(depo12_hist)
    print_summary(brevo_hist)
    make_plots(text_hist, depo8_hist, depo12_hist, brevo_hist)
    print("\nDone.")
