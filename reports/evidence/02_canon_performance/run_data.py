"""
F2: Canon improves the Llama baseline on text modeling; Canon-AC ≈ Canon-ABCD.

Fetches training loss curves for Canon-ABCD, Canon-AC, and Llama on text modeling.
Prints final loss values and crossover step, then saves plots.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from utils import fetch_training_history, plot_variance_curves

WANDB_API_KEY = "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"
ENTITY = "kirill456z"
PROJECT = "physics4llm"
OUT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(OUT_DIR, exist_ok=True)

RUNS = {
    "Llama baseline": [
        "canon_text_0.1.130",   # display: text_llama_bs_128_seq_len_2048_0.1.130
    ],
    "Canon-ABCD": [
        "canon_text_0.1.129",   # display: text_ks_4_default_init_with_residual_trainable_bs_128_seq_len_2048_0.1.129
    ],
    "Canon-AC": [
        "text_ks_4_default_init_with_residual_trainable_bs_128_seq_len_2048_0.1.138",  # display: ...AC...
    ],
}

PALETTE = ["#2166ac", "#d6604d", "#f4a582"]   # blue=Llama, dark-red=ABCD, light-red=AC


def print_summary(histories):
    print(f"\n{'='*60}")
    print("Final-loss summary (trailing 100 steps)")
    print(f"{'='*60}")
    for group, runs in histories.items():
        for rid, hist in runs.items():
            if hist.empty or "loss/out" not in hist.columns:
                print(f"  {group:20s}  NO DATA")
                continue
            s = hist["loss/out"].dropna()
            final = float(s.iloc[-100:].mean()) if len(s) >= 100 else float(s.mean())
            print(f"  {group:20s}  final_loss={final:.4f}  steps={int(hist['_step'].max())}")

    # Detect crossover step (where Canon-ABCD first goes below Llama)
    abcd_runs = list(histories["Canon-ABCD"].values())
    llama_runs = list(histories["Llama baseline"].values())
    if abcd_runs and llama_runs:
        h_abcd = abcd_runs[0][["_step", "loss/out"]].dropna()
        h_llama = llama_runs[0][["_step", "loss/out"]].dropna()
        merged = h_abcd.merge(h_llama, on="_step", suffixes=("_canon", "_llama"))
        merged = merged.dropna()
        if not merged.empty:
            cross = merged[merged["loss/out_canon"] < merged["loss/out_llama"]]
            if not cross.empty:
                print(f"\n  Canon-ABCD crosses below Llama at step ≈ {int(cross['_step'].iloc[0])}")


def make_plots(histories):
    # --- Plot 1: Canon-ABCD vs Llama ---
    subset = {k: v for k, v in histories.items() if k != "Canon-AC"}
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_variance_curves(
        subset, "loss/out", ax=ax,
        title="Canon-ABCD vs Llama baseline — text modeling",
        smoothing=15, alpha_individual=0.5,
        palette=[PALETTE[0], PALETTE[1]],
    )
    ax.set_ylim(0.72, 1.05)
    ax.axhline(0.790, color=PALETTE[0], linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(0.776, color=PALETTE[1], linestyle="--", linewidth=0.8, alpha=0.5)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "canon_vs_llama.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")

    # --- Plot 2: All three (AC vs ABCD) ---
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_variance_curves(
        histories, "loss/out", ax=ax,
        title="Canon-AC ≈ Canon-ABCD — pre-attention and pre-FFN positions capture most of the gain",
        smoothing=15, alpha_individual=0.5,
        palette=PALETTE,
    )
    ax.set_ylim(0.72, 1.05)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "ac_vs_abcd.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    print("Fetching runs...")
    histories = fetch_training_history(
        RUNS, metric_keys=["loss/out"],
        api_key=WANDB_API_KEY, entity=ENTITY, project=PROJECT, samples=500,
    )
    print_summary(histories)
    make_plots(histories)
    print("\nDone.")
