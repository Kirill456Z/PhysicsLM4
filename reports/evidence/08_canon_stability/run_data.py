"""
Exp 2 — Canon stability on synthetic tasks (depo-only).

Compares seed variance of Canon-ABCD vs Llama on depo×2 (depo_edges_list + depo_adj_list)
with 5 seeds each, to isolate whether the architecture itself reduces training variance
independent of task mixing.

New runs (5 seeds each):
  Canon: canon_stability_depo_canon_s{1..5}_0.4.{202..206}
  Llama: canon_stability_depo_llama_s{1..5}_0.4.{207..211}

Reference (from F5 — Llama depo×2, 3 seeds, same config):
  seed_variance_exps_remaining_only_depo_{55,56,57}_0.4.{186,187,188}
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import fetch_training_history, plot_variance_curves

WANDB_API_KEY = "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"
ENTITY = "kirill456z"
PROJECT = "physics4llm"
OUT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Run IDs ────────────────────────────────────────────────────────────────────
CANON_IDS = [
    "canon_stability_depo_canon_s1_0.4.202",
    "canon_stability_depo_canon_s2_0.4.203",
    "canon_stability_depo_canon_s3_0.4.204",
    "canon_stability_depo_canon_s4_0.4.205",
    "canon_stability_depo_canon_s5_0.4.206",
]
LLAMA_IDS = [
    "canon_stability_depo_llama_s1_0.4.207",
    "canon_stability_depo_llama_s2_0.4.208",
    "canon_stability_depo_llama_s3_0.4.209",
    "canon_stability_depo_llama_s4_0.4.210",
    "canon_stability_depo_llama_s5_0.4.211",
]
# F5 reference: Llama depo×2, 3 seeds (for historical comparison)
F5_LLAMA_DEPO_IDS = [
    "seed_variance_exps_remaining_only_depo_55_0.4.186",
    "seed_variance_exps_remaining_only_depo_56_0.4.187",
    "seed_variance_exps_remaining_only_depo_57_0.4.188",
]

GROUPS = {
    "Canon depo×2 (new, 5 seeds)": CANON_IDS,
    "Llama depo×2 (new, 5 seeds)": LLAMA_IDS,
    "Llama depo×2 (F5 ref, 3 seeds)": F5_LLAMA_DEPO_IDS,
}


def print_variance_table(histories):
    print(f"\n{'='*68}")
    print("SEED VARIANCE TABLE  (loss/out, trailing 100-step average per seed)")
    print(f"{'='*68}")
    print(f"{'Group':>35}  {'n':>4}  {'mean':>8}  {'std':>8}  {'min':>8}  {'max':>8}")
    for group, runs in histories.items():
        vals = []
        for rid, hist in runs.items():
            if hist.empty or "loss/out" not in hist.columns:
                continue
            s = hist["loss/out"].dropna()
            n = min(100, len(s))
            if n > 0:
                vals.append(float(s.iloc[-n:].mean()))
        if not vals:
            print(f"  {group:>33}  NO DATA")
            continue
        print(f"  {group:>33}  {len(vals):>4}  {np.mean(vals):>8.4f}  "
              f"{np.std(vals, ddof=1) if len(vals)>1 else 0:>8.4f}  "
              f"{min(vals):>8.4f}  {max(vals):>8.4f}")
        for i, v in enumerate(vals):
            print(f"    seed {i+1}: {v:.4f}")
    print()
    # Key comparison
    canon_vals = []
    llama_vals = []
    for rid, hist in histories["Canon depo×2 (new, 5 seeds)"].items():
        s = hist["loss/out"].dropna() if "loss/out" in hist.columns else pd.Series()
        n = min(100, len(s))
        if n > 0: canon_vals.append(float(s.iloc[-n:].mean()))
    for rid, hist in histories["Llama depo×2 (new, 5 seeds)"].items():
        s = hist["loss/out"].dropna() if "loss/out" in hist.columns else pd.Series()
        n = min(100, len(s))
        if n > 0: llama_vals.append(float(s.iloc[-n:].mean()))

    c_std = np.std(canon_vals, ddof=1) if len(canon_vals) > 1 else 0
    l_std = np.std(llama_vals, ddof=1) if len(llama_vals) > 1 else 0
    print(f"  Canon std = {c_std:.4f}   Llama std = {l_std:.4f}")
    if c_std < l_std:
        print(f"  → Canon is MORE STABLE  (std ratio Canon/Llama = {c_std/l_std:.2f}×)")
        print(f"  → Architecture effect: Canon reduces seed variance by {(1-c_std/l_std)*100:.0f}%")
    elif c_std > l_std * 1.1:
        print(f"  → Canon is LESS STABLE  (std ratio Canon/Llama = {c_std/l_std:.2f}×)")
        print(f"  → Architecture does NOT reduce variance; task mixing is a data-only fix")
    else:
        print(f"  → Canon and Llama have SIMILAR variance (std ratio = {c_std/l_std:.2f}×)")
        print(f"  → Architecture does not meaningfully affect stability")


def plot_training_curves(histories):
    palette = ["#d6604d", "#2166ac", "#aaaaaa"]
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_variance_curves(
        histories, "loss/out", ax=ax,
        title="Depo×2 training loss: Canon vs Llama (5 seeds each)\n"
              "Does Canon reduce seed-to-seed variance?",
        smoothing=15, alpha_individual=0.25, show_individual=True,
        palette=palette,
    )
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:g}"))
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "canon_vs_llama_depo_variance_curves.png")
    fig.savefig(path, dpi=150); plt.close(fig); print(f"Saved: {path}")


def plot_variance_bars(histories):
    """Bar chart of std across seeds for each group."""
    groups = list(histories.keys())
    stds, means = [], []
    for group in groups:
        vals = []
        for rid, hist in histories[group].items():
            if hist.empty or "loss/out" not in hist.columns:
                continue
            s = hist["loss/out"].dropna()
            n = min(100, len(s))
            if n > 0: vals.append(float(s.iloc[-n:].mean()))
        stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
        means.append(np.mean(vals) if vals else float("nan"))

    colors = ["#d6604d", "#2166ac", "#aaaaaa"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: std bars
    ax = axes[0]
    bars = ax.bar(groups, stds, color=colors[:len(groups)], alpha=0.8, edgecolor="white")
    ax.set_ylabel("Std of final loss/out across seeds")
    ax.set_title("Seed variance: Canon vs Llama on depo×2")
    ax.tick_params(axis="x", rotation=15, labelsize=8)
    ax.grid(axis="y", alpha=0.35)
    for bar, val in zip(bars, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    # Right: mean ± std
    ax = axes[1]
    xs = np.arange(len(groups))
    for i, (g, m, s, c) in enumerate(zip(groups, means, stds, colors)):
        ax.bar(i, m, yerr=s, color=c, alpha=0.75, capsize=5)
    ax.set_xticks(xs)
    ax.set_xticklabels(groups, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Final loss/out (mean ± std)")
    ax.set_title("Mean ± std final loss by group")
    ax.grid(axis="y", alpha=0.35)

    fig.suptitle("Does Canon reduce seed variance on depo-only training?", fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "variance_bars_canon_vs_llama_depo.png")
    fig.savefig(path, dpi=150); plt.close(fig); print(f"Saved: {path}")


def plot_depo_eval(histories):
    """Plot depo hop_4 accuracy (primary eval metric) across seeds."""
    metric = "evals/synthetic/depo_edges_list/hop_4/accuracy"
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"Canon depo×2 (new, 5 seeds)": "#d6604d",
              "Llama depo×2 (new, 5 seeds)": "#2166ac",
              "Llama depo×2 (F5 ref, 3 seeds)": "#aaaaaa"}
    any_data = False
    for group, runs in histories.items():
        color = colors.get(group, "gray")
        final_vals = []
        for rid, hist in runs.items():
            if hist.empty or metric not in hist.columns:
                continue
            s = hist[[col for col in hist.columns if col in ["_step", metric]]].dropna()
            if s.empty: continue
            ax.plot(s["_step"], s[metric], color=color, alpha=0.3, lw=1)
            final_vals.append(float(s[metric].iloc[-20:].mean()))
            any_data = True
        if final_vals:
            ax.axhline(np.mean(final_vals), color=color, ls="--", lw=2,
                       label=f"{group} (mean={np.mean(final_vals):.3f})")
    if any_data:
        ax.set_xlabel("Step"); ax.set_ylabel("Depo hop-4 accuracy")
        ax.set_title("Depo eval (hop-4 accuracy) across seeds: Canon vs Llama")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, ls="--")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "depo_hop4_accuracy_canon_vs_llama.png")
    fig.savefig(path, dpi=150); plt.close(fig); print(f"Saved: {path}")


if __name__ == "__main__":
    kw = dict(api_key=WANDB_API_KEY, entity=ENTITY, project=PROJECT, samples=300)
    metric_keys = ["loss/out",
                   "evals/synthetic/depo_edges_list/hop_4/accuracy",
                   "evals/synthetic/depo_adj_list/hop_4/accuracy"]

    print("Fetching Canon stability runs…")
    histories = fetch_training_history(GROUPS, metric_keys=metric_keys, **kw)

    print_variance_table(histories)
    plot_training_curves(histories)
    plot_variance_bars(histories)
    plot_depo_eval(histories)

    print("\nDone.")
