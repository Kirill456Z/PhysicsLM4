"""
F6: Canon scales consistently — improvement over Llama is robust across model sizes (3M–100M).

Fetches summary stats for 5 seeds × 4 sizes × 2 architectures.
Averages over seeds, prints the scaling table, then saves scaling-law plots.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import (
    fetch_runs_results, plot_scaling_law, plot_all_tasks_primary_metrics,
    TASK_NAMES, TASK_PRIMARY_METRIC, wandb_eval_key,
)

WANDB_API_KEY = "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"
ENTITY = "kirill456z"
PROJECT = "physics4llm"
OUT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(OUT_DIR, exist_ok=True)

RUN_IDS = {
    "llama": [
        # 3m
        "scaling_law_exps_3_llama_3m_0.4.84",
        "remaining_runs_llama_3m_seed_55_0.4.134",
        "remaining_runs_llama_3m_seed_56_0.4.135",
        "scaling_law_4_more_seeds_llama_3m_seed_57_0.4.160",
        "scaling_law_4_more_seeds_llama_3m_seed_58_0.4.161",
        # 10m
        "scaling_law_exps_3_llama_10m_0.4.82",
        "remaining_runs_llama_10m_seed_55_0.4.132",
        "remaining_runs_llama_10m_seed_56_0.4.133",
        "scaling_law_4_more_seeds_llama_10m_seed_57_0.4.156",
        "scaling_law_4_more_seeds_llama_10m_seed_58_0.4.157",
        # 30m
        "scaling_law_exps_3_llama_30m_0.4.83",
        "depo_variance_impact_investigation_2_all_tasks_seed_55_0.4.120",
        "depo_variance_impact_investigation_2_all_tasks_seed_56_0.4.121",
        "scaling_law_4_more_seeds_llama_30m_seed_57_0.4.158",
        "scaling_law_4_more_seeds_llama_30m_seed_58_0.4.159",
        # 100m
        "scaling_law_exps_3_llama_100m_0.4.81",
        "seed_variance_comp_llama_seed_55_0.4.99",
        "seed_variance_comp_llama_seed_56_0.4.100",
        "scaling_law_4_more_seeds_llama_100m_seed_57_0.4.154",
        "scaling_law_4_more_seeds_llama_100m_seed_58_0.4.155",
    ],
    "canon": [
        # 3m
        "scaling_law_exps_3_canon_3m_0.4.80",
        "remaining_runs_canon_3m_seed_55_0.4.130",
        "remaining_runs_canon_3m_seed_56_0.4.131",
        "scaling_law_4_more_seeds_canon_3m_seed_57_0.4.152",
        "scaling_law_4_more_seeds_canon_3m_seed_58_0.4.153",
        # 10m
        "scaling_law_exps_3_canon_10m_0.4.78",
        "remaining_runs_canon_10m_seed_55_0.4.126",
        "remaining_runs_canon_10m_seed_56_0.4.127",
        "scaling_law_4_more_seeds_canon_10m_seed_57_0.4.148",
        "scaling_law_4_more_seeds_canon_10m_seed_58_0.4.149",
        # 30m
        "scaling_law_exps_3_canon_30m_0.4.79",
        "remaining_runs_canon_30m_seed_55_0.4.128",
        "remaining_runs_canon_30m_seed_56_0.4.129",
        "scaling_law_4_more_seeds_canon_30m_seed_57_0.4.150",
        "scaling_law_4_more_seeds_canon_30m_seed_58_0.4.151",
        # 100m
        "scaling_law_exps_3_canon_100m_0.4.77",
        "canon_seed_variance_100m_canon_3e-4_seed_55_0.4.118",
        "canon_seed_variance_100m_canon_3e-4_seed_56_0.4.119",
        "scaling_law_4_more_seeds_canon_100m_seed_57_0.4.146",
        "scaling_law_4_more_seeds_canon_100m_seed_58_0.4.147",
    ],
}


def print_scaling_table(avg_df):
    print(f"\n{'='*70}")
    print("Scaling law summary (5 seeds averaged per arch × size)")
    print(f"{'='*70}")
    print(f"{'Arch':>8}  {'Params':>12}  {'loss/out':>10}")
    for _, row in avg_df.sort_values(["label", "n_params"]).iterrows():
        print(f"  {row['label']:>6}  {int(row['n_params']):>12,}  {row['loss/out']:>10.4f}")

    print(f"\n  Improvement (Llama - Canon) per size:")
    llama_df = avg_df[avg_df["label"] == "llama"].sort_values("n_params").reset_index(drop=True)
    canon_df = avg_df[avg_df["label"] == "canon"].sort_values("n_params").reset_index(drop=True)
    for i in range(min(len(llama_df), len(canon_df))):
        delta = llama_df.loc[i, "loss/out"] - canon_df.loc[i, "loss/out"]
        params = int(llama_df.loc[i, "n_params"])
        print(f"    {params:>12,} params  delta={delta:+.4f}")


def make_loss_scaling_plot(avg_df):
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_scaling_law(avg_df, "loss/out", ax=ax, log_x=True, log_y=True,
                     title="Scaling law: final loss vs model size\nCanon consistently below Llama")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "loss_scaling.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"Saved: {path}")


def make_primary_metrics_plot(avg_df):
    fig = plot_all_tasks_primary_metrics(avg_df, log_x=True, log_y=False)
    path = os.path.join(OUT_DIR, "all_tasks_primary_metrics.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"Saved: {path}")


def make_avg_metric_plot(avg_df):
    primary_keys = [wandb_eval_key(t, m) for t, m in TASK_PRIMARY_METRIC.items()]
    available = [k for k in primary_keys if k in avg_df.columns]
    if not available:
        print("  (no eval metrics available for avg-metric plot)")
        return
    avg_df = avg_df.copy()
    avg_df["avg_primary"] = avg_df[available].mean(axis=1)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [p["color"] for p in plt.rcParams["axes.prop_cycle"]]
    for i, (label, group) in enumerate(avg_df.groupby("label")):
        g = group.dropna(subset=["avg_primary"]).sort_values("n_params")
        ax.scatter(g["n_params"], 1 - g["avg_primary"], zorder=3,
                   color=colors[i], label=label, s=60)
        ax.plot(g["n_params"], 1 - g["avg_primary"], "--", color=colors[i], alpha=0.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Model parameters"); ax.set_ylabel("Avg error (1 - primary metric)")
    ax.set_title("Average error across all tasks vs model size")
    ax.legend(); ax.grid(True, which="both", alpha=0.3, linestyle="--")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "avg_metric_scaling.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"Saved: {path}")


# ── Exp 7: Per-task Canon vs Llama breakdown at 30M ───────────────────────────

TASK_PRIMARY = {
    "Depo (el)":       ("evals/synthetic/depo_edges_list/hop_4/accuracy",         "hop_4"),
    "Depo (al)":       ("evals/synthetic/depo_adj_list/hop_4/accuracy",           "hop_4"),
    "Concomp (el)":    ("evals/synthetic/concomp_factor_edges_list/prefix_accuracy", "prefix_acc"),
    "Concomp (al)":    ("evals/synthetic/concomp_factor_adj_list/prefix_accuracy",   "prefix_acc"),
    "ShortPath (el)":  ("evals/synthetic/shortest_path_edges_list/set_accuracy",   "set_acc"),
    "ShortPath (al)":  ("evals/synthetic/shortest_path_adj_list/set_accuracy",     "set_acc"),
    "BFS (el)":        ("evals/synthetic/bfs_edges_list/set_recall",               "set_recall"),
    "BFS (al)":        ("evals/synthetic/bfs_adj_list/set_recall",                 "set_recall"),
}

# Approximate ~25M parameter scale
TARGET_PARAMS = 25_000_000
PARAM_TOLERANCE = 5_000_000


def print_per_task_breakdown(results_df):
    """Print per-task Canon vs Llama metrics at 30M, with mean±std across seeds."""
    print(f"\n{'='*80}")
    print("Exp 7: Per-task Canon vs Llama at 30M (5 seeds each, mean ± std)")
    print(f"{'='*80}")
    subset = results_df[abs(results_df["n_params"] - TARGET_PARAMS) < PARAM_TOLERANCE]
    if subset.empty:
        print("  No 30M runs found."); return

    for label, (metric, short) in TASK_PRIMARY.items():
        if metric not in subset.columns: continue
        c_vals = subset[subset["label"] == "canon"][metric].dropna()
        l_vals = subset[subset["label"] == "llama"][metric].dropna()
        if c_vals.empty or l_vals.empty: continue
        c_m, c_s = c_vals.mean(), c_vals.std(ddof=1)
        l_m, l_s = l_vals.mean(), l_vals.std(ddof=1)
        delta = c_m - l_m
        print(f"  {label:>16}  Canon={c_m:.4f}±{c_s:.4f}  Llama={l_m:.4f}±{l_s:.4f}  "
              f"Δ={delta:+.4f}")


def make_per_task_breakdown_plot(results_df):
    """Grouped bar chart: Canon vs Llama per task at 30M, with error bars."""
    subset = results_df[abs(results_df["n_params"] - TARGET_PARAMS) < PARAM_TOLERANCE]
    if subset.empty:
        print("  No 30M runs for per-task plot."); return

    task_labels, canon_means, canon_stds, llama_means, llama_stds, deltas = [], [], [], [], [], []
    for label, (metric, _) in TASK_PRIMARY.items():
        if metric not in subset.columns: continue
        c_vals = subset[subset["label"] == "canon"][metric].dropna()
        l_vals = subset[subset["label"] == "llama"][metric].dropna()
        if c_vals.empty or l_vals.empty: continue
        task_labels.append(label)
        canon_means.append(c_vals.mean()); canon_stds.append(c_vals.std(ddof=1) if len(c_vals)>1 else 0)
        llama_means.append(l_vals.mean()); llama_stds.append(l_vals.std(ddof=1) if len(l_vals)>1 else 0)
        deltas.append(c_vals.mean() - l_vals.mean())

    x = np.arange(len(task_labels)); w = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Left: Canon vs Llama absolute
    ax = axes[0]
    ax.bar(x - w/2, llama_means, w, yerr=llama_stds, label="Llama", color="#2166ac",
           alpha=0.8, capsize=4)
    ax.bar(x + w/2, canon_means, w, yerr=canon_stds, label="Canon", color="#d6604d",
           alpha=0.8, capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(task_labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Primary metric"); ax.set_title("Canon vs Llama per task at ~30M (mean ± std, 5 seeds)")
    ax.legend(); ax.grid(axis="y", alpha=0.3, ls="--")

    # Right: Canon improvement (delta), sorted
    ax = axes[1]
    sorted_idx = np.argsort(deltas)[::-1]
    s_labels = [task_labels[i] for i in sorted_idx]
    s_deltas = [deltas[i] for i in sorted_idx]
    bar_colors = ["#d6604d" if d > 0 else "#2166ac" for d in s_deltas]
    ax.bar(range(len(s_labels)), s_deltas, color=bar_colors, alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(s_labels)))
    ax.set_xticklabels(s_labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Canon − Llama"); ax.set_title("Exp 7: Canon improvement per task (sorted)")
    ax.grid(axis="y", alpha=0.3, ls="--")
    for i, d in enumerate(s_deltas):
        ax.text(i, d + (0.003 if d >= 0 else -0.006), f"{d:+.3f}",
                ha="center", fontsize=7)

    fig.suptitle("Exp 7: Per-task Canon vs Llama breakdown at 30M\n"
                 "Which reasoning tasks benefit most from Canon?",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "per_task_breakdown_30m.png")
    fig.savefig(path, dpi=150); plt.close(fig); print(f"Saved: {path}")


if __name__ == "__main__":
    print("Fetching run summaries (5 seeds × 4 sizes × 2 archs = 40 runs)...")
    results_df = fetch_runs_results(
        run_ids=RUN_IDS,
        api_key=WANDB_API_KEY, entity=ENTITY, project=PROJECT,
    )
    print(f"\nFetched {len(results_df)} runs")

    # Average over seeds → one row per (label, n_params)
    id_cols = {"run_id", "run_name", "label", "n_params"}
    metric_cols = [c for c in results_df.columns if c not in id_cols]
    group_cols = [c for c in ["label", "n_params"] if c in results_df.columns]
    avg_df = results_df.groupby(group_cols, as_index=False)[metric_cols].mean(numeric_only=True)
    print(f"Averaged to {len(avg_df)} rows (one per arch × size)")

    print_scaling_table(avg_df)
    make_loss_scaling_plot(avg_df)
    make_primary_metrics_plot(avg_df)
    make_avg_metric_plot(avg_df)

    # Exp 7: per-task breakdown
    print("\n=== Exp 7: Per-task Canon vs Llama at 30M ===")
    print_per_task_breakdown(results_df)
    make_per_task_breakdown_plot(results_df)

    print("\nDone.")
