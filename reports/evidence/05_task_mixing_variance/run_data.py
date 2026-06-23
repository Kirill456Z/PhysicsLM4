"""
F5: Mixing more diverse synthetic tasks into training substantially reduces seed variance.

Fetches summary and training curves for 5 task-diversity groups (1–8 tasks),
3 seeds each. Prints a variance summary table and saves bar + curve plots.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from utils import (
    fetch_runs_results, fetch_training_history,
    plot_variance_curves, plot_final_metric_bars,
    TASK_NAMES, get_wandb_metric_keys,
)

WANDB_API_KEY = "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"
ENTITY = "kirill456z"
PROJECT = "physics4llm"
OUT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(OUT_DIR, exist_ok=True)

RUN_IDS = {
    "all_tasks": [
        "depo_variance_impact_investigation_2_all_tasks_seed_55_0.4.120",
        "depo_variance_impact_investigation_2_all_tasks_seed_56_0.4.121",
        "scaling_law_4_more_seeds_llama_30m_seed_57_0.4.158",
    ],
    "no_depo": [
        "depo_variance_impact_investigation_2_no_depo_seed_54_0.4.122",
        "depo_variance_impact_investigation_2_no_depo_seed_55_0.4.123",
        "depo_variance_impact_investigation_2_no_depo_seed_56_0.4.124",
    ],
    "edges_only": [
        "seed_variance_exps_remaining_only_edges_list_55_0.4.192",
        "seed_variance_exps_remaining_only_edges_list_56_0.4.193",
        "seed_variance_exps_remaining_only_edges_list_57_0.4.194",
    ],
    "depo_both": [
        "seed_variance_exps_remaining_only_depo_55_0.4.186",
        "seed_variance_exps_remaining_only_depo_56_0.4.187",
        "seed_variance_exps_remaining_only_depo_57_0.4.188",
    ],
    "depo_edges": [
        "seed_variance_exps_remaining_only_depo_edges_55_0.4.189",
        "seed_variance_exps_remaining_only_depo_edges_56_0.4.190",
        "seed_variance_exps_remaining_only_depo_edges_57_0.4.191",
    ],
}

NUM_TASKS = {"all_tasks": 8, "no_depo": 6, "edges_only": 4, "depo_both": 2, "depo_edges": 1}
NICE_LABELS = {
    "all_tasks":  "All tasks (8)",
    "no_depo":    "No depo (6)",
    "edges_only": "Edges only (4)",
    "depo_both":  "Depo ×2 (2)",
    "depo_edges": "Depo edges (1)",
}
GROUPS_ORDERED = ["depo_edges", "depo_both", "edges_only", "no_depo", "all_tasks"]


def print_variance_table(results_df):
    print(f"\n{'='*70}")
    print("Variance table: loss/out mean ± std across seeds")
    print(f"{'='*70}")
    print(f"{'Group':>20}  {'#tasks':>6}  {'mean':>8}  {'std':>8}  {'seeds':>6}")
    for group in GROUPS_ORDERED:
        sub = results_df[results_df["label"] == group]
        if sub.empty:
            continue
        vals = sub["loss/out"].dropna().tolist()
        print(f"  {NICE_LABELS[group]:>18}  {NUM_TASKS[group]:>6}  "
              f"{np.mean(vals):>8.4f}  {np.std(vals, ddof=1):>8.4f}  {len(vals):>6}")


def make_bar_chart(results_df):
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [p["color"] for p in plt.rcParams["axes.prop_cycle"]]
    bar_groups = [g for g in GROUPS_ORDERED if g in results_df["label"].values]
    means, stds, seeds_lists = [], [], []
    for g in bar_groups:
        sub = results_df[results_df["label"] == g]["loss/out"].dropna()
        means.append(sub.mean())
        stds.append(sub.std(ddof=1) if len(sub) > 1 else 0.0)
        seeds_lists.append(sub.tolist())

    xs = np.arange(len(bar_groups))
    for i, (g, m, s, seeds) in enumerate(zip(bar_groups, means, stds, seeds_lists)):
        color = colors[i % len(colors)]
        ax.bar(i, m, yerr=s, color=color, alpha=0.75, capsize=5, label=NICE_LABELS[g])
        jitter = np.random.default_rng(i).uniform(-0.12, 0.12, len(seeds))
        ax.scatter(i + jitter, seeds, color=color, zorder=3, s=40, edgecolors="white", linewidths=0.5)

    ax.set_xticks(xs)
    ax.set_xticklabels([NICE_LABELS[g] for g in bar_groups], rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Final loss/out"); ax.set_xlabel("Task diversity")
    ax.set_title("More tasks → lower mean loss and lower seed variance\n(bars = mean ± std, dots = individual seeds)")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "variance_bar_chart.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"Saved: {path}")


def make_std_vs_tasks_plot(results_df):
    groups_with_data = [g for g in GROUPS_ORDERED if g in results_df["label"].values]
    xs = [NUM_TASKS[g] for g in groups_with_data]
    means = [results_df[results_df["label"] == g]["loss/out"].mean() for g in groups_with_data]
    stds  = [results_df[results_df["label"] == g]["loss/out"].std(ddof=1) for g in groups_with_data]
    labels = [NICE_LABELS[g] for g in groups_with_data]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    ax = axes[0]
    ax.errorbar(xs, means, yerr=stds, fmt="o-", capsize=5, capthick=1.5, linewidth=2,
                markersize=8, color="steelblue", ecolor="#888")
    for x, y, lbl in zip(xs, means, labels):
        ax.annotate(lbl, xy=(x, y), xytext=(4, 6), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Number of training tasks"); ax.set_ylabel("loss/out  (mean ± std)")
    ax.set_title("More tasks → lower and more stable loss")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(axis="y", alpha=0.35)

    ax = axes[1]
    colors = [p["color"] for p in plt.rcParams["axes.prop_cycle"]]
    ax.bar(labels, stds, color=[colors[i % len(colors)] for i in range(len(labels))], alpha=0.8)
    ax.set_ylabel("Std of loss/out across seeds"); ax.set_title("Seed variance vs. task diversity")
    ax.tick_params(axis="x", rotation=15); ax.grid(axis="y", alpha=0.35)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "std_vs_tasks.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"Saved: {path}")


def make_curve_plot(histories):
    from matplotlib import cm as _cm, colors as _mcolors
    curve_groups = ["depo_edges", "depo_both", "edges_only", "all_tasks"]
    n = len(curve_groups)
    palette = [_mcolors.to_hex(_cm.RdYlGn(i / (n - 1))) for i in range(n)]
    histories_nice = {NICE_LABELS[g]: histories[g] for g in curve_groups if g in histories}

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_variance_curves(
        histories_nice, "loss/out", ax=ax,
        title="Training loss: mean ± std band per task diversity group",
        smoothing=20, alpha_individual=0.2, show_individual=True, palette=palette,
    )
    ax.set_yscale("log")
    ax.set_ylim(top=1.5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:g}"))
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "training_curves_by_task_group.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    print("Fetching run summaries...")
    results_df = fetch_runs_results(
        run_ids=RUN_IDS,
        api_key=WANDB_API_KEY, entity=ENTITY, project=PROJECT,
    )
    results_df["num_tasks"] = results_df["label"].map(NUM_TASKS)

    print_variance_table(results_df)

    print("\nFetching training curves...")
    histories = fetch_training_history(
        groups=RUN_IDS,
        metric_keys=["loss/out"],
        api_key=WANDB_API_KEY, entity=ENTITY, project=PROJECT, samples=300,
    )

    make_bar_chart(results_df)
    make_std_vs_tasks_plot(results_df)
    make_curve_plot(histories)

    print("\nDone.")
