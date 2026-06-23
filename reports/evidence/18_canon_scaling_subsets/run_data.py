"""
Seed count vs. scaling-law smoothness.

The Canon-vs-Llama loss scaling law is only clean once enough seeds are
averaged per point. With a single seed the curves are noisy and the Canon
advantage is easy to miss; by 5 seeds the two curves separate cleanly.

This script fetches 5 seeds x 4 sizes x 2 architectures, then re-draws the
loss scaling law using only the first 1, 3, and 5 seeds per (arch, size).
It produces six plots:

  loss_scaling_1seed.png / _3seed.png / _5seed.png
      mean loss vs model size (no variance shown)
  loss_scaling_1seed_var.png / _3seed_var.png / _5seed_var.png
      same, with mean +/- std bands across seeds per point

Plus two combined 1x3 panels (mean-only and with-variance) for the report.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import fetch_runs_results

WANDB_API_KEY = "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"
ENTITY = "kirill456z"
PROJECT = "physics4llm"
OUT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(OUT_DIR, exist_ok=True)

METRIC = "loss/out"
SEED_COUNTS = [1, 3, 5]

# 5 seeds per size, listed seed-by-seed and contiguous per size block.
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

SEEDS_PER_SIZE = 5
COLORS = {"llama": "#2166ac", "canon": "#d6604d"}


def first_n_seed_ids(n: int) -> set[str]:
    """Run ids for the first n seeds of every (arch, size) block."""
    keep = set()
    for ids in RUN_IDS.values():
        for start in range(0, len(ids), SEEDS_PER_SIZE):
            keep.update(ids[start:start + n])
    return keep


def seed_stats(results_df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Mean and std of METRIC across the first n seeds, per (label, n_params)."""
    keep = first_n_seed_ids(n)
    sub = results_df[results_df["run_id"].isin(keep)]
    stats = (
        sub.groupby(["label", "n_params"], as_index=False)[METRIC]
        .agg(mean="mean", std="std", count="count")
    )
    stats["std"] = stats["std"].fillna(0.0)  # std is NaN for a single seed
    return stats


def _draw(ax, stats: pd.DataFrame, with_bands: bool) -> None:
    for label in ["llama", "canon"]:
        g = stats[stats["label"] == label].sort_values("n_params")
        if g.empty:
            continue
        x = g["n_params"].to_numpy(float)
        y = g["mean"].to_numpy(float)
        s = g["std"].to_numpy(float)
        c = COLORS[label]
        if with_bands:
            ax.errorbar(x, y, yerr=s, fmt="o", color=c, capsize=4,
                        markersize=6, zorder=3, label=label)
            ax.fill_between(x, y - s, y + s, color=c, alpha=0.18)
        else:
            ax.scatter(x, y, color=c, s=45, zorder=3, label=label)
        ax.plot(x, y, "--", color=c, alpha=0.45)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Model parameters")
    ax.set_ylabel(METRIC)
    ax.grid(True, which="both", alpha=0.3, linestyle="--")
    ax.legend(fontsize=9)


def make_single_plot(stats: pd.DataFrame, n: int, with_bands: bool) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    _draw(ax, stats, with_bands)
    suffix = "mean +/- std across seeds" if with_bands else "seed mean"
    ax.set_title(f"Loss scaling law - {n} seed{'s' if n > 1 else ''} per point\n({suffix})")
    fig.tight_layout()
    tag = "var" if with_bands else "mean"
    path = os.path.join(OUT_DIR, f"loss_scaling_{n}seed_{tag}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def make_combined_panel(all_stats: dict[int, pd.DataFrame], with_bands: bool) -> None:
    fig, axes = plt.subplots(1, len(SEED_COUNTS), figsize=(18, 5), sharey=True)
    for ax, n in zip(axes, SEED_COUNTS):
        _draw(ax, all_stats[n], with_bands)
        ax.set_title(f"{n} seed{'s' if n > 1 else ''} per point")
        if ax is not axes[0]:
            ax.set_ylabel("")
    kind = "with mean +/- std bands" if with_bands else "seed means only"
    fig.suptitle(
        f"Loss scaling law sharpens with more seeds ({kind})\n"
        "Canon's advantage over Llama is noisy at 1 seed, clean at 5",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    tag = "var" if with_bands else "mean"
    path = os.path.join(OUT_DIR, f"loss_scaling_panel_{tag}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    print("Fetching run summaries (5 seeds x 4 sizes x 2 archs = 40 runs)...")
    results_df = fetch_runs_results(
        run_ids=RUN_IDS,
        api_key=WANDB_API_KEY, entity=ENTITY, project=PROJECT,
    )
    print(f"\nFetched {len(results_df)} runs")

    all_stats = {n: seed_stats(results_df, n) for n in SEED_COUNTS}

    print(f"\n{'='*60}\nLoss (mean +/- std) per (arch, size) by seed count\n{'='*60}")
    for n in SEED_COUNTS:
        print(f"\n-- {n} seed(s) --")
        for _, r in all_stats[n].sort_values(["label", "n_params"]).iterrows():
            print(f"  {r['label']:>6}  {int(r['n_params']):>12,}  "
                  f"{r['mean']:.4f} +/- {r['std']:.4f}  (n={int(r['count'])})")

    # Six required plots: 1/3/5 seeds, mean-only and with variance bands.
    for n in SEED_COUNTS:
        make_single_plot(all_stats[n], n, with_bands=False)
    for n in SEED_COUNTS:
        make_single_plot(all_stats[n], n, with_bands=True)

    # Combined panels for the report.
    make_combined_panel(all_stats, with_bands=False)
    make_combined_panel(all_stats, with_bands=True)

    print("\nDone.")
