"""
Depo grokking × task-correlation analysis.

Quantization rule: depo_avg = mean(hop_4/accuracy over edges_list and adj_list)
  depo_grokked = 1  if  depo_avg > 0.2
  depo_grokked = 0  otherwise

For each other task (BFS, SP, ConComp) we average the metric over both encoding
variants and compute:
  - Point-biserial correlation r between depo_grokked and the task metric
  - Mann-Whitney U p-value (one-sided: grokked > not-grokked)

Data source: all scaling-law runs (Canon + Llama, 3M–300M, both seed batches,
including single-layer-AC variants).

Plots saved to data_synthetic_pretrain/plots/:
  depo_corr_strips.png   — per-task strip+box plots split by depo_grokked
  depo_corr_bars.png     — point-biserial r bar chart per task
  depo_corr_scatter.png  — depo_avg vs each task metric, threshold line, colored by scale
"""
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import wandb
from scipy import stats

# ── Config ────────────────────────────────────────────────────────────────────

WANDB_API_KEY = (
    "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"
)
ENTITY = "kirill456z"
PROJECT = "physics4llm"
DEPO_THRESHOLD = 0.2

OUT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(OUT_DIR, exist_ok=True)

C_GROKKED = "#d6604d"    # red  — depo grokked
C_NOT     = "#4393c3"    # blue — depo not grokked

# ── Run inventory ─────────────────────────────────────────────────────────────

ALL_RUNS = [
    # ── scaling_law_4_more_seeds (5 arch × 4 scales × 2 seeds = 16) ──────────
    "scaling_law_4_more_seeds_canon_3m_seed_57_0.4.152",
    "scaling_law_4_more_seeds_canon_3m_seed_58_0.4.153",
    "scaling_law_4_more_seeds_canon_10m_seed_57_0.4.148",
    "scaling_law_4_more_seeds_canon_10m_seed_58_0.4.149",
    "scaling_law_4_more_seeds_canon_30m_seed_57_0.4.150",
    "scaling_law_4_more_seeds_canon_30m_seed_58_0.4.151",
    "scaling_law_4_more_seeds_canon_100m_seed_57_0.4.146",
    "scaling_law_4_more_seeds_canon_100m_seed_58_0.4.147",
    "scaling_law_4_more_seeds_llama_3m_seed_57_0.4.160",
    "scaling_law_4_more_seeds_llama_3m_seed_58_0.4.161",
    "scaling_law_4_more_seeds_llama_10m_seed_57_0.4.156",
    "scaling_law_4_more_seeds_llama_10m_seed_58_0.4.157",
    "scaling_law_4_more_seeds_llama_30m_seed_57_0.4.158",
    "scaling_law_4_more_seeds_llama_30m_seed_58_0.4.159",
    "scaling_law_4_more_seeds_llama_100m_seed_57_0.4.154",
    "scaling_law_4_more_seeds_llama_100m_seed_58_0.4.155",
    # ── scaling_law_exps_3 (original single-seed batch: Canon + Llama × 4 scales) ─
    "scaling_law_exps_3_canon_3m_0.4.80",
    "scaling_law_exps_3_canon_10m_0.4.78",
    "scaling_law_exps_3_canon_30m_0.4.79",
    "scaling_law_exps_3_canon_100m_0.4.77",
    "scaling_law_exps_3_llama_3m_0.4.84",
    "scaling_law_exps_3_llama_10m_0.4.82",
    "scaling_law_exps_3_llama_30m_0.4.83",
    "scaling_law_exps_3_llama_100m_0.4.81",
    # ── Canon first-layer-AC ablation (4 scales × 2 seed batches = 6) ─────────
    "scaling_law_single_layer_only_2_canon_first_layer_ac_3m_0.4.89",
    "scaling_law_single_layer_only_2_canon_first_layer_ac_10m_0.4.87",
    "scaling_law_single_layer_only_2_canon_first_layer_ac_30_0.4.88",
    "scaling_law_single_layer_only_2_canon_first_layer_ac_100m_0.4.86",
    "scaling_law_single_layer_only_canon_first_layer_ac_30m_0.4.75",
    "scaling_law_single_layer_only_canon_first_layer_ac_300m_0.4.74",
]

# ── Task metrics to correlate with depo ──────────────────────────────────────
# Each entry: (label, [(wandb_key, weight), ...])
# We average the listed keys (equal weight) to get a single scalar per run.
TASKS = [
    ("BFS", [
        ("evals/synthetic/bfs_edges_list/set_recall",              1),
        ("evals/synthetic/bfs_adj_list/set_recall",                1),
    ]),
    ("SP", [
        ("evals/synthetic/shortest_path_edges_list/set_accuracy",  1),
        ("evals/synthetic/shortest_path_adj_list/set_accuracy",    1),
    ]),
    ("ConComp", [
        ("evals/synthetic/concomp_factor_edges_list/prefix_accuracy", 1),
        ("evals/synthetic/concomp_factor_adj_list/prefix_accuracy",   1),
    ]),
]

DEPO_KEYS = [
    "evals/synthetic/depo_edges_list/hop_4/accuracy",
    "evals/synthetic/depo_adj_list/hop_4/accuracy",
]

SCALE_LABELS = {
    "3m": "3M", "10m": "10M", "30m": "30M", "100m": "100M", "300m": "300M",
}


# ── Data fetching ─────────────────────────────────────────────────────────────

def _avg_keys(summary: dict, keys: list) -> float:
    vals = [summary.get(k) for k in keys]
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(vals)) if vals else float("nan")


def _infer_scale(run_name: str) -> str:
    for token in ["300m", "100m", "30m", "10m", "3m"]:
        if token in run_name:
            return SCALE_LABELS[token]
    return "?"


def fetch_data(api: wandb.Api) -> pd.DataFrame:
    all_keys = DEPO_KEYS + [k for _, task_keys in TASKS for k, _ in task_keys]
    rows = []
    for run_id in ALL_RUNS:
        try:
            run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
            summary = dict(run.summary)
            depo_avg = _avg_keys(summary, DEPO_KEYS)
            row = {
                "run_id": run_id,
                "scale": _infer_scale(run_id),
                "depo_avg": depo_avg,
                "depo_grokked": int(depo_avg > DEPO_THRESHOLD) if not math.isnan(depo_avg) else float("nan"),
            }
            for label, task_keys in TASKS:
                row[label] = _avg_keys(summary, [k for k, _ in task_keys])
            rows.append(row)
            g_str = "GROKKED" if row["depo_grokked"] == 1 else "not"
            print(f"  {run_id[-30:]:30s}  depo={depo_avg:.3f} [{g_str:8s}]  "
                  + "  ".join(f"{lbl}={row[lbl]:.3f}" for lbl, _ in TASKS
                               if not math.isnan(row[lbl])))
        except Exception as exc:
            print(f"  [warn] {run_id}: {exc}")
    return pd.DataFrame(rows)


# ── Statistics ────────────────────────────────────────────────────────────────

def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Point-biserial r and Mann-Whitney p (grokked > not) for each task."""
    valid = df.dropna(subset=["depo_grokked"])
    rows = []
    for label, _ in TASKS:
        sub = valid.dropna(subset=[label])
        x_bin = sub["depo_grokked"].values.astype(float)
        y_cont = sub[label].values.astype(float)

        r, p_pearson = stats.pointbiserialr(x_bin, y_cont)

        grokked_vals = sub.loc[sub["depo_grokked"] == 1, label].values
        not_vals     = sub.loc[sub["depo_grokked"] == 0, label].values
        if len(grokked_vals) > 0 and len(not_vals) > 0:
            _, p_mw = stats.mannwhitneyu(grokked_vals, not_vals, alternative="greater")
            mean_grokked = float(np.mean(grokked_vals))
            mean_not     = float(np.mean(not_vals))
        else:
            p_mw = float("nan")
            mean_grokked = mean_not = float("nan")

        rows.append({
            "task": label,
            "r": r,
            "p_pearson": p_pearson,
            "p_mw": p_mw,
            "mean_grokked": mean_grokked,
            "mean_not": mean_not,
            "n_grokked": int((x_bin == 1).sum()),
            "n_not":     int((x_bin == 0).sum()),
        })
    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame, corr: pd.DataFrame) -> None:
    valid = df.dropna(subset=["depo_grokked"])
    n_g = int((valid["depo_grokked"] == 1).sum())
    n_n = int((valid["depo_grokked"] == 0).sum())
    print(f"\n{'='*70}")
    print(f"Depo grokking threshold: > {DEPO_THRESHOLD}")
    print(f"  Grokked (1): n={n_g}   Not grokked (0): n={n_n}   Total: {n_g+n_n}")
    print(f"\n{'Task':10s}  {'r':>7}  {'p_pearson':>10}  {'p_mw':>10}  "
          f"{'mean(g=1)':>10}  {'mean(g=0)':>10}  {'Δ':>8}")
    for _, row in corr.iterrows():
        delta = row["mean_grokked"] - row["mean_not"]
        sig = "*" if row["p_mw"] < 0.05 else ""
        print(f"{row['task']:10s}  {row['r']:>7.3f}  {row['p_pearson']:>10.4f}  "
              f"{row['p_mw']:>10.4f}  {row['mean_grokked']:>10.3f}  "
              f"{row['mean_not']:>10.3f}  {delta:>+8.3f} {sig}")
    print(f"{'='*70}")


# ── Plots ─────────────────────────────────────────────────────────────────────

SCALE_MARKERS = {"3M": "o", "10M": "s", "30M": "^", "100M": "D", "300M": "P", "?": "x"}
SCALE_COLORS  = {"3M": "#aaaaaa", "10M": "#74add1", "30M": "#f46d43",
                 "100M": "#313695", "300M": "#006837", "?": "black"}


def plot_strips(df: pd.DataFrame, corr: pd.DataFrame) -> None:
    """Strip + box plots: per-task metric split by depo_grokked."""
    valid = df.dropna(subset=["depo_grokked"])
    n_tasks = len(TASKS)
    fig, axes = plt.subplots(1, n_tasks, figsize=(5 * n_tasks, 5))
    if n_tasks == 1:
        axes = [axes]

    rng = np.random.default_rng(0)
    for ax, (label, _) in zip(axes, TASKS):
        sub = valid.dropna(subset=[label])
        for xi, (group_val, color, group_name) in enumerate(
            [(0, C_NOT, "Not grokked\n(depo ≤ 0.2)"),
             (1, C_GROKKED, "Grokked\n(depo > 0.2)")]
        ):
            vals = sub.loc[sub["depo_grokked"] == group_val, label].values
            if len(vals) == 0:
                continue
            # box
            bp = ax.boxplot(vals, positions=[xi], widths=0.35, patch_artist=True,
                            medianprops=dict(color="white", lw=2),
                            boxprops=dict(facecolor=color, alpha=0.35),
                            whiskerprops=dict(color=color),
                            capprops=dict(color=color),
                            flierprops=dict(marker="", linestyle="none"))
            # individual points colored by scale
            jit = rng.uniform(-0.12, 0.12, len(vals))
            scales = sub.loc[sub["depo_grokked"] == group_val, "scale"].values
            for v, j, sc in zip(vals, jit, scales):
                ax.scatter(xi + j, v,
                           color=SCALE_COLORS.get(sc, "black"),
                           marker=SCALE_MARKERS.get(sc, "o"),
                           s=60, zorder=4, edgecolors="white", linewidths=0.5)

        # annotate with r and p_mw
        row = corr[corr["task"] == label].iloc[0]
        sig = "*" if row["p_mw"] < 0.05 else "n.s."
        ax.text(0.5, 1.03,
                f"r = {row['r']:.3f}  |  p_MW = {row['p_mw']:.3f} {sig}",
                ha="center", va="bottom", transform=ax.transAxes, fontsize=9)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Not grokked\n(depo ≤ 0.2)", "Grokked\n(depo > 0.2)"],
                           fontsize=9)
        ax.set_ylabel(f"{label} (avg both encodings)")
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_ylim(-0.05, 1.10)
        ax.grid(axis="y", alpha=0.3, ls="--")

    # legend for scales
    handles = [mpatches.Patch(color=SCALE_COLORS[s], label=s)
               for s in ["3M", "10M", "30M", "100M", "300M"]]
    axes[-1].legend(handles=handles, title="Scale", fontsize=8,
                    loc="lower right", framealpha=0.8)

    fig.suptitle(
        "Does Depo grokking predict other task performance?\n"
        f"(Depo grokked ≡ depo_avg > {DEPO_THRESHOLD}, n={len(valid)} runs)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "depo_corr_strips.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_correlation_bars(corr: pd.DataFrame) -> None:
    """Horizontal bar chart of point-biserial r per task."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    tasks  = corr["task"].values
    r_vals = corr["r"].values
    p_vals = corr["p_mw"].values
    colors = [C_GROKKED if r > 0 else C_NOT for r in r_vals]

    bars = ax.barh(tasks, r_vals, color=colors, alpha=0.8, height=0.5)
    for bar, r, p in zip(bars, r_vals, p_vals):
        sig = "*" if p < 0.05 else "n.s."
        xpos = r + 0.01 if r >= 0 else r - 0.01
        ha = "left" if r >= 0 else "right"
        ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                f"r={r:.3f} ({sig})", va="center", ha=ha, fontsize=9)

    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Point-biserial correlation r\n(depo_grokked ↔ task metric)")
    ax.set_title(
        f"Correlation between Depo grokking (threshold={DEPO_THRESHOLD})\n"
        "and other task performance",
        fontsize=10,
    )
    ax.set_xlim(-0.15, 1.05)
    ax.grid(axis="x", alpha=0.3, ls="--")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "depo_corr_bars.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_scatter(df: pd.DataFrame) -> None:
    """Scatter: depo_avg vs each task metric, threshold marked, points colored by scale."""
    valid = df.dropna(subset=["depo_avg"])
    n_tasks = len(TASKS)
    fig, axes = plt.subplots(1, n_tasks, figsize=(5 * n_tasks, 4.5))
    if n_tasks == 1:
        axes = [axes]

    for ax, (label, _) in zip(axes, TASKS):
        sub = valid.dropna(subset=[label])
        for _, row in sub.iterrows():
            sc = row["scale"]
            ax.scatter(row["depo_avg"], row[label],
                       color=SCALE_COLORS.get(sc, "black"),
                       marker=SCALE_MARKERS.get(sc, "o"),
                       s=60, zorder=3, edgecolors="white", linewidths=0.5,
                       label=sc)

        # Threshold line
        ax.axvline(DEPO_THRESHOLD, color="black", lw=1.2, ls="--", alpha=0.7,
                   label=f"threshold={DEPO_THRESHOLD}")
        ax.fill_betweenx([0, 1.05], 0, DEPO_THRESHOLD,
                         color=C_NOT, alpha=0.05)
        ax.fill_betweenx([0, 1.05], DEPO_THRESHOLD, 1,
                         color=C_GROKKED, alpha=0.05)
        ax.text(DEPO_THRESHOLD / 2, 1.03, "not grokked",
                ha="center", va="bottom", fontsize=8, color=C_NOT)
        ax.text((1 + DEPO_THRESHOLD) / 2, 1.03, "grokked",
                ha="center", va="bottom", fontsize=8, color=C_GROKKED)

        ax.set_xlabel("Depo hop_4 accuracy (avg encodings)")
        ax.set_ylabel(f"{label} (avg encodings)")
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.10)
        ax.grid(alpha=0.3, ls="--")

    # shared scale legend (deduplicated)
    handles = [mpatches.Patch(color=SCALE_COLORS[s], label=s)
               for s in ["3M", "10M", "30M", "100M", "300M"]]
    handles.append(plt.Line2D([0], [0], ls="--", color="black", label=f"threshold={DEPO_THRESHOLD}"))
    axes[-1].legend(handles=handles, title="Scale", fontsize=8,
                    loc="lower right", framealpha=0.8)

    fig.suptitle(
        "Depo hop_4 accuracy vs other task performance\n"
        "(each point = one run; color = model scale)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "depo_corr_scatter.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    api = wandb.Api(api_key=WANDB_API_KEY)

    print("Fetching run summaries …")
    df = fetch_data(api)
    print(f"\nFetched {len(df)} runs total.")

    corr = compute_correlations(df)
    print_summary(df, corr)

    print("\nGenerating plots …")
    plot_strips(df, corr)
    plot_correlation_bars(corr)
    plot_scatter(df)

    print("\nDone.")
