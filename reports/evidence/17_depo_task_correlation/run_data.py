"""
Exp 17 — Depo grokking × task correlation.

Does grokking Depo (hop_4/accuracy > 0.2) predict performance on BFS, SP,
and ConComp?

Data: the 40 scaling-law runs from canon_scaling_laws.ipynb —
Canon + Llama × {3M, 10M, 30M, 100M} × 5 seeds each.

Metrics (edges_list and adj_list kept separate throughout, no averaging):
  Depo    — depo_{edges,adj}_list/hop_4/accuracy
             quantized: grokked=1 if > 0.2, else 0
  BFS     — bfs_{edges,adj}_list/prefix_accuracy
  SP      — shortest_path_{edges,adj}_list/prefix_accuracy
  ConComp — concomp_factor_{edges,adj}_list/prefix_accuracy

For each of the 6 non-depo metrics we compute point-biserial r against
both depo_edges_grokked and depo_adj_grokked (12 correlations total).

Plots saved to plots/:
  depo_corr_scatter.png  — 2×6 scatter grid: depo (continuous) vs each task,
                           threshold line, points coloured by model scale
  depo_corr_strips.png   — 2×6 strip+box grid: per-task distribution split by
                           depo_grokked (0 vs 1)
  depo_corr_bars.png     — point-biserial r bar chart for all 12 combinations
"""
import os, sys, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import wandb
from scipy import stats

WANDB_API_KEY = "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"
ENTITY        = "kirill456z"
PROJECT       = "physics4llm"
OUT_DIR       = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(OUT_DIR, exist_ok=True)

DEPO_THRESHOLD = 0.2
C_RED, C_BLUE  = "#d6604d", "#2166ac"
C_GROKKED, C_NOT = C_RED, C_BLUE

# ── Run inventory (from canon_scaling_laws.ipynb) ────────────────────────────

RUNS = {
    "Canon": [
        # 3M
        "scaling_law_exps_3_canon_3m_0.4.80",
        "remaining_runs_canon_3m_seed_55_0.4.130",
        "remaining_runs_canon_3m_seed_56_0.4.131",
        "scaling_law_4_more_seeds_canon_3m_seed_57_0.4.152",
        "scaling_law_4_more_seeds_canon_3m_seed_58_0.4.153",
        # 10M
        "scaling_law_exps_3_canon_10m_0.4.78",
        "remaining_runs_canon_10m_seed_55_0.4.126",
        "remaining_runs_canon_10m_seed_56_0.4.127",
        "scaling_law_4_more_seeds_canon_10m_seed_57_0.4.148",
        "scaling_law_4_more_seeds_canon_10m_seed_58_0.4.149",
        # 30M
        "scaling_law_exps_3_canon_30m_0.4.79",
        "remaining_runs_canon_30m_seed_55_0.4.128",
        "remaining_runs_canon_30m_seed_56_0.4.129",
        "scaling_law_4_more_seeds_canon_30m_seed_57_0.4.150",
        "scaling_law_4_more_seeds_canon_30m_seed_58_0.4.151",
        # 100M
        "scaling_law_exps_3_canon_100m_0.4.77",
        "canon_seed_variance_100m_canon_3e-4_seed_55_0.4.118",
        "canon_seed_variance_100m_canon_3e-4_seed_56_0.4.119",
        "scaling_law_4_more_seeds_canon_100m_seed_57_0.4.146",
        "scaling_law_4_more_seeds_canon_100m_seed_58_0.4.147",
    ],
    "Llama": [
        # 3M
        "scaling_law_exps_3_llama_3m_0.4.84",
        "remaining_runs_llama_3m_seed_55_0.4.134",
        "remaining_runs_llama_3m_seed_56_0.4.135",
        "scaling_law_4_more_seeds_llama_3m_seed_57_0.4.160",
        "scaling_law_4_more_seeds_llama_3m_seed_58_0.4.161",
        # 10M
        "scaling_law_exps_3_llama_10m_0.4.82",
        "remaining_runs_llama_10m_seed_55_0.4.132",
        "remaining_runs_llama_10m_seed_56_0.4.133",
        "scaling_law_4_more_seeds_llama_10m_seed_57_0.4.156",
        "scaling_law_4_more_seeds_llama_10m_seed_58_0.4.157",
        # 30M
        "scaling_law_exps_3_llama_30m_0.4.83",
        "depo_variance_impact_investigation_2_all_tasks_seed_55_0.4.120",
        "depo_variance_impact_investigation_2_all_tasks_seed_56_0.4.121",
        "scaling_law_4_more_seeds_llama_30m_seed_57_0.4.158",
        "scaling_law_4_more_seeds_llama_30m_seed_58_0.4.159",
        # 100M
        "scaling_law_exps_3_llama_100m_0.4.81",
        "seed_variance_comp_llama_seed_55_0.4.99",
        "seed_variance_comp_llama_seed_56_0.4.100",
        "scaling_law_4_more_seeds_llama_100m_seed_57_0.4.154",
        "scaling_law_4_more_seeds_llama_100m_seed_58_0.4.155",
    ],
}

# (wandb key, column name)
DEPO_COLS = [
    ("evals/synthetic/depo_edges_list/hop_4/accuracy",             "depo_edges"),
    ("evals/synthetic/depo_adj_list/hop_4/accuracy",               "depo_adj"),
]
OTHER_COLS = [
    ("evals/synthetic/bfs_edges_list/prefix_accuracy",             "BFS (edges)"),
    ("evals/synthetic/bfs_adj_list/prefix_accuracy",               "BFS (adj)"),
    ("evals/synthetic/shortest_path_edges_list/prefix_accuracy",   "SP (edges)"),
    ("evals/synthetic/shortest_path_adj_list/prefix_accuracy",     "SP (adj)"),
    ("evals/synthetic/concomp_factor_edges_list/prefix_accuracy",  "ConComp (edges)"),
    ("evals/synthetic/concomp_factor_adj_list/prefix_accuracy",    "ConComp (adj)"),
]
ALL_KEYS = [k for k, _ in DEPO_COLS + OTHER_COLS]

SCALE_TOKENS  = [("100m", "100M"), ("30m", "30M"), ("10m", "10M"), ("3m", "3M")]
SCALE_COLORS  = {"3M": "#aaaaaa", "10M": "#74add1", "30M": "#f46d43", "100M": "#313695"}
SCALE_MARKERS = {"3M": "o", "10M": "s", "30M": "^", "100M": "D"}


def _infer_scale(run_id: str) -> str:
    for tok, label in SCALE_TOKENS:
        if tok in run_id:
            return label
    return "?"


# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_all(api: wandb.Api) -> pd.DataFrame:
    rows = []
    for arch, ids in RUNS.items():
        print(f"\n{'─'*60}\nFetching {arch} …")
        for run_id in ids:
            try:
                summary = dict(api.run(f"{ENTITY}/{PROJECT}/{run_id}").summary)
                row = {"run_id": run_id, "arch": arch, "scale": _infer_scale(run_id)}
                for key, col in DEPO_COLS + OTHER_COLS:
                    v = summary.get(key)
                    row[col] = float(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else float("nan")
                rows.append(row)
                depo_str = "  ".join(
                    f"{col}={row[col]:.3f}" if np.isfinite(row[col]) else f"{col}=NaN"
                    for _, col in DEPO_COLS
                )
                print(f"  {run_id[-40:]:40s}  {depo_str}")
            except Exception as exc:
                print(f"  [warn] {run_id}: {exc}")

    df = pd.DataFrame(rows)
    # add quantized grokked columns
    for _, col in DEPO_COLS:
        df[f"{col}_grokked"] = df[col].apply(
            lambda v: int(v > DEPO_THRESHOLD) if np.isfinite(v) else float("nan")
        )
    return df


# ── Statistics ────────────────────────────────────────────────────────────────

def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Point-biserial r and Mann-Whitney p for every (depo_variant × other_task) pair."""
    rows = []
    for _, depo_col in DEPO_COLS:
        grokked_col = f"{depo_col}_grokked"
        valid = df.dropna(subset=[grokked_col])
        for _, other_col in OTHER_COLS:
            sub = valid.dropna(subset=[other_col])
            x   = sub[grokked_col].values.astype(float)
            y   = sub[other_col].values.astype(float)
            r, p_pb = stats.pointbiserialr(x, y)
            g_vals  = sub.loc[sub[grokked_col] == 1, other_col].values
            n_vals  = sub.loc[sub[grokked_col] == 0, other_col].values
            _, p_mw = (
                stats.mannwhitneyu(g_vals, n_vals, alternative="greater")
                if len(g_vals) > 0 and len(n_vals) > 0
                else (float("nan"), float("nan"))
            )
            rows.append({
                "depo":         depo_col,
                "task":         other_col,
                "r":            float(r),
                "p_mw":         float(p_mw),
                "mean_grokked": float(np.mean(g_vals)) if len(g_vals) else float("nan"),
                "mean_not":     float(np.mean(n_vals)) if len(n_vals) else float("nan"),
                "n_grokked":    int((x == 1).sum()),
                "n_not":        int((x == 0).sum()),
            })
    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame, corr: pd.DataFrame) -> None:
    print(f"\n{'='*76}")
    print(f"Exp 17 — Depo grokking × task correlation  (threshold = {DEPO_THRESHOLD})")
    print(f"{'='*76}")
    for _, depo_col in DEPO_COLS:
        grokked_col = f"{depo_col}_grokked"
        n_g = int((df[grokked_col] == 1).sum())
        n_n = int((df[grokked_col] == 0).sum())
        print(f"\n[{depo_col}]  grokked={n_g}  not={n_n}")
        sub = corr[corr["depo"] == depo_col]
        print(f"  {'Task':22s}  {'r':>7}  {'p_mw':>9}  {'mean(1)':>9}  {'mean(0)':>9}  {'Δ':>7}  sig")
        for _, row in sub.iterrows():
            delta = row["mean_grokked"] - row["mean_not"]
            sig   = "*" if row["p_mw"] < 0.05 else "n.s."
            print(f"  {row['task']:22s}  {row['r']:>7.3f}  {row['p_mw']:>9.4f}  "
                  f"{row['mean_grokked']:>9.3f}  {row['mean_not']:>9.3f}  {delta:>+7.3f}  {sig}")
    print(f"{'='*76}")


# ── Plots ─────────────────────────────────────────────────────────────────────

OTHER_LABELS = [col for _, col in OTHER_COLS]
DEPO_LABELS  = [col for _, col in DEPO_COLS]


def plot_scatter(df: pd.DataFrame, corr: pd.DataFrame) -> None:
    """2×6 scatter grid: depo (continuous) vs each other task, coloured by scale."""
    nrows, ncols = 2, len(OTHER_COLS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))

    for ri, depo_col in enumerate(DEPO_LABELS):
        sub_corr = corr[corr["depo"] == depo_col]
        for ci, other_col in enumerate(OTHER_LABELS):
            ax  = axes[ri, ci]
            sub = df.dropna(subset=[depo_col, other_col])
            for _, row in sub.iterrows():
                sc = row["scale"]
                ax.scatter(row[depo_col], row[other_col],
                           color=SCALE_COLORS.get(sc, "black"),
                           marker=SCALE_MARKERS.get(sc, "x"),
                           s=55, zorder=3, edgecolors="white", linewidths=0.5)

            ax.axvline(DEPO_THRESHOLD, color="black", lw=1.1, ls="--", alpha=0.6)
            row_c = sub_corr[sub_corr["task"] == other_col].iloc[0]
            sig   = "*" if row_c["p_mw"] < 0.05 else "n.s."
            ax.text(0.04, 0.96, f"r={row_c['r']:.2f} ({sig})",
                    transform=ax.transAxes, fontsize=8, va="top",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75))

            if ri == nrows - 1:
                ax.set_xlabel(other_col, fontsize=8)
            if ci == 0:
                ax.set_ylabel(f"{depo_col}\nhop_4 accuracy", fontsize=8)
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(other_col if ri == 0 else "", fontsize=9)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.25, ls="--")

    # row labels
    for ri, depo_col in enumerate(DEPO_LABELS):
        axes[ri, 0].set_ylabel(f"{depo_col}\nhop_4 accuracy", fontsize=8)

    # shared scale legend
    handles = [mpatches.Patch(color=SCALE_COLORS[s], label=s)
               for s in ["3M", "10M", "30M", "100M"]]
    handles.append(plt.Line2D([0], [0], ls="--", color="black",
                               label=f"threshold={DEPO_THRESHOLD}"))
    axes[0, -1].legend(handles=handles, title="Scale", fontsize=7,
                       loc="lower right", framealpha=0.8)

    fig.suptitle(
        "Depo hop_4 accuracy (continuous) vs other tasks\n"
        "(each point = one run;  r = point-biserial with grokked label)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "depo_corr_scatter.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_strips(df: pd.DataFrame, corr: pd.DataFrame) -> None:
    """2×6 strip+box grid: other task distribution split by depo_grokked."""
    rng   = np.random.default_rng(0)
    nrows, ncols = 2, len(OTHER_COLS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))

    for ri, depo_col in enumerate(DEPO_LABELS):
        grokked_col = f"{depo_col}_grokked"
        sub_corr = corr[corr["depo"] == depo_col]
        for ci, other_col in enumerate(OTHER_LABELS):
            ax  = axes[ri, ci]
            sub = df.dropna(subset=[grokked_col, other_col])
            for xi, (gval, color) in enumerate([(0, C_NOT), (1, C_GROKKED)]):
                vals   = sub.loc[sub[grokked_col] == gval, other_col].values
                if len(vals) == 0:
                    continue
                ax.boxplot(
                    vals, positions=[xi], widths=0.32, patch_artist=True,
                    medianprops=dict(color="white", lw=2.0),
                    boxprops=dict(facecolor=color, alpha=0.28),
                    whiskerprops=dict(color=color), capprops=dict(color=color),
                    flierprops=dict(marker=""),
                )
                scales = sub.loc[sub[grokked_col] == gval, "scale"].values
                jit    = rng.uniform(-0.12, 0.12, len(vals))
                for v, j, sc in zip(vals, jit, scales):
                    ax.scatter(xi + j, v,
                               color=SCALE_COLORS.get(sc, "black"),
                               marker=SCALE_MARKERS.get(sc, "x"),
                               s=50, zorder=4, edgecolors="white", linewidths=0.5)

            row_c = sub_corr[sub_corr["task"] == other_col].iloc[0]
            sig   = "*" if row_c["p_mw"] < 0.05 else "n.s."
            ax.text(0.5, 1.02,
                    f"r={row_c['r']:.2f}  p={row_c['p_mw']:.3f} {sig}",
                    ha="center", va="bottom", transform=ax.transAxes, fontsize=8)

            ax.set_xticks([0, 1])
            ax.set_xticklabels(["≤0.2", ">0.2"], fontsize=8)
            ax.set_ylim(-0.05, 1.12)
            ax.grid(axis="y", alpha=0.25, ls="--")
            if ri == nrows - 1:
                ax.set_xlabel("Depo grokked", fontsize=8)
            if ci == 0:
                ax.set_ylabel(f"{depo_col}: prefix_acc", fontsize=8)
            ax.set_title(other_col if ri == 0 else "", fontsize=9)
            ax.tick_params(labelsize=7)

    # row labels on y-axis
    for ri, depo_col in enumerate(DEPO_LABELS):
        axes[ri, 0].set_ylabel(f"Depo = {depo_col}\n\n{OTHER_LABELS[0]}", fontsize=7)
        axes[ri, 0].set_ylabel(depo_col.replace("_", "\n") + "\n→ " + OTHER_LABELS[0], fontsize=7)

    # scale legend
    handles = [mpatches.Patch(color=SCALE_COLORS[s], label=s)
               for s in ["3M", "10M", "30M", "100M"]]
    axes[0, -1].legend(handles=handles, title="Scale", fontsize=7,
                       loc="lower right", framealpha=0.8)

    n_total = df["depo_edges_grokked"].notna().sum()
    fig.suptitle(
        f"Task performance split by Depo grokking  (threshold={DEPO_THRESHOLD},  n={n_total} runs)\n"
        "Top row: depo_edges;  Bottom row: depo_adj",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "depo_corr_strips.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_correlation_bars(corr: pd.DataFrame) -> None:
    """Point-biserial r bars for all 12 (depo × task) combinations."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=False)

    for ax, depo_col in zip(axes, DEPO_LABELS):
        sub    = corr[corr["depo"] == depo_col].copy()
        tasks  = sub["task"].values
        r_vals = sub["r"].values
        p_vals = sub["p_mw"].values
        colors = [C_GROKKED if r >= 0 else C_NOT for r in r_vals]

        bars = ax.barh(tasks, r_vals, color=colors, alpha=0.82, height=0.5)
        for bar, r, p in zip(bars, r_vals, p_vals):
            if not np.isfinite(r):
                continue
            sig  = "*" if p < 0.05 else "n.s."
            xpos = r + 0.01 if r >= 0 else r - 0.01
            ha   = "left"    if r >= 0 else "right"
            ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                    f"r={r:.3f} ({sig})", va="center", ha=ha, fontsize=9)

        ax.axvline(0, color="black", lw=0.8, ls="--")
        ax.set_xlabel("Point-biserial r  (depo_grokked ↔ task metric)")
        ax.set_title(f"Depo = {depo_col}", fontsize=10, fontweight="bold")
        ax.set_xlim(-0.15, 1.15)
        ax.grid(axis="x", alpha=0.3, ls="--")

    fig.suptitle(
        f"Point-biserial correlation: Depo grokking (>{DEPO_THRESHOLD}) vs other tasks\n"
        "(prefix_accuracy for BFS / SP / ConComp;  hop_4/accuracy for Depo)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "depo_corr_bars.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_depo_edges_vs_concomp(df: pd.DataFrame, corr: pd.DataFrame) -> None:
    """Focused scatter: depo_edges (x) vs ConComp edges and adj (y), y clipped to [0.18, 0.42]."""
    concomp_cols = ["ConComp (edges)", "ConComp (adj)"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    sub_corr = corr[corr["depo"] == "depo_edges"]

    for ax, task_col in zip(axes, concomp_cols):
        sub = df.dropna(subset=["depo_edges", task_col])
        for _, row in sub.iterrows():
            sc = row["scale"]
            arch = row["arch"]
            marker = "o" if arch == "Canon" else "^"
            ax.scatter(row["depo_edges"], row[task_col],
                       color=SCALE_COLORS.get(sc, "black"),
                       marker=marker,
                       s=70, zorder=3, edgecolors="white", linewidths=0.6)

        ax.axvline(DEPO_THRESHOLD, color="black", lw=1.1, ls="--", alpha=0.6,
                   label=f"threshold={DEPO_THRESHOLD}")

        row_c = sub_corr[sub_corr["task"] == task_col].iloc[0]
        sig   = "*" if row_c["p_mw"] < 0.05 else "n.s."
        ax.text(0.04, 0.96, f"r = {row_c['r']:.3f} ({sig})",
                transform=ax.transAxes, fontsize=10, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.set_xlabel("Depo edges  hop_4/accuracy", fontsize=10)
        ax.set_ylabel(f"{task_col}  prefix_accuracy", fontsize=10)
        ax.set_title(task_col, fontsize=11, fontweight="bold")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0.18, 0.42)
        ax.grid(alpha=0.3, ls="--")

    # legend: scale colours + arch markers
    scale_handles = [mpatches.Patch(color=SCALE_COLORS[s], label=s)
                     for s in ["3M", "10M", "30M", "100M"]]
    arch_handles  = [
        plt.Line2D([0], [0], marker="o", color="gray", ls="none", label="Canon", markersize=7),
        plt.Line2D([0], [0], marker="^", color="gray", ls="none", label="Llama",  markersize=7),
    ]
    axes[1].legend(handles=scale_handles + arch_handles,
                   title="Scale / Arch", fontsize=8, loc="upper left", framealpha=0.85)

    fig.suptitle(
        "ConComp prefix_accuracy vs Depo edges hop_4/accuracy\n"
        "(40 scaling-law runs; colour = scale, marker = arch)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "depo_edges_vs_concomp.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    api = wandb.Api(api_key=WANDB_API_KEY)

    print("=" * 76)
    print("Exp 17 — Depo grokking × task correlation")
    print("=" * 76)

    df   = fetch_all(api)
    corr = compute_correlations(df)
    print_summary(df, corr)

    print("\nGenerating plots …")
    plot_scatter(df, corr)
    plot_strips(df, corr)
    plot_correlation_bars(corr)
    plot_depo_edges_vs_concomp(df, corr)

    print("\nDone.")
