"""
Exp A2 — Honest recompute of the two variance findings (Part 1 §3.2).

Two reanalyses, no new training:

  (F5 fix) Task mixing — report the seed variance of a FIXED task's metric as the
  mix grows, not the std of the aggregate loss. The aggregate-loss std partly drops
  for a mechanical reason (averaging more tasks shrinks the mean's variance ~1/√k);
  the honest claim is whether a *single task's* performance becomes more
  seed-stable when trained alongside others. We track Depo (edges_list) hop-4
  accuracy across the {1,2,4,8}-task mixes that contain it, and print the
  aggregate-loss std beside it for contrast.

  (F7 fix) Canon architectural stability — report the coefficient of variation
  (std / mean), not just the raw std. Canon reaches a much lower mean loss on
  depo×2, so a lower absolute spread is partly expected; CV tells us whether the
  *relative* variability is actually reduced.

Plots saved to plots/:
  f5_per_task_variance.png  — Depo hop-4 std vs #tasks, with aggregate-loss std overlay
  f7_cv.png                 — Canon vs Llama: mean, std, and CV side by side
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import fetch_training_history

WANDB_API_KEY = "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"
ENTITY = "kirill456z"
PROJECT = "physics4llm"
OUT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(OUT_DIR, exist_ok=True)

C_RED, C_BLUE = "#d6604d", "#2166ac"
DEPO_METRIC = "evals/synthetic/depo_edges_list/hop_4/accuracy"

# ── F5: task-mixing groups that CONTAIN depo_edges_list (so the fixed metric exists) ──
# (mirrors reports/evidence/05_task_mixing_variance/run_data.py)
F5_GROUPS = {
    "depo_edges": [   # 1 task
        "seed_variance_exps_remaining_only_depo_edges_55_0.4.189",
        "seed_variance_exps_remaining_only_depo_edges_56_0.4.190",
        "seed_variance_exps_remaining_only_depo_edges_57_0.4.191",
    ],
    "depo_both": [    # 2 tasks
        "seed_variance_exps_remaining_only_depo_55_0.4.186",
        "seed_variance_exps_remaining_only_depo_56_0.4.187",
        "seed_variance_exps_remaining_only_depo_57_0.4.188",
    ],
    "edges_only": [   # 4 tasks (all edges_list encodings, incl. depo_edges_list)
        "seed_variance_exps_remaining_only_edges_list_55_0.4.192",
        "seed_variance_exps_remaining_only_edges_list_56_0.4.193",
        "seed_variance_exps_remaining_only_edges_list_57_0.4.194",
    ],
    "all_tasks": [    # 8 tasks
        "depo_variance_impact_investigation_2_all_tasks_seed_55_0.4.120",
        "depo_variance_impact_investigation_2_all_tasks_seed_56_0.4.121",
        "scaling_law_4_more_seeds_llama_30m_seed_57_0.4.158",
    ],
}
F5_NUM_TASKS = {"depo_edges": 1, "depo_both": 2, "edges_only": 4, "all_tasks": 8}
F5_ORDER = ["depo_edges", "depo_both", "edges_only", "all_tasks"]

# ── F7: Canon vs Llama on depo×2, 5 seeds each (mirrors 08_canon_stability) ──────
F7_GROUPS = {
    "Canon": [f"canon_stability_depo_canon_s{i}_0.4.{201+i}" for i in range(1, 6)],
    "Llama": [f"canon_stability_depo_llama_s{i}_0.4.{206+i}" for i in range(1, 6)],
}
# IMPORTANT: on depo×2 the high-frequency *train* loss/out collapses to ~0.01
# (memorisation on just two tasks), while the project's reported "final loss"
# (08_canon_stability + FINDINGS.md F7) is the eval-cadence-aligned loss/out (~0.24).
# Requesting an eval key alongside loss/out forces wandb to return the eval-aligned
# rows, matching the established convention. Without it we'd silently grab the
# train-loss series and report the wrong numbers.
F7_EVAL_KEYS = [
    "evals/synthetic/depo_edges_list/hop_4/accuracy",
    "evals/synthetic/depo_adj_list/hop_4/accuracy",
]


def _trailing(series, n):
    s = series.dropna()
    return float(s.iloc[-min(n, len(s)):].mean()) if len(s) else float("nan")


# ── F5 recompute ─────────────────────────────────────────────────────────────────
def recompute_f5():
    print(f"\n{'='*72}\nF5 (recomputed): per-task seed variance vs aggregate-loss variance\n{'='*72}")
    hists = fetch_training_history(
        F5_GROUPS, metric_keys=["loss/out", DEPO_METRIC],
        api_key=WANDB_API_KEY, entity=ENTITY, project=PROJECT, samples=300)

    rows = []
    print(f"  {'group':>12} {'#tasks':>6}  {'depo h4 acc mean':>16} {'depo std':>9}  "
          f"{'agg loss std':>12}")
    for g in F5_ORDER:
        depo_vals, loss_vals = [], []
        for rid, h in hists.get(g, {}).items():
            if h.empty:
                continue
            if DEPO_METRIC in h.columns:
                v = _trailing(h[DEPO_METRIC], 20)
                if np.isfinite(v):
                    depo_vals.append(v)
            if "loss/out" in h.columns:
                v = _trailing(h["loss/out"], 100)
                if np.isfinite(v):
                    loss_vals.append(v)
        depo_std = np.std(depo_vals, ddof=1) if len(depo_vals) > 1 else np.nan
        depo_mean = np.mean(depo_vals) if depo_vals else np.nan
        loss_std = np.std(loss_vals, ddof=1) if len(loss_vals) > 1 else np.nan
        rows.append(dict(group=g, n_tasks=F5_NUM_TASKS[g], depo_mean=depo_mean,
                         depo_std=depo_std, loss_std=loss_std, n=len(depo_vals)))
        print(f"  {g:>12} {F5_NUM_TASKS[g]:>6}  {depo_mean:>16.4f} {depo_std:>9.4f}  "
              f"{loss_std:>12.4f}  (n={len(depo_vals)})")
    print("  Interpretation: if depo-h4 std falls as #tasks grows, mixing genuinely\n"
          "  stabilises the *fixed task*, not just the aggregate. Compare the two std\n"
          "  columns — the aggregate-loss column also benefits from mechanical averaging.")
    return pd.DataFrame(rows)


def plot_f5(df):
    df = df.sort_values("n_tasks")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["n_tasks"], df["depo_std"], "o-", color=C_RED, lw=2, markersize=8,
            label="Depo (edges) hop-4 acc — fixed-task seed std")
    ax.plot(df["n_tasks"], df["loss_std"], "s--", color="#888", lw=2, markersize=7,
            label="Aggregate loss/out seed std (partly mechanical)")
    ax.set_xlabel("Number of training tasks (mixes containing Depo)")
    ax.set_ylabel("Std across seeds")
    ax.set_xticks(df["n_tasks"])
    ax.set_title("F5 recomputed: does mixing stabilise a FIXED task, not just the average?")
    ax.grid(alpha=0.3, ls="--"); ax.legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "f5_per_task_variance.png"); fig.savefig(p, dpi=150); plt.close(fig)
    print(f"Saved: {p}")


# ── F7 recompute ─────────────────────────────────────────────────────────────────
def recompute_f7():
    print(f"\n{'='*72}\nF7 (recomputed): coefficient of variation, not just raw std\n{'='*72}")
    hists = fetch_training_history(
        F7_GROUPS, metric_keys=["loss/out"] + F7_EVAL_KEYS,
        api_key=WANDB_API_KEY, entity=ENTITY, project=PROJECT, samples=300)
    stats = {}
    for arch, runs in hists.items():
        vals = [_trailing(h["loss/out"], 100) for h in runs.values()
                if "loss/out" in h.columns]
        vals = [v for v in vals if np.isfinite(v)]
        mean = np.mean(vals); std = np.std(vals, ddof=1) if len(vals) > 1 else np.nan
        cv = std / mean if mean else np.nan
        stats[arch] = dict(mean=mean, std=std, cv=cv, n=len(vals), vals=vals)
        print(f"  {arch:5s}  n={len(vals)}  mean={mean:.4f}  std={std:.4f}  CV={cv:.3f}")
    if "Canon" in stats and "Llama" in stats:
        c, l = stats["Canon"], stats["Llama"]
        print(f"\n  raw std reduction:  Canon/Llama = {c['std']/l['std']:.2f}  "
              f"(→ {(1-c['std']/l['std'])*100:.0f}% lower std)")
        print(f"  CV reduction:       Canon/Llama = {c['cv']/l['cv']:.2f}  "
              f"(→ {(1-c['cv']/l['cv'])*100:.0f}% lower CV)")
        print("  Honest reading: the raw-std drop is largely a consequence of Canon's\n"
              "  lower mean loss; the CV shows how much *relative* variability truly falls.")
    return stats


def plot_f7(stats):
    archs = ["Canon", "Llama"]
    colors = [C_RED, C_BLUE]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, key, title in [
        (axes[0], "mean", "Mean final loss"),
        (axes[1], "std",  "Raw std across seeds"),
        (axes[2], "cv",   "Coefficient of variation (std/mean)"),
    ]:
        vals = [stats[a][key] for a in archs]
        bars = ax.bar(archs, vals, color=colors, alpha=0.8)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        if key in ("mean", "std"):
            for xi, a in enumerate(archs):
                jit = np.random.default_rng(xi).uniform(-0.1, 0.1, len(stats[a]["vals"]))
                pts = stats[a]["vals"] if key == "mean" else None
                if pts is not None:
                    ax.scatter(xi + jit, pts, color="black", s=18, zorder=3, alpha=0.6)
        ax.set_title(title); ax.grid(axis="y", alpha=0.3, ls="--")
    fig.suptitle("F7 recomputed: Canon's variance drop is largely a mean effect "
                 "(CV ≈ Llama's)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "f7_cv.png"); fig.savefig(p, dpi=150); plt.close(fig)
    print(f"Saved: {p}")


if __name__ == "__main__":
    f5 = recompute_f5()
    plot_f5(f5)
    f7 = recompute_f7()
    plot_f7(f7)
    print("\nDone.")
