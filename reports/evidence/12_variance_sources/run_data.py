"""
Where does the seed variance come from, and does task mixing actually help?

Three questions, answered with data (no new training):

  Q1  Which task is the noise source? For each of the 8 tasks we compute the
      seed-to-seed std of its final primary metric, pooled over all 30M / 8-task
      seeds (Llama + Canon). Ranks tasks from noisiest to most stable.

  Q2  Does mixing make the AGGREGATE training smoother across seeds? We compare
      the cross-seed std band of the aggregate loss/out trajectory for the
      depo-only mix vs the 8-task mix, and the per-curve jumpiness
      (mean |Δ| between consecutive points).

  Q3  Does mixing help DEPO ITSELF, or does Depo just crater to a floor? We track
      Depo (edges) hop-4 accuracy: its mean and seed-std in depo-only vs the
      8-task mix. Distinguishes "mixing stabilises the hard task" from
      "mixing drives the hard task to a reliably-failed floor."

Plots -> plots/:
  q1_per_task_noise.png     bar chart, per-task seed std (noisiness ranking)
  q2_aggregate_smoothness.png  loss/out trajectories, depo-only vs 8-task, mean+-std band
  q3_depo_in_isolation_vs_mix.png  Depo hop-4 acc: depo-only vs 8-task (mean, std, seeds)
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import (
    fetch_training_history, plot_variance_curves,
    TASK_NAMES, TASK_PRIMARY_METRIC, wandb_eval_key,
)

WANDB_API_KEY = "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"
ENTITY = "kirill456z"
PROJECT = "physics4llm"
OUT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(OUT_DIR, exist_ok=True)

C_RED, C_BLUE, C_GREY = "#d6604d", "#2166ac", "#888888"

# ── 30M / 8-task pool (same runs the scaling + dynamics analyses use) ─────────
POOL_30M = {
    "llama": [
        "scaling_law_exps_3_llama_30m_0.4.83",
        "depo_variance_impact_investigation_2_all_tasks_seed_55_0.4.120",
        "depo_variance_impact_investigation_2_all_tasks_seed_56_0.4.121",
        "scaling_law_4_more_seeds_llama_30m_seed_57_0.4.158",
        "scaling_law_4_more_seeds_llama_30m_seed_58_0.4.159",
    ],
    "canon": [
        "scaling_law_exps_3_canon_30m_0.4.79",
        "remaining_runs_canon_30m_seed_55_0.4.128",
        "remaining_runs_canon_30m_seed_56_0.4.129",
        "scaling_law_4_more_seeds_canon_30m_seed_57_0.4.150",
        "scaling_law_4_more_seeds_canon_30m_seed_58_0.4.151",
    ],
}

# ── depo-only (2-task) vs 8-task mixes, 3 matched seeds each ──────────────────
DEPO_ONLY = {
    "depo_only": [
        "seed_variance_exps_remaining_only_depo_55_0.4.186",
        "seed_variance_exps_remaining_only_depo_56_0.4.187",
        "seed_variance_exps_remaining_only_depo_57_0.4.188",
    ],
}
EIGHT_TASK = {
    "eight_task": [
        "depo_variance_impact_investigation_2_all_tasks_seed_55_0.4.120",
        "depo_variance_impact_investigation_2_all_tasks_seed_56_0.4.121",
        "scaling_law_4_more_seeds_llama_30m_seed_57_0.4.158",
    ],
}

DEPO_KEY = "evals/synthetic/depo_edges_list/hop_4/accuracy"


def _trailing(series, n):
    s = series.dropna()
    return float(s.iloc[-min(n, len(s)):].mean()) if len(s) else float("nan")


def _jumpiness(series):
    """Mean absolute step-to-step change of a (smoothed) trajectory."""
    s = series.dropna().to_numpy()
    return float(np.mean(np.abs(np.diff(s)))) if len(s) > 1 else float("nan")


# ── Q1: which task is the noise source? ───────────────────────────────────────
def q1_per_task_noise():
    print(f"\n{'='*78}\nQ1: per-task seed-to-seed std at 30M / 8-task (noise source ranking)\n{'='*78}")
    keys = [wandb_eval_key(t, TASK_PRIMARY_METRIC[t]) for t in TASK_NAMES]
    hists = fetch_training_history(
        POOL_30M, metric_keys=keys,
        api_key=WANDB_API_KEY, entity=ENTITY, project=PROJECT, samples=400)

    rows = []
    for t in TASK_NAMES:
        key = wandb_eval_key(t, TASK_PRIMARY_METRIC[t])
        vals = []
        for arch in POOL_30M:
            for rid, h in hists.get(arch, {}).items():
                if key in h.columns:
                    v = _trailing(h[key], 10)
                    if np.isfinite(v):
                        vals.append(v)
        if len(vals) > 1:
            rows.append(dict(task=t, mean=np.mean(vals), std=np.std(vals, ddof=1), n=len(vals)))
    df = pd.DataFrame(rows).sort_values("std", ascending=False).reset_index(drop=True)
    print(f"  {'task':>28}  {'mean':>7}  {'seed std':>8}  {'n':>3}")
    for _, r in df.iterrows():
        print(f"  {r['task']:>28}  {r['mean']:>7.3f}  {r['std']:>8.4f}  {int(r['n']):>3}")
    print("  Interpretation: the task(s) at the top dominate the aggregate seed variance.")
    return df


def plot_q1(df):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = [C_RED if t.startswith("depo") else C_BLUE for t in df["task"]]
    ax.barh(df["task"][::-1], df["std"][::-1], color=colors[::-1], alpha=0.85)
    ax.set_xlabel("Seed-to-seed std of final primary metric (30M, 8-task pool)")
    ax.set_title("Q1: Depo is the dominant noise source; BFS / ShortPath / ConComp are stable")
    ax.grid(axis="x", alpha=0.3, ls="--")
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "q1_per_task_noise.png"); fig.savefig(p, dpi=150); plt.close(fig)
    print(f"Saved: {p}")


# ── Q2: does mixing smooth the aggregate trajectory across seeds? ─────────────
def q2_aggregate_smoothness():
    print(f"\n{'='*78}\nQ2: aggregate loss/out smoothness — depo-only vs 8-task mix\n{'='*78}")
    groups = {**DEPO_ONLY, **EIGHT_TASK}
    hists = fetch_training_history(
        groups, metric_keys=["loss/out"],
        api_key=WANDB_API_KEY, entity=ENTITY, project=PROJECT, samples=400)

    stats = {}
    for g, runs in hists.items():
        jumps, finals = [], []
        for rid, h in runs.items():
            if "loss/out" not in h.columns:
                continue
            s = h[["_step", "loss/out"]].dropna().set_index("_step")["loss/out"]
            s_sm = s.rolling(10, min_periods=1).mean()
            jumps.append(_jumpiness(s_sm))
            finals.append(_trailing(s, 100))
        stats[g] = dict(jump=np.nanmean(jumps), final_std=np.std(finals, ddof=1) if len(finals) > 1 else np.nan,
                        final_mean=np.nanmean(finals), n=len(finals))
        print(f"  {g:>12}  n={stats[g]['n']}  final loss mean={stats[g]['final_mean']:.4f}  "
              f"cross-seed std={stats[g]['final_std']:.4f}  curve jumpiness={stats[g]['jump']:.5f}")
    print("  Interpretation: lower cross-seed std AND lower jumpiness on the 8-task mix\n"
          "  would mean mixing genuinely stabilises the aggregate training process.")

    # plot
    fig, ax = plt.subplots(figsize=(9, 5))
    nice = {"depo_only": "Depo only (2 tasks)", "eight_task": "8-task mix"}
    plot_variance_curves({nice[g]: hists[g] for g in groups}, "loss/out", ax=ax,
                         title="Q2: aggregate loss trajectory — depo-only is jumpy & seed-divergent,\n8-task mix is smooth & seed-consistent",
                         smoothing=10, palette=[C_RED, C_BLUE])
    ax.set_yscale("log"); ax.set_ylim(top=1.5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:g}"))
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "q2_aggregate_smoothness.png"); fig.savefig(p, dpi=150); plt.close(fig)
    print(f"Saved: {p}")
    return stats


# ── Q3: does mixing help DEPO, or does Depo just crater? ──────────────────────
def q3_depo_isolation_vs_mix():
    print(f"\n{'='*78}\nQ3: Depo (edges) hop-4 accuracy — depo-only vs 8-task mix\n{'='*78}")
    groups = {**DEPO_ONLY, **EIGHT_TASK}
    hists = fetch_training_history(
        groups, metric_keys=["loss/out", DEPO_KEY],
        api_key=WANDB_API_KEY, entity=ENTITY, project=PROJECT, samples=400)

    out = {}
    for g, runs in hists.items():
        vals = []
        for rid, h in runs.items():
            if DEPO_KEY in h.columns:
                v = _trailing(h[DEPO_KEY], 10)
                if np.isfinite(v):
                    vals.append(v)
        out[g] = vals
        m = np.mean(vals) if vals else np.nan
        s = np.std(vals, ddof=1) if len(vals) > 1 else np.nan
        print(f"  {g:>12}  n={len(vals)}  depo h4 acc mean={m:.4f}  seed std={s:.4f}  "
              f"seeds={[round(v,3) for v in vals]}")
    print("  Interpretation: if 8-task mix has LOWER depo std but also FAR LOWER depo mean,\n"
          "  mixing stabilises Depo by failing it reliably (floor), not by solving it.")

    fig, ax = plt.subplots(figsize=(7, 5))
    nice = {"depo_only": "Depo only\n(2 tasks)", "eight_task": "8-task mix"}
    labels = [nice[g] for g in groups]
    means = [np.mean(out[g]) if out[g] else np.nan for g in groups]
    stds = [np.std(out[g], ddof=1) if len(out[g]) > 1 else 0 for g in groups]
    xs = np.arange(len(groups))
    ax.bar(xs, means, yerr=stds, color=[C_RED, C_BLUE], alpha=0.8, capsize=6)
    for i, g in enumerate(groups):
        jit = np.random.default_rng(i).uniform(-0.1, 0.1, len(out[g]))
        ax.scatter(i + jit, out[g], color="black", s=40, zorder=3, alpha=0.7)
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_ylabel("Depo (edges) hop-4 accuracy")
    ax.set_title("Q3: does mixing help Depo, or floor it?")
    ax.grid(axis="y", alpha=0.3, ls="--")
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "q3_depo_in_isolation_vs_mix.png"); fig.savefig(p, dpi=150); plt.close(fig)
    print(f"Saved: {p}")
    return out


if __name__ == "__main__":
    q1 = q1_per_task_noise(); plot_q1(q1)
    q2 = q2_aggregate_smoothness()
    q3 = q3_depo_isolation_vs_mix()
    print("\nDone.")
