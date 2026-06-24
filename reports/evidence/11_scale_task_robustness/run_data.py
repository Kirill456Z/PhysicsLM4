"""
Exp A3 — Per-scale + per-task seed-robustness (Part 2 §4.2).

Two reanalyses, no new training:

  (Scale robustness) For each scale in {3M, 10M, 30M, 100M}, pool all available
  Canon and Llama seeds from the scaling-law experiments and apply the same
  permutation test used in Exp A (09_seed_flip). At 30M we also include the 3
  dynamics-batch seeds per arch (8+8 total), matching the full Exp-A pool.
  Expected: significant at 3M / 10M, null at 30M (already confirmed by Exp A),
  unknown at 100M.

  (Per-task Depo robustness at 30M) Depo hop-4 accuracy shows a large Canon gain
  (+0.57–0.71) at 30M even though the aggregate loss is null. We test whether
  that accuracy delta is seed-robust using the same 8+8 runs from Exp A plus a
  permutation test on per-run Depo hop-4 accuracy (edges_list and adj_list).
  If significant: §4.2 can assert "Depo gain is real even where the aggregate is
  not — aggregation hides the effect."

Decision rules:
  - p < 0.05 at ≤10M → Part 2 can assert "Canon is significantly better at small scale."
  - Depo accuracy delta significant at 30M → §4.2 can assert task-specific robustness.
  - All null → honest verdict is "Canon helps only where we haven't checked carefully yet."

Plots saved to plots/:
  scale_robustness.png          — p-values by scale + per-scale strip plots of final loss
  verdict_vs_seedcount_scales.png — multi-panel P(Canon better) vs seed count, one per scale
  depo_accuracy_robustness.png  — strip plot of Depo hop-4 accuracy + permutation result
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb

WANDB_API_KEY = "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"
ENTITY = "kirill456z"
PROJECT = "physics4llm"
OUT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(OUT_DIR, exist_ok=True)

LOSS_TRAILING = 100
RNG = np.random.default_rng(42)
N_PERM = 20_000
N_BOOT = 20_000

C_RED, C_BLUE = "#d6604d", "#2166ac"

DEPO_METRICS = [
    "evals/synthetic/depo_edges_list/hop_4/accuracy",
    "evals/synthetic/depo_adj_list/hop_4/accuracy",
]

# ── Run inventory ─────────────────────────────────────────────────────────────
# 5 scaling-law seeds per arch at each scale; 30M also includes 3 dynamics seeds

SCALE_RUNS = {
    "3M": {
        "Canon": [
            "scaling_law_exps_3_canon_3m_0.4.80",
            "remaining_runs_canon_3m_seed_55_0.4.130",
            "remaining_runs_canon_3m_seed_56_0.4.131",
            "scaling_law_4_more_seeds_canon_3m_seed_57_0.4.152",
            "scaling_law_4_more_seeds_canon_3m_seed_58_0.4.153",
        ],
        "Llama": [
            "scaling_law_exps_3_llama_3m_0.4.84",
            "remaining_runs_llama_3m_seed_55_0.4.134",
            "remaining_runs_llama_3m_seed_56_0.4.135",
            "scaling_law_4_more_seeds_llama_3m_seed_57_0.4.160",
            "scaling_law_4_more_seeds_llama_3m_seed_58_0.4.161",
        ],
    },
    "10M": {
        "Canon": [
            "scaling_law_exps_3_canon_10m_0.4.78",
            "remaining_runs_canon_10m_seed_55_0.4.126",
            "remaining_runs_canon_10m_seed_56_0.4.127",
            "scaling_law_4_more_seeds_canon_10m_seed_57_0.4.148",
            "scaling_law_4_more_seeds_canon_10m_seed_58_0.4.149",
        ],
        "Llama": [
            "scaling_law_exps_3_llama_10m_0.4.82",
            "remaining_runs_llama_10m_seed_55_0.4.132",
            "remaining_runs_llama_10m_seed_56_0.4.133",
            "scaling_law_4_more_seeds_llama_10m_seed_57_0.4.156",
            "scaling_law_4_more_seeds_llama_10m_seed_58_0.4.157",
        ],
    },
    "30M": {
        "Canon": [
            # scaling-law batch (5 seeds)
            "scaling_law_exps_3_canon_30m_0.4.79",
            "remaining_runs_canon_30m_seed_55_0.4.128",
            "remaining_runs_canon_30m_seed_56_0.4.129",
            "scaling_law_4_more_seeds_canon_30m_seed_57_0.4.150",
            "scaling_law_4_more_seeds_canon_30m_seed_58_0.4.151",
            # dynamics batch (3 seeds)
            "dynamics_on_synthetic_canon_s1_0.4.196",
            "dynamics_on_synthetic_canon_s2_0.4.197",
            "dynamics_on_synthetic_canon_s3_0.4.198",
        ],
        "Llama": [
            # scaling-law batch (5 seeds)
            "scaling_law_exps_3_llama_30m_0.4.83",
            "depo_variance_impact_investigation_2_all_tasks_seed_55_0.4.120",
            "depo_variance_impact_investigation_2_all_tasks_seed_56_0.4.121",
            "scaling_law_4_more_seeds_llama_30m_seed_57_0.4.158",
            "scaling_law_4_more_seeds_llama_30m_seed_58_0.4.159",
            # dynamics batch (3 seeds)
            "dynamics_on_synthetic_llama_s1_0.4.199",
            "dynamics_on_synthetic_llama_s2_0.4.200",
            "dynamics_on_synthetic_llama_s3_0.4.201",
        ],
    },
    "100M": {
        "Canon": [
            "scaling_law_exps_3_canon_100m_0.4.77",
            "canon_seed_variance_100m_canon_3e-4_seed_55_0.4.118",
            "canon_seed_variance_100m_canon_3e-4_seed_56_0.4.119",
            "scaling_law_4_more_seeds_canon_100m_seed_57_0.4.146",
            "scaling_law_4_more_seeds_canon_100m_seed_58_0.4.147",
        ],
        "Llama": [
            "scaling_law_exps_3_llama_100m_0.4.81",
            "seed_variance_comp_llama_seed_55_0.4.99",
            "seed_variance_comp_llama_seed_56_0.4.100",
            "scaling_law_4_more_seeds_llama_100m_seed_57_0.4.154",
            "scaling_law_4_more_seeds_llama_100m_seed_58_0.4.155",
        ],
    },
}

SCALES_ORDER = ["3M", "10M", "30M", "100M"]


# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_final_loss(run_id, api):
    """Return trailing-100 mean of loss/out for one run, or NaN on failure."""
    try:
        run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
        h = run.history(samples=400, keys=["loss/out"])
        s = h["loss/out"].dropna() if "loss/out" in h.columns else pd.Series(dtype=float)
        return float(s.iloc[-LOSS_TRAILING:].mean()) if len(s) >= 5 else float("nan")
    except Exception as exc:
        print(f"  [warn] {run_id}: {exc}")
        return float("nan")


def fetch_depo_accuracy(run_id, api):
    """Return mean of the last 20 eval points for each Depo metric."""
    results = {}
    try:
        run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
        h = run.history(samples=400, keys=DEPO_METRICS)
        for m in DEPO_METRICS:
            s = h[m].dropna() if m in h.columns else pd.Series(dtype=float)
            results[m] = float(s.iloc[-20:].mean()) if len(s) >= 3 else float("nan")
    except Exception as exc:
        print(f"  [warn] {run_id}: {exc}")
        for m in DEPO_METRICS:
            results[m] = float("nan")
    return results


def fetch_all_scales(api):
    """Return {scale: {"Canon": [loss,...], "Llama": [loss,...]}}."""
    data = {}
    for scale in SCALES_ORDER:
        print(f"\n{'─'*60}\nFetching {scale} …")
        data[scale] = {}
        for arch in ["Canon", "Llama"]:
            vals = []
            for rid in SCALE_RUNS[scale][arch]:
                v = fetch_final_loss(rid, api)
                status = f"{v:.4f}" if np.isfinite(v) else "NaN"
                print(f"  {arch:5s} {rid:55s} → {status}")
                vals.append(v)
            data[scale][arch] = [v for v in vals if np.isfinite(v)]
    return data


def fetch_depo_at_30m(api):
    """Return {"Canon": [...], "Llama": [...]} for each Depo metric, averaged."""
    print(f"\n{'─'*60}\nFetching Depo accuracy for 30M (8+8 seeds) …")
    rows = []
    for arch in ["Canon", "Llama"]:
        for rid in SCALE_RUNS["30M"][arch]:
            acc = fetch_depo_accuracy(rid, api)
            acc["arch"] = arch
            acc["run_id"] = rid
            print(f"  {arch:5s} {rid:55s} → "
                  f"el={acc[DEPO_METRICS[0]]:.4f}  al={acc[DEPO_METRICS[1]]:.4f}")
            rows.append(acc)
    return pd.DataFrame(rows)


# ── Statistics ────────────────────────────────────────────────────────────────

def permutation_test(canon, llama, n=N_PERM, rng=None):
    if rng is None:
        rng = RNG
    canon = np.asarray([v for v in canon if np.isfinite(v)], dtype=float)
    llama = np.asarray([v for v in llama if np.isfinite(v)], dtype=float)
    if len(canon) == 0 or len(llama) == 0:
        return float("nan"), float("nan")
    obs = llama.mean() - canon.mean()   # positive ⇒ Canon better (lower loss)
    pool = np.concatenate([canon, llama])
    nc = len(canon)
    count = sum(1 for _ in range(n)
                if abs(rng.permutation(pool)[nc:].mean() - rng.permutation(pool)[:nc].mean()) >= abs(obs))
    p = (count + 1) / (n + 1)
    return obs, p


def permutation_test_accuracy(canon, llama, n=N_PERM, rng=None):
    """For accuracy: canon higher is better. obs = canon.mean() - llama.mean()."""
    if rng is None:
        rng = RNG
    canon = np.asarray([v for v in canon if np.isfinite(v)], dtype=float)
    llama = np.asarray([v for v in llama if np.isfinite(v)], dtype=float)
    if len(canon) == 0 or len(llama) == 0:
        return float("nan"), float("nan")
    obs = canon.mean() - llama.mean()   # positive ⇒ Canon better (higher accuracy)
    pool = np.concatenate([canon, llama])
    nc = len(canon)
    count = sum(1 for _ in range(n)
                if abs(rng.permutation(pool)[nc:].mean() - rng.permutation(pool)[:nc].mean()) >= abs(obs))
    p = (count + 1) / (n + 1)
    return obs, p


def verdict_vs_seedcount(canon, llama, n=N_BOOT, higher_is_better_for_canon=False):
    """Bootstrap P(Canon judged better) for k=1..min(|canon|,|llama|) seeds."""
    canon = np.asarray([v for v in canon if np.isfinite(v)], dtype=float)
    llama = np.asarray([v for v in llama if np.isfinite(v)], dtype=float)
    kmax = min(len(canon), len(llama))
    ks, p_correct, gap_lo, gap_hi, gap_med = [], [], [], [], []
    for k in range(1, kmax + 1):
        gaps = np.empty(n)
        for i in range(n):
            c = RNG.choice(canon, size=k, replace=True).mean()
            l = RNG.choice(llama, size=k, replace=True).mean()
            if higher_is_better_for_canon:
                gaps[i] = c - l       # >0 ⇒ Canon better (for accuracy)
            else:
                gaps[i] = l - c       # >0 ⇒ Canon better (for loss, lower=better)
        ks.append(k)
        p_correct.append(float((gaps > 0).mean()))
        lo, hi = np.percentile(gaps, [2.5, 97.5])
        gap_lo.append(lo); gap_hi.append(hi); gap_med.append(float(np.median(gaps)))
    return dict(ks=ks, p=p_correct, lo=gap_lo, hi=gap_hi, med=gap_med)


# ── Print summaries ──────────────────────────────────────────────────────────

def print_scale_summary(scale_data):
    print(f"\n{'='*72}\nSCALE-BY-SCALE PERMUTATION TEST SUMMARY\n{'='*72}")
    print(f"  {'Scale':>6}  {'n_C':>4}  {'n_L':>4}  {'Canon mean':>11}  "
          f"{'Llama mean':>11}  {'gap':>8}  {'p-value':>9}  verdict")
    results = []
    for scale in SCALES_ORDER:
        c = np.asarray(scale_data[scale]["Canon"])
        l = np.asarray(scale_data[scale]["Llama"])
        gap, p = permutation_test(c.tolist(), l.tolist())
        sig = "**SIGNIFICANT**" if p < 0.05 else "null"
        print(f"  {scale:>6}  {len(c):>4}  {len(l):>4}  {c.mean():>11.4f}  "
              f"{l.mean():>11.4f}  {gap:>+8.4f}  {p:>9.4f}  {sig}")
        results.append(dict(scale=scale, nc=len(c), nl=len(l),
                            canon_mean=float(c.mean()), llama_mean=float(l.mean()),
                            gap=gap, p=p, significant=p < 0.05))
    return results


def print_depo_summary(depo_df):
    print(f"\n{'='*72}\nDEPO HOP-4 ACCURACY ROBUSTNESS AT 30M (8+8 seeds)\n{'='*72}")
    depo_results = []
    for metric in DEPO_METRICS:
        short = metric.split("/")[-3].replace("_list", "")  # depo_edges / depo_adj
        c_vals = depo_df[depo_df.arch == "Canon"][metric].dropna().tolist()
        l_vals = depo_df[depo_df.arch == "Llama"][metric].dropna().tolist()
        gap, p = permutation_test_accuracy(c_vals, l_vals)
        sig = "**SIGNIFICANT**" if p < 0.05 else "null"
        print(f"\n  [{short}]")
        print(f"    Canon  n={len(c_vals)}  mean={np.mean(c_vals):.4f}  "
              f"std={np.std(c_vals, ddof=1):.4f}")
        print(f"    Llama  n={len(l_vals)}  mean={np.mean(l_vals):.4f}  "
              f"std={np.std(l_vals, ddof=1):.4f}")
        print(f"    Canon−Llama gap = {gap:+.4f}   permutation p = {p:.4f}   {sig}")
        depo_results.append(dict(metric=short, gap=gap, p=p, significant=p < 0.05,
                                 canon_vals=c_vals, llama_vals=l_vals))
    return depo_results


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_scale_robustness(scale_data, scale_results):
    """
    Two-panel figure:
      Left  — p-value bar chart by scale (dashed line at α=0.05)
      Right — strip plot of final loss per arch per scale (4 groups of 2 columns)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: p-values
    ax = axes[0]
    ps = [r["p"] for r in scale_results]
    bar_colors = ["#2ca02c" if p < 0.05 else "#aaaaaa" for p in ps]
    bars = ax.bar(SCALES_ORDER, ps, color=bar_colors, alpha=0.85, edgecolor="white")
    ax.axhline(0.05, color="red", ls="--", lw=1.5, label="α = 0.05")
    ax.set_ylabel("Permutation p-value (two-sided)")
    ax.set_title("Scale-wise permutation test\n(green = significant at α=0.05)")
    ax.set_ylim(0, max(1.05, max(ps) * 1.1))
    ax.grid(axis="y", alpha=0.3, ls="--")
    ax.legend(fontsize=9)
    for bar, p in zip(bars, ps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"p={p:.3f}", ha="center", va="bottom", fontsize=9)

    # Right: strip plots
    ax = axes[1]
    n_scales = len(SCALES_ORDER)
    x_centers = np.arange(n_scales) * 3   # 3 units between scale groups
    for xi, scale in zip(x_centers, SCALES_ORDER):
        for arch_off, arch, color in [(-0.5, "Canon", C_RED), (0.5, "Llama", C_BLUE)]:
            vals = scale_data[scale][arch]
            jit = RNG.uniform(-0.2, 0.2, len(vals))
            ax.scatter(np.full(len(vals), xi + arch_off) + jit, vals,
                       color=color, alpha=0.75, s=55, edgecolors="white", linewidths=0.5,
                       label=arch if xi == x_centers[0] else "")
            ax.hlines(np.mean(vals), xi + arch_off - 0.3, xi + arch_off + 0.3,
                      color=color, lw=2.5, zorder=5)
    ax.set_xticks(x_centers)
    ax.set_xticklabels(SCALES_ORDER)
    ax.set_ylabel("Final loss/out (trailing-100 mean)")
    ax.set_title("Per-scale final loss (mean marked, all seeds shown)")
    ax.grid(axis="y", alpha=0.3, ls="--")
    ax.legend(fontsize=9)

    fig.suptitle("Exp A3 — Scale robustness: is Canon significantly better at each scale?",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "scale_robustness.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"Saved: {path}")


def plot_verdict_vs_seedcount_scales(scale_data):
    """4-panel plot: P(Canon better) vs seed count, one panel per scale."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for ax, scale in zip(axes, SCALES_ORDER):
        c = scale_data[scale]["Canon"]
        l = scale_data[scale]["Llama"]
        vc = verdict_vs_seedcount(c, l)
        ax.plot(vc["ks"], vc["p"], "o-", color="#444", lw=2)
        ax.fill_between(vc["ks"], vc["lo"], vc["hi"], alpha=0.15, color="#888")
        ax.axhline(0.95, color="green", ls="--", lw=1, label="95% confident")
        ax.axhline(0.5, color="gray", ls=":", lw=1)
        ax.set_xlabel("# seeds per arch")
        ax.set_ylabel("P(Canon better)")
        ax.set_ylim(0.0, 1.05)
        ax.set_title(f"{scale} (n={len(c)}+{len(l)})")
        ax.grid(alpha=0.3, ls="--")
        ax.legend(fontsize=7)
    fig.suptitle("Exp A3 — P(Canon judged better) vs seed count by scale",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "verdict_vs_seedcount_scales.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"Saved: {path}")


def plot_depo_accuracy_robustness(depo_df, depo_results):
    """
    Two-panel figure for Depo accuracy at 30M:
      Left  — strip plot per arch per metric (edges_list and adj_list)
      Right — verdict-vs-seed-count for edges_list (the more informative metric)
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: strip plots
    ax = axes[0]
    positions = {"edges": 0, "adj": 2}
    for short, metric in zip(["edges", "adj"], DEPO_METRICS):
        xi = positions[short]
        for arch_off, arch, color in [(-0.4, "Canon", C_RED), (0.4, "Llama", C_BLUE)]:
            vals = depo_df[depo_df.arch == arch][metric].dropna().tolist()
            jit = RNG.uniform(-0.15, 0.15, len(vals))
            ax.scatter(np.full(len(vals), xi + arch_off) + jit, vals,
                       color=color, alpha=0.8, s=60, edgecolors="white", linewidths=0.5,
                       label=arch if short == "edges" else "")
            ax.hlines(np.mean(vals), xi + arch_off - 0.25, xi + arch_off + 0.25,
                      color=color, lw=2.5, zorder=5)

    # annotate p-values
    for i, (short, dr) in enumerate(zip(["edges", "adj"], depo_results)):
        xi = positions[short]
        p_str = f"p={dr['p']:.3f}"
        sig_str = "✓" if dr["significant"] else "n.s."
        ax.text(xi, ax.get_ylim()[1] if ax.get_ylim()[1] < 1 else 0.95,
                f"Δ={dr['gap']:+.3f}\n{p_str} {sig_str}",
                ha="center", va="bottom", fontsize=9,
                color="green" if dr["significant"] else "#888")

    ax.set_xticks([0, 2])
    ax.set_xticklabels(["Depo (edges_list)", "Depo (adj_list)"])
    ax.set_ylabel("Hop-4 accuracy (final 20-eval mean)")
    ax.set_title("Depo hop-4 accuracy at 30M (8+8 seeds)\nCanon vs Llama with permutation p")
    ax.grid(axis="y", alpha=0.3, ls="--")
    ax.legend(fontsize=9)

    # Right: verdict-vs-seedcount for edges_list
    ax = axes[1]
    c_vals = depo_df[depo_df.arch == "Canon"][DEPO_METRICS[0]].dropna().tolist()
    l_vals = depo_df[depo_df.arch == "Llama"][DEPO_METRICS[0]].dropna().tolist()
    vc = verdict_vs_seedcount(c_vals, l_vals, higher_is_better_for_canon=True)
    ax.plot(vc["ks"], vc["p"], "o-", color="#444", lw=2)
    ax.fill_between(vc["ks"], vc["lo"], vc["hi"], alpha=0.15, color="#888",
                    label="95% gap interval")
    ax.axhline(0.95, color="green", ls="--", lw=1, label="95% confident")
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_xlabel("# seeds per arch")
    ax.set_ylabel("P(Canon judged better on Depo)")
    ax.set_title("Verdict vs seed count — Depo edges_list accuracy\n"
                 "(does the sign stabilise with more seeds?)")
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3, ls="--")
    ax.legend(fontsize=9)

    fig.suptitle("Exp A3 — Depo accuracy gain at 30M: seed-robust even where aggregate is null?",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "depo_accuracy_robustness.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    api = wandb.Api(api_key=WANDB_API_KEY)

    print("=" * 72)
    print("Exp A3 — Per-scale + per-task seed-robustness")
    print("=" * 72)

    # 1. Fetch final loss at each scale
    scale_data = fetch_all_scales(api)

    # 2. Scale permutation tests + summary
    scale_results = print_scale_summary(scale_data)

    # 3. Fetch Depo accuracy for the 30M / 8+8 pool
    depo_df = fetch_depo_at_30m(api)

    # 4. Depo permutation tests + summary
    depo_results = print_depo_summary(depo_df)

    # 5. Plots
    plot_scale_robustness(scale_data, scale_results)
    plot_verdict_vs_seedcount_scales(scale_data)
    plot_depo_accuracy_robustness(depo_df, depo_results)

    print("\n" + "=" * 72)
    print("DECISION SUMMARY")
    print("=" * 72)
    sig_scales = [r["scale"] for r in scale_results if r["significant"]]
    if sig_scales:
        print(f"  Significant aggregate loss gap at: {', '.join(sig_scales)}")
        print("  → Part 2 CAN assert Canon is significantly better at small scale.")
    else:
        print("  No scale shows a significant aggregate loss gap.")
        print("  → Canon helps only conditionally; the methodological cautionary "
              "tale framing stands.")

    depo_sig = [r for r in depo_results if r["significant"]]
    if depo_sig:
        tasks = ", ".join(r["metric"] for r in depo_sig)
        print(f"\n  Depo accuracy delta is SIGNIFICANT at 30M ({tasks}).")
        print("  → §4.2 CAN assert: 'Depo gain is real even where the aggregate is null'.")
    else:
        print("\n  Depo accuracy delta is NOT significant at 30M.")
        print("  → Depo gain is also unreliable at the 8-seed level.")

    print("\nDone.")
