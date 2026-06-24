"""
Exp B — De-ceiling BFS/SP (Part 2 §4.2).

Training: 30M Canon-ABCD vs Llama on BFS×2 + ShortestPath×2 at hard difficulty
(max_nodes=120), 3 seeds each. The standard difficulty has Llama near ceiling
(0.91–0.97); the raised difficulty puts Llama mid-range (~0.5–0.7) so the
headroom confound is removed.

Runs:
  Canon: bfs_sp_hard_canon_s{1,2,3}_0.4.{213,214,215}
  Llama:  bfs_sp_hard_llama_s{1,2,3}_0.4.{216,217,218}

Decision rule (from MISSING_RUNS_AFTER_PIVOT.md):
  Canon Δ >> Exp-A noise band (~0.015 on aggregate loss, ~0.11 on Depo accuracy)
      → ceiling artifact; Canon helps wherever there is headroom; drop the
        feature-specialisation / long-range-specific framing.
  Canon Δ ≈ 0 on hard BFS/SP (while Depo Δ stays large)
      → Canon IS genuinely specific to long-range / multi-hop reasoning;
        §4.2 + §4.3 mechanism story is earned.

Metrics fetched per run:
  loss/out                                       (aggregate)
  evals/synthetic/bfs_edges_list/set_recall
  evals/synthetic/bfs_adj_list/set_recall
  evals/synthetic/shortest_path_edges_list/set_accuracy
  evals/synthetic/shortest_path_adj_list/set_accuracy

Plots saved to plots/:
  bfs_sp_loss_comparison.png        — final loss strip plot + permutation p
  bfs_sp_accuracy_comparison.png    — per-task accuracy strip plots + p-values
  bfs_sp_training_curves.png        — loss/out training curves Canon vs Llama
  bfs_sp_accuracy_curves.png        — accuracy training curves for all 4 subtasks
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

LOSS_TRAILING = 100   # steps for final-loss average
ACC_TRAILING  = 20    # eval points for final-accuracy average
RNG = np.random.default_rng(42)
N_PERM = 20_000

C_RED,  C_BLUE  = "#d6604d", "#2166ac"

# ── Run inventory ──────────────────────────────────────────────────────────────
RUNS = {
    "Canon": [
        "bfs_sp_hard_canon_s1_0.4.213",
        "bfs_sp_hard_canon_s2_0.4.214",
        "bfs_sp_hard_canon_s3_0.4.215",
    ],
    "Llama": [
        "bfs_sp_hard_llama_s1_0.4.216",
        "bfs_sp_hard_llama_s2_0.4.217",
        "bfs_sp_hard_llama_s3_0.4.218",
    ],
}

# Metrics: (wandb_key, short_label, higher_is_better)
# Note: the hard-difficulty generators log under bfs_hard_* and shortest_path_hard_*
TASK_METRICS = [
    ("evals/synthetic/bfs_hard_edges_list/set_recall",           "BFS (edges)",  True),
    ("evals/synthetic/bfs_hard_adj_list/set_recall",             "BFS (adj)",    True),
    ("evals/synthetic/shortest_path_hard_edges_list/set_accuracy","SP (edges)",   True),
    ("evals/synthetic/shortest_path_hard_adj_list/set_accuracy", "SP (adj)",     True),
]

# Exp-A noise reference for the aggregate loss gap (30M / 8-task mix)
EXP_A_LOSS_NOISE = 0.015   # 1-sigma-ish gap from 09_seed_flip


# ── Data fetching ──────────────────────────────────────────────────────────────

def fetch_run(run_id, api):
    """Return a dict with final_loss and per-task final accuracies for one run."""
    result = {"run_id": run_id, "final_loss": float("nan")}
    for _, label, _ in TASK_METRICS:
        result[label] = float("nan")
    try:
        run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")

        # loss
        h_loss = run.history(samples=500, keys=["loss/out"])
        s_loss = h_loss["loss/out"].dropna() if "loss/out" in h_loss.columns else pd.Series(dtype=float)
        if len(s_loss) >= 5:
            result["final_loss"] = float(s_loss.iloc[-LOSS_TRAILING:].mean())
        result["_loss_series"] = s_loss

        # task accuracies
        acc_keys = [m for m, _, _ in TASK_METRICS]
        h_acc = run.history(samples=500, keys=acc_keys)
        for key, label, _ in TASK_METRICS:
            s = h_acc[key].dropna() if key in h_acc.columns else pd.Series(dtype=float)
            if len(s) >= 3:
                result[label] = float(s.iloc[-ACC_TRAILING:].mean())
            result[f"_series_{label}"] = s

    except Exception as exc:
        print(f"  [warn] {run_id}: {exc}")
    return result


def fetch_all(api):
    rows = []
    for arch in ["Canon", "Llama"]:
        print(f"\n{'─'*60}\nFetching {arch} …")
        for rid in RUNS[arch]:
            r = fetch_run(rid, api)
            r["arch"] = arch
            loss_str = f"{r['final_loss']:.4f}" if np.isfinite(r["final_loss"]) else "NaN"
            acc_strs = "  ".join(
                f"{label}={r[label]:.3f}" if np.isfinite(r[label]) else f"{label}=NaN"
                for _, label, _ in TASK_METRICS
            )
            print(f"  {arch:6s} {rid:45s}  loss={loss_str}  {acc_strs}")
            rows.append(r)
    return rows


# ── Statistics ─────────────────────────────────────────────────────────────────

def permutation_test(canon_vals, llama_vals, higher_is_better_for_canon=False):
    c = np.asarray([v for v in canon_vals if np.isfinite(v)], dtype=float)
    l = np.asarray([v for v in llama_vals if np.isfinite(v)], dtype=float)
    if len(c) == 0 or len(l) == 0:
        return float("nan"), float("nan")
    if higher_is_better_for_canon:
        obs = c.mean() - l.mean()   # >0 ⇒ Canon better (accuracy)
    else:
        obs = l.mean() - c.mean()   # >0 ⇒ Canon better (lower loss)
    pool = np.concatenate([c, l])
    nc = len(c)
    count = sum(
        1 for _ in range(N_PERM)
        if abs(RNG.permutation(pool)[nc:].mean() - RNG.permutation(pool)[:nc].mean()) >= abs(obs)
    )
    p = (count + 1) / (N_PERM + 1)
    return float(obs), float(p)


# ── Print summary ──────────────────────────────────────────────────────────────

def print_summary(rows):
    canon_rows = [r for r in rows if r["arch"] == "Canon"]
    llama_rows = [r for r in rows if r["arch"] == "Llama"]

    print(f"\n{'='*72}")
    print("Exp B — De-ceiling BFS/SP: Canon vs Llama at hard difficulty")
    print(f"{'='*72}")

    # Aggregate loss
    c_loss = [r["final_loss"] for r in canon_rows]
    l_loss = [r["final_loss"] for r in llama_rows]
    obs, p = permutation_test(c_loss, l_loss, higher_is_better_for_canon=False)
    sig = "**SIGNIFICANT**" if p < 0.05 else "null"
    print(f"\n[Aggregate loss/out]")
    print(f"  Canon  n={len([v for v in c_loss if np.isfinite(v)])}  "
          f"mean={np.nanmean(c_loss):.4f}  std={np.nanstd(c_loss, ddof=1):.4f}")
    print(f"  Llama  n={len([v for v in l_loss if np.isfinite(v)])}  "
          f"mean={np.nanmean(l_loss):.4f}  std={np.nanstd(l_loss, ddof=1):.4f}")
    print(f"  Llama−Canon gap = {obs:+.4f}   p = {p:.4f}   {sig}")
    print(f"  (Exp-A noise reference on 8-task mix: ≈ {EXP_A_LOSS_NOISE:.3f})")

    results = [dict(metric="loss/out", gap=obs, p=p, significant=p < 0.05,
                    canon_mean=np.nanmean(c_loss), llama_mean=np.nanmean(l_loss),
                    higher_is_better_for_canon=False)]

    # Per-task accuracies
    print(f"\n[Per-task accuracy]")
    print(f"  {'Task':25s}  {'Canon mean':>12}  {'Llama mean':>12}  {'Δ (Canon−Llama)':>16}  {'p':>8}  verdict")
    for _, label, hib in TASK_METRICS:
        c_vals = [r[label] for r in canon_rows]
        l_vals = [r[label] for r in llama_rows]
        obs_a, p_a = permutation_test(c_vals, l_vals, higher_is_better_for_canon=True)
        sig_a = "**SIGNIFICANT**" if p_a < 0.05 else "null"
        c_mean = np.nanmean(c_vals)
        l_mean = np.nanmean(l_vals)
        print(f"  {label:25s}  {c_mean:>12.4f}  {l_mean:>12.4f}  {obs_a:>+16.4f}  {p_a:>8.4f}  {sig_a}")
        results.append(dict(metric=label, gap=obs_a, p=p_a, significant=p_a < 0.05,
                            canon_mean=c_mean, llama_mean=l_mean,
                            higher_is_better_for_canon=True))

    return results


def print_decision(results):
    loss_r = next(r for r in results if r["metric"] == "loss/out")
    acc_results = [r for r in results if r["metric"] != "loss/out"]

    print(f"\n{'='*72}")
    print("DECISION")
    print(f"{'='*72}")

    sig_tasks = [r for r in acc_results if r["significant"]]
    null_tasks = [r for r in acc_results if not r["significant"]]
    large_gap_tasks = [r for r in acc_results
                       if abs(r["gap"]) > 0.10 and r["p"] < 0.05]

    if large_gap_tasks:
        print("\n  CEILING ARTIFACT verdict:")
        print("  Canon shows a SIGNIFICANT and LARGE accuracy gain on hard BFS/SP.")
        print("  The earlier 'concentration on Depo' was a ceiling/headroom artifact,")
        print("  not a specificity to long-range reasoning.")
        print("  → Soften §4.2: 'Canon helps wherever there is headroom.'")
        print("  → DROP the feature-specialisation / long-range-specific mechanism framing.")
    elif not sig_tasks:
        print("\n  GENUINE SPECIFICITY verdict:")
        print("  Canon shows NO significant accuracy gain on hard BFS/SP.")
        print("  Combined with the large, robust Depo gain (+0.11–0.17, Exp A3),")
        print("  this supports Canon being GENUINELY SPECIFIC to long-range / multi-hop.")
        print("  → §4.2 + §4.3 mechanism story is EARNED.")
        print("  → Keep the feature-specialisation framing.")
    else:
        print(f"\n  MIXED verdict: {len(sig_tasks)}/{len(acc_results)} tasks significant.")
        for r in acc_results:
            tag = "SIG" if r["significant"] else "n.s."
            print(f"    {r['metric']:25s}  Δ={r['gap']:+.4f}  p={r['p']:.4f}  {tag}")
        print("  Interpret with caution; see plots for per-task detail.")

    # Llama baseline check — was the difficulty bump effective?
    llama_accs = [r["llama_mean"] for r in acc_results if np.isfinite(r["llama_mean"])]
    mean_llama_acc = float(np.mean(llama_accs)) if llama_accs else float("nan")
    print(f"\n  Llama mean accuracy on hard tasks = {mean_llama_acc:.3f}")
    if mean_llama_acc < 0.85:
        print("  → Difficulty bump EFFECTIVE: Llama is out of the ceiling region.")
    else:
        print("  → WARNING: Llama still near ceiling (≥0.85); difficulty bump may be insufficient.")


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_loss_comparison(rows, results):
    """Strip plot of final loss/out with permutation p-value annotation."""
    canon_rows = [r for r in rows if r["arch"] == "Canon"]
    llama_rows = [r for r in rows if r["arch"] == "Llama"]
    c_loss = [r["final_loss"] for r in canon_rows]
    l_loss = [r["final_loss"] for r in llama_rows]

    fig, ax = plt.subplots(figsize=(6, 5))
    for xi, arch, vals, color in [(0, "Canon", c_loss, C_RED), (1, "Llama", l_loss, C_BLUE)]:
        finite = [v for v in vals if np.isfinite(v)]
        jit = RNG.uniform(-0.12, 0.12, len(finite))
        ax.scatter(np.full(len(finite), xi) + jit, finite,
                   color=color, s=90, alpha=0.85, edgecolors="white", linewidths=0.6,
                   label=arch, zorder=4)
        if finite:
            ax.hlines(np.mean(finite), xi - 0.22, xi + 0.22,
                      color=color, lw=2.8, zorder=5)

    loss_r = next(r for r in results if r["metric"] == "loss/out")
    sig_str = "p={:.3f} {}".format(loss_r["p"], "✓" if loss_r["significant"] else "n.s.")
    ymax = max(
        [v for v in c_loss + l_loss if np.isfinite(v)], default=1.0
    )
    ax.text(0.5, ymax * 1.01, sig_str, ha="center", va="bottom", fontsize=10,
            color="green" if loss_r["significant"] else "#888")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Canon-ABCD", "Llama"])
    ax.set_ylabel("Final loss/out (trailing-100 mean)")
    ax.set_title("Exp B — Aggregate loss on hard BFS/SP\n(max_nodes=120, 30M, 3 seeds each)")
    ax.grid(axis="y", alpha=0.3, ls="--")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "bfs_sp_loss_comparison.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"Saved: {path}")


def plot_accuracy_comparison(rows, results):
    """4-panel strip plots, one per task metric, with p-value annotations."""
    canon_rows = [r for r in rows if r["arch"] == "Canon"]
    llama_rows = [r for r in rows if r["arch"] == "Llama"]
    acc_results = [r for r in results if r["metric"] != "loss/out"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, (_, label, _), res in zip(axes, TASK_METRICS, acc_results):
        c_vals = [r[label] for r in canon_rows]
        l_vals = [r[label] for r in llama_rows]
        for xi, arch, vals, color in [(0, "Canon", c_vals, C_RED), (1, "Llama", l_vals, C_BLUE)]:
            finite = [v for v in vals if np.isfinite(v)]
            jit = RNG.uniform(-0.12, 0.12, len(finite))
            ax.scatter(np.full(len(finite), xi) + jit, finite,
                       color=color, s=90, alpha=0.85, edgecolors="white", linewidths=0.6,
                       label=arch, zorder=4)
            if finite:
                ax.hlines(np.mean(finite), xi - 0.22, xi + 0.22,
                          color=color, lw=2.8, zorder=5)

        sig_str = "Δ={:+.3f}\np={:.3f} {}".format(
            res["gap"], res["p"], "✓" if res["significant"] else "n.s.")
        ymax = max([v for v in c_vals + l_vals if np.isfinite(v)], default=1.0)
        ax.text(0.5, min(ymax * 1.03, 1.02), sig_str,
                ha="center", va="bottom", fontsize=9,
                color="green" if res["significant"] else "#888")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Canon", "Llama"], fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("Final accuracy" if ax == axes[0] else "")
        ax.set_ylim(-0.05, 1.10)
        ax.grid(axis="y", alpha=0.3, ls="--")
        if ax == axes[0]:
            ax.legend(fontsize=8)

    fig.suptitle("Exp B — Per-task accuracy on hard BFS/SP (max_nodes=120)\n"
                 "Does Canon gain where Llama has headroom?",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "bfs_sp_accuracy_comparison.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"Saved: {path}")


def plot_training_curves_loss(rows):
    """loss/out training curves, Canon vs Llama (mean ± range)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for arch, color in [("Canon", C_RED), ("Llama", C_BLUE)]:
        arch_rows = [r for r in rows if r["arch"] == arch]
        series_list = [r["_loss_series"].reset_index(drop=True)
                       for r in arch_rows if len(r["_loss_series"]) > 0]
        if not series_list:
            continue
        min_len = min(len(s) for s in series_list)
        mat = np.stack([s.values[:min_len] for s in series_list])
        steps = np.arange(min_len)
        mean = mat.mean(axis=0)
        lo   = mat.min(axis=0)
        hi   = mat.max(axis=0)
        ax.plot(steps, mean, color=color, lw=2, label=arch)
        ax.fill_between(steps, lo, hi, color=color, alpha=0.15)

    ax.set_xlabel("Training step (log-frequency samples)")
    ax.set_ylabel("loss/out")
    ax.set_title("Exp B — Loss training curves on hard BFS/SP\n"
                 "(mean ± seed range, 3 seeds each)")
    ax.grid(alpha=0.3, ls="--")
    ax.legend(fontsize=10)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "bfs_sp_training_curves.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"Saved: {path}")


def plot_accuracy_curves(rows):
    """Accuracy training curves for all 4 subtasks, 2×2 panel."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()
    for ax, (key, label, _) in zip(axes, TASK_METRICS):
        for arch, color in [("Canon", C_RED), ("Llama", C_BLUE)]:
            arch_rows = [r for r in rows if r["arch"] == arch]
            series_list = [r[f"_series_{label}"].reset_index(drop=True)
                           for r in arch_rows if len(r[f"_series_{label}"]) > 0]
            if not series_list:
                continue
            min_len = min(len(s) for s in series_list)
            mat = np.stack([s.values[:min_len] for s in series_list])
            steps = np.arange(min_len)
            mean = mat.mean(axis=0)
            lo   = mat.min(axis=0)
            hi   = mat.max(axis=0)
            ax.plot(steps, mean, color=color, lw=2, label=arch)
            ax.fill_between(steps, lo, hi, color=color, alpha=0.15)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Eval checkpoint")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3, ls="--")
        ax.legend(fontsize=8)

    fig.suptitle("Exp B — Accuracy training curves on hard BFS/SP\n"
                 "(mean ± seed range, 3 seeds each)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "bfs_sp_accuracy_curves.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"Saved: {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    api = wandb.Api(api_key=WANDB_API_KEY)

    print("=" * 72)
    print("Exp B — De-ceiling BFS/SP")
    print("=" * 72)

    rows = fetch_all(api)
    results = print_summary(rows)
    print_decision(results)

    plot_loss_comparison(rows, results)
    plot_accuracy_comparison(rows, results)
    plot_training_curves_loss(rows)
    plot_accuracy_curves(rows)

    print("\nDone.")
