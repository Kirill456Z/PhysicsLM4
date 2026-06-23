"""
Exp 9 — Depth ablation at fixed parameter count (Part 2 §4.3).

Three depth/width configs × {Canon-ABCD, Llama} × 2 seeds = 12 runs.
All trained on the standard 8-task synthetic mix, 30M params, 80k steps,
with full dynamics logging (grad_contrib, kurtosis).

Configs (approx. 30M params throughout):
  4L  / ~768D  — few Canon insertion points
  8L  / 512D   — standard (cross-checks existing 30M runs)
  16L / ~360D  — many Canon insertion points

Runs:
  Canon 4L:  depth_ablation_canon_4l_s{1,2}_0.4.{221,222}
  Canon 8L:  depth_ablation_canon_8l_s{1,2}_0.4.{223,224}
  Canon 16L: depth_ablation_canon_16l_s{1,2}_0.4.{219,220}  ← may still be running
  Llama 4L:  depth_ablation_llama_4l_s{1,2}_0.4.{227,228}
  Llama 8L:  depth_ablation_llama_8l_s{1,2}_0.4.{229,230}
  Llama 16L: depth_ablation_llama_16l_s{1,2}_0.4.{225,226}

Decision rule (MISSING_RUNS_AFTER_PIVOT.md §9):
  Canon−Llama loss gap grows with depth (4L < 8L < 16L)
      → gradient-ratio control IS the primary mechanism (causal).
  Gap is depth-independent
      → gradient story is correlational only; lean on §4.2 scope instead.

Metrics fetched per run:
  loss/out                          (aggregate loss, time-series + final)
  grad_contrib/layer_0..layer_{N-1} (gradient contributions, final snapshot)
  outlier_features/kurtosis/layer_0..layer_{N-1}  (kurtosis, final snapshot)
  evals/synthetic/depo_{edges,adj}_list/hop_4/accuracy
  evals/synthetic/bfs_{edges,adj}_list/set_recall
  evals/synthetic/shortest_path_{edges,adj}_list/set_accuracy

Key derived quantities:
  grad_ratio = mean(deep-half layers) / mean(shallow-half layers)  of grad_contrib
  Canon−Llama loss gap at each depth

Plots saved to plots/:
  depth_loss_comparison.png     — final loss by depth, Canon vs Llama + gap trend
  depth_grad_ratio.png          — deep/shallow grad ratio by depth and arch
  depth_depo_accuracy.png       — Depo hop-4 accuracy by depth and arch
  depth_training_curves.png     — loss/out training curves, 3-panel (one per depth)
  depth_kurtosis.png            — mean kurtosis (deep layers) by depth and arch
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb

WANDB_API_KEY = "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"
ENTITY  = "kirill456z"
PROJECT = "physics4llm"
OUT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(OUT_DIR, exist_ok=True)

LOSS_TRAILING = 100   # trailing steps for final-loss average
ACC_TRAILING  = 20    # trailing eval points for accuracy
RNG = np.random.default_rng(42)

C_RED, C_BLUE = "#d6604d", "#2166ac"

# ── Run inventory ──────────────────────────────────────────────────────────────

RUNS = {
    "4L": {
        "Canon": ["depth_ablation_canon_4l_s1_0.4.221", "depth_ablation_canon_4l_s2_0.4.222"],
        "Llama": ["depth_ablation_llama_4l_s1_0.4.227", "depth_ablation_llama_4l_s2_0.4.228"],
    },
    "8L": {
        "Canon": ["depth_ablation_canon_8l_s1_0.4.223", "depth_ablation_canon_8l_s2_0.4.224"],
        "Llama": ["depth_ablation_llama_8l_s1_0.4.229", "depth_ablation_llama_8l_s2_0.4.230"],
    },
    "16L": {
        "Canon": ["depth_ablation_canon_16l_s1_0.4.219", "depth_ablation_canon_16l_s2_0.4.220"],
        "Llama": ["depth_ablation_llama_16l_s1_0.4.225", "depth_ablation_llama_16l_s2_0.4.226"],
    },
}
DEPTHS   = ["4L", "8L", "16L"]
N_LAYERS = {"4L": 4, "8L": 8, "16L": 16}

DEPO_METRICS = [
    "evals/synthetic/depo_edges_list/hop_4/accuracy",
    "evals/synthetic/depo_adj_list/hop_4/accuracy",
]
BFS_METRICS = [
    "evals/synthetic/bfs_edges_list/set_recall",
    "evals/synthetic/bfs_adj_list/set_recall",
]
SP_METRICS = [
    "evals/synthetic/shortest_path_edges_list/set_accuracy",
    "evals/synthetic/shortest_path_adj_list/set_accuracy",
]


# ── Data fetching ──────────────────────────────────────────────────────────────

def _grad_keys(n_layers):
    return [f"grad_contrib/layer_{i}" for i in range(n_layers)]

def _kurtosis_keys(n_layers):
    return [f"outlier_features/kurtosis/layer_{i}" for i in range(n_layers)]

def fetch_run(run_id, depth, api):
    n = N_LAYERS[depth]
    result = {
        "run_id": run_id, "depth": depth,
        "final_loss": float("nan"),
        "grad_ratio": float("nan"),
        "mean_kurtosis_deep": float("nan"),
        "depo_edges": float("nan"), "depo_adj": float("nan"),
        "bfs_edges": float("nan"), "bfs_adj": float("nan"),
        "sp_edges": float("nan"), "sp_adj": float("nan"),
        "_loss_series": pd.Series(dtype=float),
        "_state": "unknown",
        "_step": 0,
    }
    try:
        run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
        result["_state"] = run.state
        summary = run.summary._json_dict

        # current step
        result["_step"] = int(summary.get("global_step", summary.get("_step", 0)))

        # loss series
        h = run.history(samples=500, keys=["loss/out"])
        s = h["loss/out"].dropna() if "loss/out" in h.columns else pd.Series(dtype=float)
        result["_loss_series"] = s.reset_index(drop=True)
        if len(s) >= 5:
            result["final_loss"] = float(s.iloc[-LOSS_TRAILING:].mean())

        # grad_contrib — use summary (final snapshot)
        gkeys = _grad_keys(n)
        gvals = [summary.get(k, float("nan")) for k in gkeys]
        finite_g = [v for v in gvals if np.isfinite(v)]
        if len(finite_g) >= n:
            half = n // 2
            shallow = np.mean(gvals[:half])
            deep    = np.mean(gvals[half:])
            result["grad_ratio"] = float(deep / shallow) if shallow > 0 else float("nan")

        # kurtosis — deep-half mean
        kkeys = _kurtosis_keys(n)
        kvals = [summary.get(k, float("nan")) for k in kkeys]
        half = n // 2
        deep_k = [v for v in kvals[half:] if np.isfinite(v)]
        if deep_k:
            result["mean_kurtosis_deep"] = float(np.mean(deep_k))

        # eval accuracies — trailing mean from history
        all_acc_keys = DEPO_METRICS + BFS_METRICS + SP_METRICS
        ha = run.history(samples=500, keys=all_acc_keys)
        def _acc(key):
            s = ha[key].dropna() if key in ha.columns else pd.Series(dtype=float)
            return float(s.iloc[-ACC_TRAILING:].mean()) if len(s) >= 3 else float("nan")
        result["depo_edges"] = _acc(DEPO_METRICS[0])
        result["depo_adj"]   = _acc(DEPO_METRICS[1])
        result["bfs_edges"]  = _acc(BFS_METRICS[0])
        result["bfs_adj"]    = _acc(BFS_METRICS[1])
        result["sp_edges"]   = _acc(SP_METRICS[0])
        result["sp_adj"]     = _acc(SP_METRICS[1])

    except Exception as exc:
        print(f"  [warn] {run_id}: {exc}")
    return result


def fetch_all(api):
    rows = []
    for depth in DEPTHS:
        for arch in ["Canon", "Llama"]:
            print(f"\n{'─'*60}\nFetching {depth} {arch} …")
            for rid in RUNS[depth][arch]:
                r = fetch_run(rid, depth, api)
                r["arch"] = arch
                step_str = f"step={r['_step']}" if r["_step"] else ""
                state_str = f"[{r['_state']}]" if r["_state"] != "finished" else ""
                loss_str  = f"{r['final_loss']:.4f}" if np.isfinite(r["final_loss"]) else "NaN"
                gr_str    = f"{r['grad_ratio']:.3f}" if np.isfinite(r["grad_ratio"]) else "NaN"
                depo_str  = f"{np.nanmean([r['depo_edges'], r['depo_adj']]):.3f}" if any(np.isfinite(r[k]) for k in ["depo_edges","depo_adj"]) else "NaN"
                print(f"  {arch:6s} {rid:45s} {state_str:12s} {step_str:12s} "
                      f"loss={loss_str}  grad_ratio={gr_str}  depo={depo_str}")
                rows.append(r)
    return rows


# ── Aggregation ────────────────────────────────────────────────────────────────

def aggregate(rows):
    """Return {depth: {arch: {metric: (mean, std)}}} averaged over seeds."""
    agg = {}
    for depth in DEPTHS:
        agg[depth] = {}
        for arch in ["Canon", "Llama"]:
            subset = [r for r in rows if r["depth"] == depth and r["arch"] == arch]
            metrics = ["final_loss", "grad_ratio", "mean_kurtosis_deep",
                       "depo_edges", "depo_adj", "bfs_edges", "bfs_adj", "sp_edges", "sp_adj"]
            agg[depth][arch] = {}
            for m in metrics:
                vals = [r[m] for r in subset if np.isfinite(r[m])]
                agg[depth][arch][m] = (
                    float(np.mean(vals)) if vals else float("nan"),
                    float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan"),
                    vals,
                )
    return agg


# ── Print summary ──────────────────────────────────────────────────────────────

def print_summary(rows, agg):
    print(f"\n{'='*72}")
    print("Exp 9 — Depth ablation: Canon vs Llama at 4L / 8L / 16L")
    print(f"{'='*72}")

    print(f"\n{'─'*72}")
    print(f"  {'Depth':>6}  {'Arch':>6}  {'n':>2}  {'loss/out':>10}  {'grad_ratio':>11}  "
          f"{'kurtosis_deep':>14}  {'depo_mean':>10}")
    print(f"{'─'*72}")
    for depth in DEPTHS:
        for arch in ["Canon", "Llama"]:
            a = agg[depth][arch]
            n = len(a["final_loss"][2])
            loss_m, loss_s = a["final_loss"][:2]
            gr_m,   gr_s   = a["grad_ratio"][:2]
            kt_m,   kt_s   = a["mean_kurtosis_deep"][:2]
            de_e = a["depo_edges"][0]; de_a = a["depo_adj"][0]
            depo_m = float(np.nanmean([de_e, de_a]))
            print(f"  {depth:>6}  {arch:>6}  {n:>2}  "
                  f"{loss_m:>10.4f}  {gr_m:>11.4f}  {kt_m:>14.4f}  {depo_m:>10.4f}")

    print(f"\n{'─'*72}")
    print("Canon−Llama gap by depth:")
    gaps = []
    for depth in DEPTHS:
        lc = agg[depth]["Canon"]["final_loss"][0]
        ll = agg[depth]["Llama"]["final_loss"][0]
        gap = ll - lc   # positive = Canon better (lower loss)
        gr_c = agg[depth]["Canon"]["grad_ratio"][0]
        gr_l = agg[depth]["Llama"]["grad_ratio"][0]
        de_c = np.nanmean([agg[depth]["Canon"]["depo_edges"][0],
                           agg[depth]["Canon"]["depo_adj"][0]])
        de_l = np.nanmean([agg[depth]["Llama"]["depo_edges"][0],
                           agg[depth]["Llama"]["depo_adj"][0]])
        gaps.append(gap)
        print(f"  {depth:>4}  loss gap={gap:+.4f}  "
              f"grad_ratio Canon={gr_c:.3f}  Llama={gr_l:.3f}  "
              f"Depo Δ={de_c-de_l:+.3f}")

    print(f"\n{'─'*72}")
    print("DECISION:")
    finite_gaps = [(d, g) for d, g in zip(DEPTHS, gaps) if np.isfinite(g)]
    if len(finite_gaps) >= 2:
        gap_vals = [g for _, g in finite_gaps]
        monotone_up   = all(gap_vals[i] < gap_vals[i+1] for i in range(len(gap_vals)-1))
        monotone_down = all(gap_vals[i] > gap_vals[i+1] for i in range(len(gap_vals)-1))
        spread = max(gap_vals) - min(gap_vals)
        if monotone_up and spread > 0.005:
            print("  GROWS WITH DEPTH → gradient-ratio control is the causal mechanism.")
            print("  §4.3 causal claim is EARNED.")
        elif spread < 0.005:
            print("  FLAT across depth → gap is depth-independent.")
            print("  Gradient story is CORRELATIONAL only; lean on §4.2 scope.")
        else:
            print("  NON-MONOTONE — interpret with caution.")
            for d, g in finite_gaps:
                print(f"    {d}: gap={g:+.4f}")
            if not monotone_up and not monotone_down:
                print("  The depth effect may be confounded by architecture differences.")
    else:
        print("  Insufficient data to decide (16L Canon still running?).")

    # warn about running runs
    running = [r for r in rows if r["_state"] not in ("finished",)]
    if running:
        print(f"\n  NOTE: {len(running)} run(s) still in progress:")
        for r in running:
            print(f"    {r['run_id']}  [{r['_state']}]  step={r['_step']}")
        print("  Results for those runs use partial data.")


# ── Plots ──────────────────────────────────────────────────────────────────────

def _bar_positions(depths, n_groups=2, width=0.35):
    x = np.arange(len(depths))
    return x - width/2, x + width/2, x, width


def plot_loss_comparison(agg):
    """3-cluster bar chart: final loss by depth, Canon vs Llama + gap line."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: absolute loss
    ax = axes[0]
    x = np.arange(len(DEPTHS))
    w = 0.35
    for i, (arch, color, off) in enumerate([("Canon", C_RED, -w/2), ("Llama", C_BLUE, w/2)]):
        means = [agg[d][arch]["final_loss"][0] for d in DEPTHS]
        stds  = [agg[d][arch]["final_loss"][1] for d in DEPTHS]
        valid = [np.isfinite(m) for m in means]
        xv = x[valid] + off
        mv = [m for m, v in zip(means, valid) if v]
        sv = [s if np.isfinite(s) else 0 for s, v in zip(stds, valid) if v]
        ax.bar(xv, mv, w, color=color, alpha=0.80, label=arch,
               yerr=sv, capsize=4, error_kw=dict(lw=1.5))
    ax.set_xticks(x); ax.set_xticklabels(DEPTHS)
    ax.set_ylabel("Final loss/out (trailing-100 mean)")
    ax.set_title("Final loss by depth\n(mean ± std across 2 seeds)")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3, ls="--")

    # Right: Canon−Llama gap
    ax = axes[1]
    gaps = []
    for depth in DEPTHS:
        lc = agg[depth]["Canon"]["final_loss"][0]
        ll = agg[depth]["Llama"]["final_loss"][0]
        gaps.append(ll - lc)
    finite_mask = [np.isfinite(g) for g in gaps]
    xv = x[finite_mask]; gv = [g for g, f in zip(gaps, finite_mask) if f]
    colors = ["#2ca02c" if g > 0 else "#d62728" for g in gv]
    ax.bar(xv, gv, 0.55, color=colors, alpha=0.8)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(0.015, color="gray", ls="--", lw=1, label="Exp-A noise (~0.015)")
    for xi, g in zip(xv, gv):
        ax.text(xi, g + (0.001 if g >= 0 else -0.003),
                f"{g:+.4f}", ha="center", va="bottom" if g >= 0 else "top", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(DEPTHS)
    ax.set_ylabel("Loss gap (Llama − Canon)")
    ax.set_title("Canon−Llama gap by depth\n(positive = Canon better)")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3, ls="--")

    fig.suptitle("Exp 9 — Does Canon's advantage grow with depth?",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "depth_loss_comparison.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"Saved: {path}")


def plot_grad_ratio(agg):
    """Gradient ratio (deep/shallow) by depth and arch."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(DEPTHS)); w = 0.35
    for arch, color, off in [("Canon", C_RED, -w/2), ("Llama", C_BLUE, w/2)]:
        means = [agg[d][arch]["grad_ratio"][0] for d in DEPTHS]
        stds  = [agg[d][arch]["grad_ratio"][1] for d in DEPTHS]
        valid = [np.isfinite(m) for m in means]
        xv = x[valid] + off
        mv = [m for m, v in zip(means, valid) if v]
        sv = [s if np.isfinite(s) else 0 for s, v in zip(stds, valid) if v]
        ax.bar(xv, mv, w, color=color, alpha=0.80, label=arch,
               yerr=sv, capsize=4, error_kw=dict(lw=1.5))
    ax.axhline(1.0, color="black", ls="--", lw=1, label="ratio=1 (uniform)")
    ax.set_xticks(x); ax.set_xticklabels(DEPTHS)
    ax.set_ylabel("Gradient ratio (deep-half / shallow-half)")
    ax.set_title("Exp 9 — Deep/shallow gradient ratio by depth\n"
                 "(>1 = deep layers dominate; Canon expected to suppress this)")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3, ls="--")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "depth_grad_ratio.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"Saved: {path}")


def plot_depo_accuracy(rows, agg):
    """Depo hop-4 accuracy strip plots by depth and arch."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (enc, field) in zip(axes, [("edges_list", "depo_edges"), ("adj_list", "depo_adj")]):
        for di, depth in enumerate(DEPTHS):
            for arch, color, off in [("Canon", C_RED, -0.25), ("Llama", C_BLUE, 0.25)]:
                xi = di * 3
                vals = [r[field] for r in rows
                        if r["depth"] == depth and r["arch"] == arch and np.isfinite(r[field])]
                jit = RNG.uniform(-0.1, 0.1, len(vals))
                ax.scatter(np.full(len(vals), xi + off) + jit, vals,
                           color=color, s=70, alpha=0.85, edgecolors="white", linewidths=0.5,
                           label=arch if di == 0 else "")
                if vals:
                    ax.hlines(np.mean(vals), xi + off - 0.18, xi + off + 0.18,
                              color=color, lw=2.5, zorder=5)
        ax.set_xticks([di * 3 for di in range(len(DEPTHS))])
        ax.set_xticklabels(DEPTHS)
        ax.set_ylabel("Depo hop-4 accuracy")
        ax.set_title(f"Depo ({enc})")
        ax.set_ylim(-0.05, 1.10)
        ax.grid(axis="y", alpha=0.3, ls="--")
        if ax == axes[0]:
            ax.legend(fontsize=9)
    fig.suptitle("Exp 9 — Depo hop-4 accuracy by depth\n"
                 "(does Canon's Depo advantage grow with depth?)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "depth_depo_accuracy.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"Saved: {path}")


def plot_training_curves(rows):
    """3-panel loss curves: one per depth, Canon vs Llama (mean ± range)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, depth in zip(axes, DEPTHS):
        for arch, color in [("Canon", C_RED), ("Llama", C_BLUE)]:
            series_list = [r["_loss_series"]
                           for r in rows if r["depth"] == depth and r["arch"] == arch
                           and len(r["_loss_series"]) > 0]
            if not series_list:
                continue
            min_len = min(len(s) for s in series_list)
            mat = np.stack([s.values[:min_len] for s in series_list])
            steps = np.arange(min_len)
            mean = mat.mean(axis=0)
            lo, hi = mat.min(axis=0), mat.max(axis=0)
            ax.plot(steps, mean, color=color, lw=2, label=arch)
            ax.fill_between(steps, lo, hi, color=color, alpha=0.15)
        ax.set_title(f"{depth}")
        ax.set_xlabel("Training step (sample index)")
        ax.set_ylabel("loss/out" if ax == axes[0] else "")
        ax.grid(alpha=0.3, ls="--")
        ax.legend(fontsize=9)
    fig.suptitle("Exp 9 — Loss training curves by depth (mean ± seed range)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "depth_training_curves.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"Saved: {path}")


def plot_kurtosis(agg):
    """Mean deep-half kurtosis by depth and arch."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(DEPTHS)); w = 0.35
    for arch, color, off in [("Canon", C_RED, -w/2), ("Llama", C_BLUE, w/2)]:
        means = [agg[d][arch]["mean_kurtosis_deep"][0] for d in DEPTHS]
        stds  = [agg[d][arch]["mean_kurtosis_deep"][1] for d in DEPTHS]
        valid = [np.isfinite(m) for m in means]
        xv = x[valid] + off
        mv = [m for m, v in zip(means, valid) if v]
        sv = [s if np.isfinite(s) else 0 for s, v in zip(stds, valid) if v]
        ax.bar(xv, mv, w, color=color, alpha=0.80, label=arch,
               yerr=sv, capsize=4, error_kw=dict(lw=1.5))
    ax.set_xticks(x); ax.set_xticklabels(DEPTHS)
    ax.set_ylabel("Mean kurtosis (deep-half layers)")
    ax.set_title("Exp 9 — Activation kurtosis (deep layers) by depth\n"
                 "(descriptive; Canon expected to raise kurtosis on synthetic)")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3, ls="--")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "depth_kurtosis.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"Saved: {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    api = wandb.Api(api_key=WANDB_API_KEY)

    print("=" * 72)
    print("Exp 9 — Depth ablation (4L / 8L / 16L)")
    print("NOTE: Canon 16L runs may still be in progress — partial data used.")
    print("=" * 72)

    rows = fetch_all(api)
    agg  = aggregate(rows)

    print_summary(rows, agg)

    plot_loss_comparison(agg)
    plot_grad_ratio(agg)
    plot_depo_accuracy(rows, agg)
    plot_training_curves(rows)
    plot_kurtosis(agg)

    print("\nDone.")
