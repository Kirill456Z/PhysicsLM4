"""
Exp 9 (repowered) — Depth ablation with 5 seeds at 16L.

Pools the original 2-seed runs (14_depth_ablation) with 3 new seeds from
depth_ablation_more_seeds/ (16L only) to resolve the non-monotone 16L result.

Seed counts after pooling:
  4L:  2 seeds per arm  (unchanged — result was already clear)
  8L:  2 seeds per arm  (unchanged — result was already clear)
  16L: 5 seeds per arm  (2 old + 3 new — this is the cell under investigation)

Decision rule:
  Permutation test at 16L (20k resamples on mean-loss difference):
    p < 0.05 and gap > 0  → Canon still better at 16L → gap is non-monotone for real
    p < 0.05 and gap < 0  → Llama better at 16L → reversal confirmed
    p ≥ 0.05              → inconclusive (Depo-variance swamps the signal at 16L too)

  Depth trend: if 4L gap > 8L gap > 16L gap with all gaps > 0 and 16L p < 0.05
    → monotone, causal gradient story is partially supported.

Original 2-seed runs (14_depth_ablation):
  Canon 4L:  depth_ablation_canon_4l_s{1,2}_0.4.{221,222}
  Canon 8L:  depth_ablation_canon_8l_s{1,2}_0.4.{223,224}
  Canon 16L: depth_ablation_canon_16l_s{1,2}_0.4.{219,220}
  Llama 4L:  depth_ablation_llama_4l_s{1,2}_0.4.{227,228}
  Llama 8L:  depth_ablation_llama_8l_s{1,2}_0.4.{229,230}
  Llama 16L: depth_ablation_llama_16l_s{1,2}_0.4.{225,226}

New 3-seed runs (depth_ablation_more_seeds — 16L only):
  Canon 16L: discovered by display-name regex "depth_ablation_more_seeds.*canon.*16l"
  Llama 16L: discovered by display-name regex "depth_ablation_more_seeds.*llama.*16l"
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import wandb

WANDB_API_KEY = "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"
ENTITY  = "kirill456z"
PROJECT = "physics4llm"
OUT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(OUT_DIR, exist_ok=True)

LOSS_TRAILING = 100
ACC_TRAILING  = 20
N_PERM        = 20_000
RNG           = np.random.default_rng(42)

C_CANON, C_LLAMA = "#d6604d", "#2166ac"

DEPTHS   = ["4L", "8L", "16L"]
N_LAYERS = {"4L": 4, "8L": 8, "16L": 16}

# Known run IDs from the original 2-seed experiment (14_depth_ablation).
# New 16L seeds are discovered dynamically via display-name search below.
ORIGINAL_RUNS = {
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

DEPO_METRICS = [
    "evals/synthetic/depo_edges_list/hop_4/accuracy",
    "evals/synthetic/depo_adj_list/hop_4/accuracy",
]


# ── Run discovery ──────────────────────────────────────────────────────────────

def discover_new_16l_runs(api):
    """
    Find the 3 new Canon and 3 new Llama 16L run IDs from depth_ablation_more_seeds.
    Returns {"Canon": [...], "Llama": [...]}.
    """
    result = {}
    for arch, pattern in [
        ("Canon", "depth_ablation_more_seeds.*canon.*16l"),
        ("Llama", "depth_ablation_more_seeds.*llama.*16l"),
    ]:
        print(f"\nSearching for new 16L {arch} runs (pattern: {pattern!r}) …")
        filters = {"displayName": {"$regex": pattern}, "state": "finished"}
        runs = list(api.runs(f"{ENTITY}/{PROJECT}", filters=filters))
        ids = []
        for r in runs:
            print(f"  found: {r.name:60s}  id={r.id}")
            ids.append(r.id)
        if not ids:
            print(f"  [warn] no finished runs matched — trying without state filter")
            runs = list(api.runs(f"{ENTITY}/{PROJECT}",
                                 filters={"displayName": {"$regex": pattern}}))
            for r in runs:
                print(f"  found (any state): {r.name:60s}  id={r.id}  state={r.state}")
                ids.append(r.id)
        result[arch] = ids
    return result


# ── Data fetching ──────────────────────────────────────────────────────────────

def _grad_keys(n): return [f"grad_contrib/layer_{i}" for i in range(n)]
def _kurt_keys(n): return [f"outlier_features/kurtosis/layer_{i}" for i in range(n)]


def fetch_run(run_id, depth, api):
    n = N_LAYERS[depth]
    rec = {
        "run_id": run_id, "depth": depth, "arch": None,
        "final_loss": float("nan"), "grad_ratio": float("nan"),
        "mean_kurtosis_deep": float("nan"),
        "depo_edges": float("nan"), "depo_adj": float("nan"),
        "_loss_series": [],
        "_state": "unknown", "_step": 0,
    }
    try:
        run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
        rec["_state"] = run.state
        summary = run.summary._json_dict
        rec["_step"] = int(summary.get("global_step", summary.get("_step", 0)))

        # loss series
        h = run.history(samples=500, keys=["loss/out"])
        col = h["loss/out"].dropna() if "loss/out" in h.columns else []
        if len(col) >= 5:
            rec["final_loss"] = float(col.iloc[-LOSS_TRAILING:].mean())
            rec["_loss_series"] = col.values.tolist()

        # gradient ratio (deep/shallow)
        gkeys = _grad_keys(n)
        gvals = [summary.get(k, float("nan")) for k in gkeys]
        finite_g = [v for v in gvals if np.isfinite(v)]
        if len(finite_g) >= n:
            half = n // 2
            shallow, deep = np.mean(gvals[:half]), np.mean(gvals[half:])
            rec["grad_ratio"] = float(deep / shallow) if shallow > 0 else float("nan")

        # kurtosis (deep half mean)
        kkeys = _kurt_keys(n)
        kvals = [summary.get(k, float("nan")) for k in kkeys]
        half = n // 2
        dk = [v for v in kvals[half:] if np.isfinite(v)]
        if dk:
            rec["mean_kurtosis_deep"] = float(np.mean(dk))

        # Depo accuracy
        ha = run.history(samples=500, keys=DEPO_METRICS)
        def _acc(key):
            s = ha[key].dropna() if key in ha.columns else []
            return float(s.iloc[-ACC_TRAILING:].mean()) if len(s) >= 3 else float("nan")
        rec["depo_edges"] = _acc(DEPO_METRICS[0])
        rec["depo_adj"]   = _acc(DEPO_METRICS[1])

    except Exception as exc:
        print(f"  [warn] {run_id}: {exc}")
    return rec


def fetch_all(api, new_16l):
    """
    Fetch all original runs + new 16L runs.
    Returns flat list of records with 'arch' and 'depth' set.
    """
    rows = []

    for depth in DEPTHS:
        for arch in ["Canon", "Llama"]:
            ids = list(ORIGINAL_RUNS[depth][arch])
            if depth == "16L":
                ids += new_16l.get(arch, [])
            tag = f"{depth} {arch} ({len(ids)} seeds)"
            print(f"\n{'─'*60}\nFetching {tag} …")
            for rid in ids:
                rec = fetch_run(rid, depth, api)
                rec["arch"] = arch
                loss_s = f"{rec['final_loss']:.4f}" if np.isfinite(rec["final_loss"]) else "NaN"
                gr_s   = f"{rec['grad_ratio']:.3f}" if np.isfinite(rec["grad_ratio"]) else "NaN"
                state_s = "" if rec["_state"] == "finished" else f"[{rec['_state']}]"
                print(f"  {arch:6s} {rid:50s} {state_s:12s} loss={loss_s}  grad_ratio={gr_s}")
                rows.append(rec)
    return rows


# ── Aggregation ────────────────────────────────────────────────────────────────

def aggregate(rows):
    """Return {depth: {arch: {metric: (mean, std, [values])}}}."""
    agg = {}
    metrics = ["final_loss", "grad_ratio", "mean_kurtosis_deep", "depo_edges", "depo_adj"]
    for depth in DEPTHS:
        agg[depth] = {}
        for arch in ["Canon", "Llama"]:
            subset = [r for r in rows if r["depth"] == depth and r["arch"] == arch]
            agg[depth][arch] = {}
            for m in metrics:
                vals = [r[m] for r in subset if np.isfinite(r[m])]
                agg[depth][arch][m] = (
                    float(np.mean(vals)) if vals else float("nan"),
                    float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan"),
                    vals,
                )
    return agg


# ── Permutation test ───────────────────────────────────────────────────────────

def permutation_test(a, b, n_perm=N_PERM):
    """
    Two-sample permutation test on the difference of means.
    Returns (observed_gap, p_value).
    observed_gap = mean(b) - mean(a)  (positive = b has higher loss = a is better).
    """
    a, b = np.array(a), np.array(b)
    if len(a) == 0 or len(b) == 0:
        return float("nan"), float("nan")
    obs = np.mean(b) - np.mean(a)
    combined = np.concatenate([a, b])
    na = len(a)
    count = 0
    for _ in range(n_perm):
        RNG.shuffle(combined)
        count += (np.mean(combined[na:]) - np.mean(combined[:na]) >= obs)
    p = count / n_perm
    return float(obs), float(p)


# ── Print summary ──────────────────────────────────────────────────────────────

def print_summary(rows, agg):
    print(f"\n{'='*72}")
    print("Exp 9 (repowered) — Depth ablation: Canon vs Llama at 4L / 8L / 16L")
    print("Seeds: 4L=2, 8L=2, 16L=5 (2 original + 3 new)")
    print(f"{'='*72}")

    print(f"\n{'Depth':>6}  {'Arch':>6}  {'n':>2}  {'loss/out':>10}±{'std':>6}  "
          f"{'grad_ratio':>10}  {'depo_mean':>10}")
    print(f"{'─'*72}")
    for depth in DEPTHS:
        for arch in ["Canon", "Llama"]:
            a = agg[depth][arch]
            n = len(a["final_loss"][2])
            lm, ls = a["final_loss"][:2]
            gm      = a["grad_ratio"][0]
            de_m    = float(np.nanmean([a["depo_edges"][0], a["depo_adj"][0]]))
            ls_s = f"{ls:.4f}" if np.isfinite(ls) else "  NaN"
            print(f"  {depth:>4}  {arch:>6}  {n:>2}  {lm:>10.4f}±{ls_s:<6}  "
                  f"{gm:>10.4f}  {de_m:>10.4f}")

    print(f"\n{'─'*72}")
    print("Canon−Llama gap and permutation test by depth:")
    gaps = []
    for depth in DEPTHS:
        canon_vals = agg[depth]["Canon"]["final_loss"][2]
        llama_vals = agg[depth]["Llama"]["final_loss"][2]
        gap, p = permutation_test(canon_vals, llama_vals)
        gaps.append((depth, gap, p, len(canon_vals), len(llama_vals)))
        gc = agg[depth]["Canon"]["grad_ratio"][0]
        gl = agg[depth]["Llama"]["grad_ratio"][0]
        sig = "**" if p < 0.05 else ("*" if p < 0.10 else "ns")
        print(f"  {depth:>4}  gap={gap:+.4f}  p={p:.4f} {sig:<2}  "
              f"n_canon={len(canon_vals)} n_llama={len(llama_vals)}  "
              f"grad_ratio C={gc:.3f} L={gl:.3f}")

    print(f"\n{'─'*72}")
    print("DECISION:")
    gap_vals  = [g for _, g, _, _, _ in gaps]
    p_vals    = [p for _, _, p, _, _ in gaps]
    finite    = [np.isfinite(g) and np.isfinite(p) for g, p in zip(gap_vals, p_vals)]

    # 16L verdict
    idx_16 = DEPTHS.index("16L")
    g16, p16 = gap_vals[idx_16], p_vals[idx_16]
    if not finite[idx_16]:
        print("  16L: insufficient data.")
    elif p16 < 0.05 and g16 < 0:
        print(f"  16L: REVERSAL CONFIRMED  gap={g16:+.4f}  p={p16:.4f}")
        print("  Canon is significantly WORSE at 16L. The depth trend is non-monotone.")
        print("  §4.3: mechanism is CORRELATIONAL only — depth gap not monotone.")
    elif p16 < 0.05 and g16 > 0:
        print(f"  16L: Canon still better at 16L  gap={g16:+.4f}  p={p16:.4f}")
        # check monotone
        valid_gaps = [(d, g) for (d, g, p, _, _), f in zip(gaps, finite) if f]
        gv = [g for _, g in valid_gaps]
        if all(gv[i] >= gv[i+1] for i in range(len(gv)-1)) and all(g > 0 for g in gv):
            print("  Gap is monotonically DECREASING (4L > 8L > 16L) and all positive.")
            print("  §4.3: depth trend supports the gradient-ratio CAUSAL story.")
        else:
            print("  Gap not monotone across all depths — treat causality with caution.")
    else:
        print(f"  16L: INCONCLUSIVE  gap={g16:+.4f}  p={p16:.4f}")
        print("  Depo variance still dominates at 16L even with 5 seeds.")
        print("  §4.3: mechanism stays CORRELATIONAL.")

    # warn on non-finished runs
    not_done = [r for r in rows if r["_state"] != "finished"]
    if not_done:
        print(f"\n  NOTE: {len(not_done)} run(s) not yet finished:")
        for r in not_done:
            print(f"    {r['run_id']}  [{r['_state']}]  step={r['_step']}")


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_loss_comparison(rows, agg):
    """Bar chart (mean ± std) + Canon−Llama gap, with individual seed scatter."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: absolute loss with individual points
    ax = axes[0]
    x = np.arange(len(DEPTHS))
    w = 0.35
    for arch, color, off in [("Canon", C_CANON, -w/2), ("Llama", C_LLAMA, w/2)]:
        means = [agg[d][arch]["final_loss"][0] for d in DEPTHS]
        stds  = [agg[d][arch]["final_loss"][1] for d in DEPTHS]
        ax.bar(x + off, means, w, color=color, alpha=0.75, label=arch,
               yerr=[s if np.isfinite(s) else 0 for s in stds],
               capsize=4, error_kw=dict(lw=1.5))
        # individual seed points
        for di, depth in enumerate(DEPTHS):
            vals = agg[depth][arch]["final_loss"][2]
            jit  = RNG.uniform(-0.08, 0.08, len(vals))
            ax.scatter(di + off + jit, vals, color=color, s=40, zorder=5,
                       edgecolors="white", linewidths=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}\n(n={'5' if d=='16L' else '2'})" for d in DEPTHS])
    ax.set_ylabel("Final loss/out (trailing-100 mean)")
    ax.set_title("Final loss by depth — Canon vs Llama\n(bar=mean, error bar=±std, dots=individual seeds)")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3, ls="--")

    # Right panel: Canon−Llama gap with p-value annotation
    ax = axes[1]
    gaps, pvals = [], []
    for depth in DEPTHS:
        cvals = agg[depth]["Canon"]["final_loss"][2]
        lvals = agg[depth]["Llama"]["final_loss"][2]
        g, p  = permutation_test(cvals, lvals)
        gaps.append(g); pvals.append(p)

    colors = []
    for g, p in zip(gaps, pvals):
        if not np.isfinite(g):
            colors.append("gray")
        elif p < 0.05:
            colors.append("#2ca02c" if g > 0 else "#d62728")
        else:
            colors.append("#aec7e8")  # light blue = not significant

    ax.bar(x, gaps, 0.55, color=colors, alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(0.015, color="gray", ls="--", lw=1, label="Exp-A noise floor (~0.015)")
    ax.axhline(-0.015, color="gray", ls="--", lw=1)
    for xi, g, p in zip(x, gaps, pvals):
        if not np.isfinite(g):
            continue
        sig = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
        label = f"{g:+.4f}\n{sig} (p={p:.3f})"
        va = "bottom" if g >= 0 else "top"
        yt = g + (0.002 if g >= 0 else -0.002)
        ax.text(xi, yt, label, ha="center", va=va, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}\n(n={'5' if d=='16L' else '2'})" for d in DEPTHS])
    ax.set_ylabel("Loss gap (Llama − Canon)\npositive = Canon better")
    ax.set_title("Canon−Llama gap by depth\n(green/red = significant p<0.05; blue = ns)")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3, ls="--")

    fig.suptitle("Exp 9 (repowered) — Does Canon's advantage grow with depth?",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "depth_loss_comparison_repowered.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"Saved: {path}")


def plot_grad_ratio(agg):
    """Gradient ratio (deep/shallow) bar chart, colour-coded by depth and arch."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(DEPTHS)); w = 0.35
    for arch, color, off in [("Canon", C_CANON, -w/2), ("Llama", C_LLAMA, w/2)]:
        means = [agg[d][arch]["grad_ratio"][0] for d in DEPTHS]
        stds  = [agg[d][arch]["grad_ratio"][1] for d in DEPTHS]
        ax.bar(x + off, means, w, color=color, alpha=0.80, label=arch,
               yerr=[s if np.isfinite(s) else 0 for s in stds],
               capsize=4, error_kw=dict(lw=1.5))
        # scatter
        for di, depth in enumerate(DEPTHS):
            vals = agg[depth][arch]["grad_ratio"][2]
            jit  = RNG.uniform(-0.08, 0.08, len(vals))
            ax.scatter(di + off + jit, vals, color=color, s=40, zorder=5,
                       edgecolors="white", linewidths=0.5)
    ax.axhline(1.0, color="black", ls="--", lw=1, label="ratio = 1 (uniform)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}\n(n={'5' if d=='16L' else '2'})" for d in DEPTHS])
    ax.set_ylabel("Gradient ratio (deep-half / shallow-half)")
    ax.set_title("Exp 9 (repowered) — Deep/shallow gradient ratio by depth\n"
                 "(Canon expected to suppress deep-layer dominance at every depth)")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3, ls="--")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "depth_grad_ratio_repowered.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"Saved: {path}")


def plot_depo_accuracy(rows, agg):
    """Depo hop-4 accuracy strip plots — edges and adj side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (enc, field) in zip(axes, [("edges_list", "depo_edges"), ("adj_list", "depo_adj")]):
        for di, depth in enumerate(DEPTHS):
            xi = di * 3
            for arch, color, off in [("Canon", C_CANON, -0.25), ("Llama", C_LLAMA, 0.25)]:
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
        ax.set_xticklabels([f"{d}\n(n={'5' if d=='16L' else '2'})" for d in DEPTHS])
        ax.set_ylabel("Depo hop-4 accuracy")
        ax.set_title(f"Depo ({enc})")
        ax.set_ylim(-0.05, 1.10)
        ax.grid(axis="y", alpha=0.3, ls="--")
        if ax == axes[0]:
            ax.legend(fontsize=9)
    fig.suptitle("Exp 9 (repowered) — Depo hop-4 accuracy by depth\n"
                 "(horizontal bar = group mean; individual seeds shown as dots)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "depth_depo_accuracy_repowered.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"Saved: {path}")


def plot_training_curves(rows):
    """3-panel training curves (loss/out): one per depth, Canon vs Llama."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, depth in zip(axes, DEPTHS):
        for arch, color in [("Canon", C_CANON), ("Llama", C_LLAMA)]:
            series = [np.array(r["_loss_series"])
                      for r in rows if r["depth"] == depth and r["arch"] == arch
                      and len(r["_loss_series"]) > 0]
            if not series:
                continue
            min_len = min(len(s) for s in series)
            mat = np.stack([s[:min_len] for s in series])
            steps = np.arange(min_len)
            mean = mat.mean(axis=0)
            lo, hi = mat.min(axis=0), mat.max(axis=0)
            n = len(series)
            ax.plot(steps, mean, color=color, lw=2, label=f"{arch} (n={n})")
            ax.fill_between(steps, lo, hi, color=color, alpha=0.15)
        ax.set_title(f"{depth}")
        ax.set_xlabel("Step (sample index)")
        ax.set_ylabel("loss/out" if depth == "4L" else "")
        ax.grid(alpha=0.3, ls="--")
        ax.legend(fontsize=9)
    fig.suptitle("Exp 9 (repowered) — Loss training curves (mean ± seed range)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "depth_training_curves_repowered.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"Saved: {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    api = wandb.Api(api_key=WANDB_API_KEY)

    print("=" * 72)
    print("Exp 9 (repowered) — Depth ablation with 5 seeds at 16L")
    print(f"Permutation test: {N_PERM:,} resamples, RNG seed 42")
    print("=" * 72)

    new_16l = discover_new_16l_runs(api)
    print(f"\nNew 16L seeds found — Canon: {len(new_16l.get('Canon', []))}, "
          f"Llama: {len(new_16l.get('Llama', []))}")

    rows = fetch_all(api, new_16l)
    agg  = aggregate(rows)

    print_summary(rows, agg)

    plot_loss_comparison(rows, agg)
    plot_grad_ratio(agg)
    plot_depo_accuracy(rows, agg)
    plot_training_curves(rows)

    print("\nDone.")
