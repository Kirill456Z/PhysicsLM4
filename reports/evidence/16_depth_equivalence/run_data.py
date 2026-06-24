"""
Depth equivalence experiment — Canon-ABCD vs Llama depth curves at fixed params.

Hypothesis: Canon-ABCD NL adds effective logical depth equivalent to some extra
Llama layers (each block gains 4 conv ops on top of attn+FFN). If true, Canon-NL
should match the performance of Llama-ML for some M > N, and the "effective depth"
offset tells us how many Llama layers the convolutions are worth.

All configs use dim=448 / 7 heads to keep params ~24-29M (same ballpark as the
original depth_ablation experiment: 4L/768D~28.7M, 8L/512D~25.4M, 16L/384D~28.5M).

New runs (llama_depth_equivalence/, discovered by name pattern):
  Canon 4L/448D  — canon_4l_s{1,2,3}
  Canon 10L/448D — canon_10l_s{1,2,3}
  Llama 10L/448D — llama_10l_s{1,2,3}
  Llama 12L/448D — llama_12l_s{1,2,3}

Existing runs reused from depth_ablation/ (14_depth_ablation):
  Canon 4L/768D  — depth_ablation_canon_4l_s{1,2}_0.4.{221,222}
  Canon 8L/512D  — depth_ablation_canon_8l_s{1,2}_0.4.{223,224}
  Canon 16L/384D — depth_ablation_canon_16l_s{1,2}_0.4.{219,220}
               + depth_ablation_more_seeds_canon_16l_s{3,4,5}_0.4.{235,236,237}
  Llama 4L/768D  — depth_ablation_llama_4l_s{1,2}_0.4.{227,228}
  Llama 8L/512D  — depth_ablation_llama_8l_s{1,2}_0.4.{229,230}
  Llama 16L/384D — depth_ablation_llama_16l_s{1,2}_0.4.{225,226}
               + depth_ablation_more_seeds_llama_16l_s{3,4,5}_0.4.{238,239,240}

Key output: two depth-vs-loss curves (Canon and Llama). Reading off where each
Canon point sits on the Llama curve gives the "equivalent Llama depth."
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import wandb

WANDB_API_KEY = "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"
ENTITY   = "kirill456z"
PROJECT  = "physics4llm"
OUT_DIR  = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(OUT_DIR, exist_ok=True)

LOSS_TRAILING = 100
ACC_TRAILING  = 20
RNG = np.random.default_rng(42)

C_CANON, C_LLAMA = "#d6604d", "#2166ac"

# ── Run inventory ──────────────────────────────────────────────────────────────
# Existing runs from 14_depth_ablation (known IDs).
EXISTING = {
    "Canon": {
        4:  (768, ["depth_ablation_canon_4l_s1_0.4.221", "depth_ablation_canon_4l_s2_0.4.222"]),
        8:  (512, ["depth_ablation_canon_8l_s1_0.4.223", "depth_ablation_canon_8l_s2_0.4.224"]),
        16: (384, ["depth_ablation_canon_16l_s1_0.4.219", "depth_ablation_canon_16l_s2_0.4.220",
                   "depth_ablation_more_seeds_canon_16l_s3_0.4.235",
                   "depth_ablation_more_seeds_canon_16l_s4_0.4.236",
                   "depth_ablation_more_seeds_canon_16l_s5_0.4.237"]),
    },
    "Llama": {
        4:  (768, ["depth_ablation_llama_4l_s1_0.4.227", "depth_ablation_llama_4l_s2_0.4.228"]),
        8:  (512, ["depth_ablation_llama_8l_s1_0.4.229", "depth_ablation_llama_8l_s2_0.4.230"]),
        16: (384, ["depth_ablation_llama_16l_s1_0.4.225", "depth_ablation_llama_16l_s2_0.4.226",
                   "depth_ablation_more_seeds_llama_16l_s3_0.4.238",
                   "depth_ablation_more_seeds_llama_16l_s4_0.4.239",
                   "depth_ablation_more_seeds_llama_16l_s5_0.4.240"]),
    },
}

# New runs — discovered by display-name pattern (IDs unknown until run).
NEW_PATTERNS = {
    "Canon": {
        4:  (448, "llama_depth_equivalence.*canon.*4l"),
        10: (448, "llama_depth_equivalence.*canon.*10l"),
    },
    "Llama": {
        10: (448, "llama_depth_equivalence.*llama.*10l"),
        12: (448, "llama_depth_equivalence.*llama.*12l"),
    },
}

DEPO_METRICS = [
    "evals/synthetic/depo_edges_list/hop_4/accuracy",
    "evals/synthetic/depo_adj_list/hop_4/accuracy",
]


# ── Discovery ──────────────────────────────────────────────────────────────────

def discover_new_runs(api):
    """
    Search wandb for new runs by display-name pattern.
    Returns {arch: {n_layers: (dim, [run_id, ...])}}
    """
    found = {"Canon": {}, "Llama": {}}
    for arch, depths in NEW_PATTERNS.items():
        for n_layers, (dim, pattern) in depths.items():
            print(f"\nSearching {arch} {n_layers}L ({pattern!r}) …")
            filters = {"displayName": {"$regex": pattern}, "state": "finished"}
            runs = list(api.runs(f"{ENTITY}/{PROJECT}", filters=filters))
            if not runs:
                print(f"  [warn] no finished runs — retrying without state filter")
                runs = list(api.runs(f"{ENTITY}/{PROJECT}",
                                     filters={"displayName": {"$regex": pattern}}))
            ids = []
            for r in runs:
                print(f"  {r.name:65s}  id={r.id}  state={r.state}")
                ids.append(r.id)
            found[arch][n_layers] = (dim, ids)
    return found


def build_run_table(new_runs):
    """Merge existing and new runs into {arch: {n_layers: (dim, [ids])}}."""
    import copy
    table = copy.deepcopy(EXISTING)
    for arch, depths in new_runs.items():
        for n_layers, (dim, ids) in depths.items():
            if ids:
                table[arch][n_layers] = (dim, ids)
            else:
                print(f"  [warn] no runs found for {arch} {n_layers}L — skipping")
    return table


# ── Fetching ───────────────────────────────────────────────────────────────────

def fetch_run(run_id, api):
    rec = {"run_id": run_id, "final_loss": float("nan"),
           "depo_edges": float("nan"), "depo_adj": float("nan"),
           "_loss_series": [], "_state": "unknown", "_step": 0}
    try:
        run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
        rec["_state"] = run.state
        summary = run.summary._json_dict
        rec["_step"] = int(summary.get("global_step", summary.get("_step", 0)))

        h = run.history(samples=500, keys=["loss/out"])
        col = h["loss/out"].dropna() if "loss/out" in h.columns else []
        if len(col) >= 5:
            rec["final_loss"] = float(col.iloc[-LOSS_TRAILING:].mean())
            rec["_loss_series"] = col.values.tolist()

        ha = run.history(samples=500, keys=DEPO_METRICS)
        def _acc(key):
            s = ha[key].dropna() if key in ha.columns else []
            return float(s.iloc[-ACC_TRAILING:].mean()) if len(s) >= 3 else float("nan")
        rec["depo_edges"] = _acc(DEPO_METRICS[0])
        rec["depo_adj"]   = _acc(DEPO_METRICS[1])
    except Exception as exc:
        print(f"  [warn] {run_id}: {exc}")
    return rec


def fetch_all(run_table, api):
    """Returns flat list of dicts with arch, n_layers, dim added."""
    rows = []
    for arch in ["Canon", "Llama"]:
        for n_layers in sorted(run_table[arch]):
            dim, ids = run_table[arch][n_layers]
            print(f"\n{'─'*60}\nFetching {arch} {n_layers}L/{dim}D ({len(ids)} seeds) …")
            for rid in ids:
                rec = fetch_run(rid, api)
                rec.update({"arch": arch, "n_layers": n_layers, "dim": dim})
                loss_s = f"{rec['final_loss']:.4f}" if np.isfinite(rec["final_loss"]) else "NaN"
                state_s = "" if rec["_state"] == "finished" else f"[{rec['_state']}]"
                print(f"  {arch:6s} {n_layers:>2}L/{dim}D  {rec['run_id']:55s} "
                      f"{state_s:12s} loss={loss_s}")
                rows.append(rec)
    return rows


# ── Aggregation ────────────────────────────────────────────────────────────────

def aggregate(rows, run_table):
    """Returns {arch: {n_layers: {metric: (mean, std, [vals])}}}."""
    agg = {}
    for arch in ["Canon", "Llama"]:
        agg[arch] = {}
        for n_layers in sorted(run_table[arch]):
            subset = [r for r in rows if r["arch"] == arch and r["n_layers"] == n_layers]
            agg[arch][n_layers] = {}
            for m in ["final_loss", "depo_edges", "depo_adj"]:
                vals = [r[m] for r in subset if np.isfinite(r[m])]
                agg[arch][n_layers][m] = (
                    float(np.mean(vals)) if vals else float("nan"),
                    float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan"),
                    vals,
                )
    return agg


# ── Interpolation helper ───────────────────────────────────────────────────────

def interpolate_equivalent_depth(llama_depths, llama_losses, target_loss):
    """
    Given the Llama depth curve (sorted by depth), interpolate the Llama depth
    at which loss == target_loss. Returns float or None if outside range.
    """
    pairs = sorted(zip(llama_depths, llama_losses))
    depths = [p[0] for p in pairs if np.isfinite(p[1])]
    losses = [p[1] for p in pairs if np.isfinite(p[1])]
    if len(depths) < 2:
        return None
    # Llama loss generally decreases with depth; find bracketing interval
    for i in range(len(losses) - 1):
        lo_l, hi_l = losses[i], losses[i + 1]
        lo_d, hi_d = depths[i], depths[i + 1]
        # handle non-monotone segments gracefully
        if (lo_l - target_loss) * (hi_l - target_loss) <= 0:
            t = (target_loss - lo_l) / (hi_l - lo_l)
            return lo_d + t * (hi_d - lo_d)
    return None


# ── Print summary ──────────────────────────────────────────────────────────────

def print_summary(agg):
    print(f"\n{'='*72}")
    print("Depth equivalence — Canon-ABCD vs Llama depth curves")
    print(f"{'='*72}")

    for arch in ["Canon", "Llama"]:
        print(f"\n  {arch}:")
        print(f"  {'n_layers':>8}  {'dim':>5}  {'n':>2}  {'loss/out':>10}  {'depo_mean':>10}")
        print(f"  {'─'*52}")
        for n_layers, a in sorted(agg[arch].items()):
            n   = len(a["final_loss"][2])
            lm  = a["final_loss"][0]
            dm  = float(np.nanmean([a["depo_edges"][0], a["depo_adj"][0]]))
            # dim is on the row objects; retrieve from first row
            print(f"  {n_layers:>8}  {'—':>5}  {n:>2}  {lm:>10.4f}  {dm:>10.4f}")

    # Equivalence reading
    print(f"\n{'─'*72}")
    print("Equivalent Llama depth for each Canon point (by loss interpolation):")
    llama_depths = sorted(agg["Llama"].keys())
    llama_losses = [agg["Llama"][d]["final_loss"][0] for d in llama_depths]

    for n_layers, a in sorted(agg["Canon"].items()):
        canon_loss = a["final_loss"][0]
        if not np.isfinite(canon_loss):
            print(f"  Canon {n_layers:>2}L  loss=NaN — skip")
            continue
        eq_depth = interpolate_equivalent_depth(llama_depths, llama_losses, canon_loss)
        if eq_depth is None:
            # check if outside range
            finite = [(d, l) for d, l in zip(llama_depths, llama_losses) if np.isfinite(l)]
            min_l = min(l for _, l in finite) if finite else float("nan")
            max_l = max(l for _, l in finite) if finite else float("nan")
            if canon_loss < min_l:
                note = f"Canon beats best Llama ({min_l:.4f}) — no Llama equivalent"
            elif canon_loss > max_l:
                note = f"Canon worse than shallowest Llama ({max_l:.4f}) — no Llama equivalent"
            else:
                note = "outside interpolation range"
            print(f"  Canon {n_layers:>2}L  loss={canon_loss:.4f}  →  {note}")
        else:
            offset = eq_depth - n_layers
            print(f"  Canon {n_layers:>2}L  loss={canon_loss:.4f}  →  equiv Llama ~{eq_depth:.1f}L  "
                  f"(+{offset:+.1f} layers)")

    print(f"\n{'─'*72}")
    print("VERDICT:")
    # Check if the offset is roughly constant (strong depth-equivalence hypothesis)
    offsets = []
    for n_layers, a in sorted(agg["Canon"].items()):
        canon_loss = a["final_loss"][0]
        if not np.isfinite(canon_loss):
            continue
        eq = interpolate_equivalent_depth(llama_depths, llama_losses, canon_loss)
        if eq is not None:
            offsets.append(eq - n_layers)
    if len(offsets) >= 2:
        mean_off = np.mean(offsets)
        std_off  = np.std(offsets)
        print(f"  Depth offset Canon→Llama: mean={mean_off:+.1f}L  std={std_off:.1f}L  "
              f"(n={len(offsets)} Canon points)")
        if std_off < 1.5:
            print(f"  Offset is CONSISTENT — Canon-ABCD NL ≈ Llama-(N+{mean_off:.0f})L.")
            print(f"  Each Canon block's 4 conv ops add ~{mean_off/8:.2f} equivalent Llama layers per block.")
        else:
            print(f"  Offset is VARIABLE (std={std_off:.1f}) — depth equivalence is not a simple constant shift.")
            print(f"  The relationship between Canon depth and Llama depth may be non-linear.")
    else:
        print("  Insufficient Canon points within Llama range for offset analysis.")


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_depth_curves(rows, agg, run_table):
    """
    Main figure: loss vs n_layers for Canon and Llama, with:
      - mean ± std bars
      - individual seed dots
      - dashed horizontal lines from each Canon point to the Llama curve
        showing the equivalent depth read-off
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    llama_depths = sorted(agg["Llama"].keys())
    llama_losses = [agg["Llama"][d]["final_loss"][0] for d in llama_depths]

    for ax_idx, (metric, ylabel, title_suffix) in enumerate([
        ("final_loss", "Loss/out (trailing-100 mean)", "Aggregate loss"),
        ("depo_mean",  "Depo hop-4 accuracy (mean of edges+adj)", "Depo accuracy"),
    ]):
        ax = axes[ax_idx]

        for arch, color, ls in [("Canon", C_CANON, "-o"), ("Llama", C_LLAMA, "-s")]:
            depths_sorted = sorted(agg[arch].keys())
            means, stds, all_depths = [], [], []
            for d in depths_sorted:
                a = agg[arch][d]
                if metric == "depo_mean":
                    vals_e = a["depo_edges"][2]
                    vals_a = a["depo_adj"][2]
                    # pair-wise mean per seed (zip to same length)
                    n = min(len(vals_e), len(vals_a))
                    if n > 0:
                        vals = [np.mean([e, av]) for e, av in zip(vals_e[:n], vals_a[:n])]
                    else:
                        vals = vals_e or vals_a
                    m = float(np.nanmean(vals)) if vals else float("nan")
                    s = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
                else:
                    m, s, vals = a[metric]
                means.append(m); stds.append(s); all_depths.append(d)

                # individual seeds
                if metric == "depo_mean":
                    pass  # already merged above; plot scatter separately
                n_seeds = len(a["final_loss"][2])  # proxy for seed count
                jit = RNG.uniform(-0.15, 0.15, n_seeds)
                if metric != "depo_mean":
                    seed_vals = a[metric][2]
                else:
                    seed_vals = vals
                ax.scatter(np.full(len(seed_vals), d) + jit, seed_vals,
                           color=color, s=30, alpha=0.6, zorder=4,
                           edgecolors="white", linewidths=0.4)

            valid = [(d, m, s) for d, m, s in zip(all_depths, means, stds) if np.isfinite(m)]
            if not valid:
                continue
            xv = [v[0] for v in valid]
            mv = [v[1] for v in valid]
            sv = [v[2] if np.isfinite(v[2]) else 0 for v in valid]
            ax.errorbar(xv, mv, yerr=sv, fmt=ls, color=color, lw=2, ms=8,
                        capsize=4, label=arch, zorder=5)

        # Equivalence arrows (loss panel only)
        if metric == "final_loss":
            for n_layers, a in sorted(agg["Canon"].items()):
                canon_loss = a["final_loss"][0]
                if not np.isfinite(canon_loss):
                    continue
                eq = interpolate_equivalent_depth(llama_depths, llama_losses, canon_loss)
                if eq is None:
                    continue
                # horizontal dashed line from Canon point to Llama curve
                ax.annotate(
                    "", xy=(eq, canon_loss), xytext=(n_layers, canon_loss),
                    arrowprops=dict(arrowstyle="->", color="gray",
                                    lw=1.2, linestyle="dashed"),
                )
                ax.text((n_layers + eq) / 2, canon_loss + 0.003,
                        f"+{eq - n_layers:.1f}L", ha="center", va="bottom",
                        fontsize=8, color="gray")

        ax.set_xlabel("n_layers")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Depth equivalence — {title_suffix}")
        ax.grid(alpha=0.3, ls="--")
        ax.legend(fontsize=9)

    fig.suptitle("Canon-ABCD vs Llama: depth curves at fixed ~30M params\n"
                 "Arrows show equivalent Llama depth for each Canon point",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "depth_equivalence_curves.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"Saved: {path}")


def plot_offset_summary(agg):
    """Bar chart of Canon→Llama depth offset per Canon depth."""
    llama_depths = sorted(agg["Llama"].keys())
    llama_losses = [agg["Llama"][d]["final_loss"][0] for d in llama_depths]

    canon_depths, offsets, canon_losses_plot = [], [], []
    for n_layers, a in sorted(agg["Canon"].items()):
        canon_loss = a["final_loss"][0]
        if not np.isfinite(canon_loss):
            continue
        eq = interpolate_equivalent_depth(llama_depths, llama_losses, canon_loss)
        if eq is None:
            continue
        canon_depths.append(n_layers)
        offsets.append(eq - n_layers)
        canon_losses_plot.append(canon_loss)

    if not offsets:
        print("  No offset data to plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    xs = np.arange(len(canon_depths))
    bars = ax.bar(xs, offsets, color=C_CANON, alpha=0.8, width=0.5)
    ax.axhline(np.mean(offsets), color="black", ls="--", lw=1.5,
               label=f"mean offset = {np.mean(offsets):+.1f}L")
    for xi, n_layers, off, loss in zip(xs, canon_depths, offsets, canon_losses_plot):
        ax.text(xi, off + 0.05, f"{off:+.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"Canon {d}L" for d in canon_depths])
    ax.set_ylabel("Equivalent Llama depth offset (layers)")
    ax.set_title("How many extra Llama layers is each Canon-ABCD model worth?\n"
                 "(interpolated from Llama depth curve)")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3, ls="--")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "depth_offset_summary.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"Saved: {path}")


def plot_training_curves(rows):
    """Training curves for the new runs only (Canon 4L, Canon 10L, Llama 10L, Llama 12L)."""
    new_groups = [
        ("Canon", 4,  448, C_CANON, "-"),
        ("Canon", 10, 448, C_CANON, "--"),
        ("Llama", 10, 448, C_LLAMA, "-"),
        ("Llama", 12, 448, C_LLAMA, "--"),
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    for arch, n_layers, dim, color, ls in new_groups:
        subset = [r for r in rows if r["arch"] == arch and r["n_layers"] == n_layers
                  and r["dim"] == dim and len(r["_loss_series"]) > 0]
        if not subset:
            continue
        min_len = min(len(r["_loss_series"]) for r in subset)
        mat = np.stack([np.array(r["_loss_series"])[:min_len] for r in subset])
        steps = np.arange(min_len)
        mean = mat.mean(axis=0)
        lo, hi = mat.min(axis=0), mat.max(axis=0)
        label = f"{arch} {n_layers}L/448D (n={len(subset)})"
        ax.plot(steps, mean, color=color, ls=ls, lw=2, label=label)
        ax.fill_between(steps, lo, hi, color=color, alpha=0.12)
    ax.set_xlabel("Step (sample index)")
    ax.set_ylabel("loss/out")
    ax.set_title("New runs — training curves (448D configs only)")
    ax.grid(alpha=0.3, ls="--"); ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "depth_equiv_training_curves.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"Saved: {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    api = wandb.Api(api_key=WANDB_API_KEY)

    print("=" * 72)
    print("Depth equivalence experiment")
    print("=" * 72)

    new_runs  = discover_new_runs(api)
    run_table = build_run_table(new_runs)

    rows = fetch_all(run_table, api)
    agg  = aggregate(rows, run_table)

    print_summary(agg)

    plot_depth_curves(rows, agg, run_table)
    plot_offset_summary(agg)
    plot_training_curves(rows)

    print("\nDone.")
