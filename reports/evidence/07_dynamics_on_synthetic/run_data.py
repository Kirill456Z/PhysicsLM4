"""
Exp 1 — Canon training dynamics on the synthetic task setup.

Fetches the same dynamics metrics that were measured on text modeling (F4) but now
on the 8-task mixed synthetic benchmark.  Prints a side-by-side comparison to the
text-modeling baseline so we can judge whether the same mechanisms are at play.

Runs (3 seeds each, 30M Canon-ABCD / Llama, trained on the 8-task mix):
  Canon: dynamics_on_synthetic_canon_s{1..3}_0.4.{196..198}
  Llama: dynamics_on_synthetic_llama_s{1..3}_0.4.{199..201}

Text-modeling reference (F4, single seed each):
  Canon: canon_text_0.1.129
  Llama: canon_text_0.1.130
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

# ── Run IDs ────────────────────────────────────────────────────────────────────
SYNTH_CANON_IDS = [
    "dynamics_on_synthetic_canon_s1_0.4.196",
    "dynamics_on_synthetic_canon_s2_0.4.197",
    "dynamics_on_synthetic_canon_s3_0.4.198",
]
SYNTH_LLAMA_IDS = [
    "dynamics_on_synthetic_llama_s1_0.4.199",
    "dynamics_on_synthetic_llama_s2_0.4.200",
    "dynamics_on_synthetic_llama_s3_0.4.201",
]
TEXT_CANON_ID = "canon_text_0.1.129"
TEXT_LLAMA_ID = "canon_text_0.1.130"

N_LAYERS = 8
CANON_TYPES = ["A", "B", "C", "D"]
KERNEL_SIZE = 4

# ── Metric key helpers ─────────────────────────────────────────────────────────
def grad_keys():
    return [f"grad_contrib/layer_{l}" for l in range(N_LAYERS)]

def kurtosis_keys():
    return [f"outlier_features/kurtosis/layer_{l}" for l in range(N_LAYERS)]

def rms_keys():
    return [f"canon_align/rms_ratio/canon{ct}/layer_{l}"
            for ct in CANON_TYPES for l in range(N_LAYERS)]

def cos_keys():
    return [f"canon_align/cos_sim/canon{ct}/layer_{l}"
            for ct in ["A", "C"] for l in range(N_LAYERS)]

def weight_keys():
    return [f"canon_weight/canon{ct}/layer_{l}/shift_{s}"
            for ct in CANON_TYPES for l in range(N_LAYERS) for s in range(KERNEL_SIZE)]

ALL_DYNAMICS_KEYS = grad_keys() + kurtosis_keys() + rms_keys() + cos_keys() + weight_keys() + ["loss/out"]
LLAMA_DYNAMICS_KEYS = grad_keys() + kurtosis_keys() + ["loss/out"]

# ── Fetch helpers ──────────────────────────────────────────────────────────────
def fetch_history(run_id, keys, samples=500):
    api = wandb.Api(api_key=WANDB_API_KEY)
    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
    # Fetch without key filter first so _step is always present, then filter cols
    hist = run.history(samples=samples)
    if hist.empty:
        return hist
    if "_step" not in hist.columns:
        hist = hist.reset_index().rename(columns={"index": "_step"})
    keep = ["_step"] + [k for k in keys if k in hist.columns]
    return hist[keep].sort_values("_step").reset_index(drop=True)


def trailing_mean(hist, key, n=100):
    if key not in hist.columns:
        return float("nan")
    s = hist[key].dropna()
    return float(s.iloc[-min(n, len(s)):].mean()) if not s.empty else float("nan")


def avg_histories(run_ids, keys, samples=500, label=""):
    """Fetch multiple seeds and return a single averaged DataFrame."""
    hists = []
    for rid in run_ids:
        print(f"  fetching {rid}…")
        h = fetch_history(rid, keys, samples=samples)
        hists.append(h)
    # Align on _step by interpolation then average
    aligned = {}
    for h in hists:
        if h.empty:
            continue
        for k in keys:
            if k in h.columns:
                aligned.setdefault(k, []).append(h.set_index("_step")[k])
    if not aligned:
        return pd.DataFrame()
    ref_index = sorted(set().union(*[s.index for s in next(iter(aligned.values()))]))
    result = {"_step": ref_index}
    for k, series_list in aligned.items():
        combined = pd.concat(series_list, axis=1).reindex(ref_index).interpolate("index")
        result[k] = combined.mean(axis=1).values
    return pd.DataFrame(result)

# ── Diagnostics print ──────────────────────────────────────────────────────────
def print_gradient_comparison(synth_canon, synth_llama, text_canon, text_llama):
    """Side-by-side: gradient ratio deep/shallow for synthetic vs text, Canon vs Llama."""
    print(f"\n{'='*72}")
    print("GRADIENT CONTRIBUTION: deep/shallow ratio and per-layer values (trailing avg)")
    print(f"{'='*72}")
    print(f"{'Layer':>5}  {'Synth-Llama':>12}  {'Synth-Canon':>12}  {'Text-Llama':>12}  {'Text-Canon':>12}")
    gk = grad_keys()
    sl = [trailing_mean(synth_llama, k) for k in gk]
    sc = [trailing_mean(synth_canon, k) for k in gk]
    tl = [trailing_mean(text_llama,  k) for k in gk]
    tc = [trailing_mean(text_canon,  k) for k in gk]
    for i in range(N_LAYERS):
        print(f"  L{i:>2}  {sl[i]:>12.4f}  {sc[i]:>12.4f}  {tl[i]:>12.4f}  {tc[i]:>12.4f}")
    # Deep/shallow ratio (L7 / L0)
    for label, vals in [("Synth-Llama", sl), ("Synth-Canon", sc),
                        ("Text-Llama",  tl), ("Text-Canon",  tc)]:
        ratio = vals[-1] / vals[0] if vals[0] != 0 else float("nan")
        print(f"  {label}: deep/shallow = {ratio:.2f}×")


def print_kurtosis_comparison(synth_canon, synth_llama, text_canon, text_llama):
    print(f"\n{'='*72}")
    print("KURTOSIS: per-layer (trailing avg)")
    print(f"{'='*72}")
    print(f"{'Layer':>5}  {'Synth-Llama':>12}  {'Synth-Canon':>12}  {'Text-Llama':>12}  {'Text-Canon':>12}")
    kk = kurtosis_keys()
    for i in range(N_LAYERS):
        k = kk[i]
        print(f"  L{i:>2}  {trailing_mean(synth_llama,k):>12.2f}  {trailing_mean(synth_canon,k):>12.2f}"
              f"  {trailing_mean(text_llama,k):>12.2f}  {trailing_mean(text_canon,k):>12.2f}")


def print_canon_weight_comparison(synth_canon, text_canon):
    print(f"\n{'='*72}")
    print("CANON WEIGHTS: shift pattern (layers 1-7 avg, trailing 50 steps)")
    print("Expected: shift_0 < shift_1 < shift_2  (cross-token mixing dominates)")
    print(f"{'='*72}")
    for ct in CANON_TYPES:
        synth_shifts, text_shifts = [], []
        for shift in range(KERNEL_SIZE):
            sv, tv = [], []
            for l in range(1, N_LAYERS):
                k = f"canon_weight/canon{ct}/layer_{l}/shift_{shift}"
                sv.append(trailing_mean(synth_canon, k, n=50))
                tv.append(trailing_mean(text_canon,  k, n=50))
            synth_shifts.append(np.nanmean(sv))
            text_shifts.append(np.nanmean(tv))
        synth_str = "  ".join(f"s{i}={v:.3f}" for i, v in enumerate(synth_shifts))
        text_str  = "  ".join(f"s{i}={v:.3f}" for i, v in enumerate(text_shifts))
        print(f"  {ct}  synth: {synth_str}")
        print(f"     text:  {text_str}")


def print_loss_comparison(synth_canon_ids, synth_llama_ids):
    print(f"\n{'='*72}")
    print("FINAL LOSS (trailing 100 steps, per seed)")
    print(f"{'='*72}")
    api = wandb.Api(api_key=WANDB_API_KEY)
    for label, ids in [("Synth-Canon", synth_canon_ids), ("Synth-Llama", synth_llama_ids)]:
        vals = []
        for rid in ids:
            run = api.run(f"{ENTITY}/{PROJECT}/{rid}")
            h = run.history(samples=200, keys=["loss/out"])
            s = h["loss/out"].dropna() if "loss/out" in h.columns else pd.Series()
            v = float(s.iloc[-100:].mean()) if len(s) >= 5 else float("nan")
            vals.append(v)
            print(f"  {label}  {rid:50s}  loss={v:.4f}")
        print(f"  {label} → mean={np.nanmean(vals):.4f}  std={np.nanstd(vals, ddof=1):.4f}\n")

# ── Plots ──────────────────────────────────────────────────────────────────────
def plot_gradient_comparison(synth_canon, synth_llama, text_canon, text_llama):
    layers = np.arange(N_LAYERS)
    gk = grad_keys()
    sc = [trailing_mean(synth_canon, k) for k in gk]
    sl = [trailing_mean(synth_llama, k) for k in gk]
    tc = [trailing_mean(text_canon,  k) for k in gk]
    tl = [trailing_mean(text_llama,  k) for k in gk]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, (canon_v, llama_v, title) in zip(axes, [
        (sc, sl, "Synthetic tasks (8-task mix)"),
        (tc, tl, "Text modeling"),
    ]):
        ax.plot(layers, llama_v, "o-", color="#2166ac", label="Llama", lw=2)
        ax.plot(layers, canon_v, "s-", color="#d6604d", label="Canon-ABCD", lw=2)
        ax.set_title(title); ax.set_xlabel("Layer"); ax.set_ylabel("Gradient contribution")
        ax.legend(); ax.grid(True, alpha=0.3, ls="--"); ax.set_xticks(layers)
    fig.suptitle("Gradient contribution per layer: synthetic vs text (end-of-training)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "gradient_comparison_synth_vs_text.png")
    fig.savefig(path, dpi=150); plt.close(fig); print(f"Saved: {path}")


def plot_kurtosis_comparison(synth_canon, synth_llama, text_canon, text_llama):
    layers = np.arange(N_LAYERS)
    kk = kurtosis_keys()
    sc = [trailing_mean(synth_canon, k) for k in kk]
    sl = [trailing_mean(synth_llama, k) for k in kk]
    tc = [trailing_mean(text_canon,  k) for k in kk]
    tl = [trailing_mean(text_llama,  k) for k in kk]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, (canon_v, llama_v, title) in zip(axes, [
        (sc, sl, "Synthetic tasks (8-task mix)"),
        (tc, tl, "Text modeling"),
    ]):
        ax.plot(layers, llama_v, "o-", color="#2166ac", label="Llama", lw=2)
        ax.plot(layers, canon_v, "s-", color="#d6604d", label="Canon-ABCD", lw=2)
        ax.set_title(title); ax.set_xlabel("Layer"); ax.set_ylabel("Kurtosis")
        ax.legend(); ax.grid(True, alpha=0.3, ls="--"); ax.set_xticks(layers)
    fig.suptitle("Activation kurtosis per layer: synthetic vs text (end-of-training)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "kurtosis_comparison_synth_vs_text.png")
    fig.savefig(path, dpi=150); plt.close(fig); print(f"Saved: {path}")


def plot_rms_ratio(synth_canon):
    layers = np.arange(N_LAYERS)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, ct in enumerate(CANON_TYPES):
        vals = [trailing_mean(synth_canon, f"canon_align/rms_ratio/canon{ct}/layer_{l}")
                for l in range(N_LAYERS)]
        ax.plot(layers, vals, "o-", color=colors[i], label=f"Canon-{ct}", lw=2)
    ax.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.6, label="identity")
    ax.set_xlabel("Layer"); ax.set_ylabel("RMS ratio (output/input)")
    ax.set_title("Canon RMS ratio on synthetic tasks\n(ratio>1 means Canon amplifies the residual stream)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, ls="--"); ax.set_xticks(layers)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "rms_ratio_synthetic.png")
    fig.savefig(path, dpi=150); plt.close(fig); print(f"Saved: {path}")


def plot_canon_weights(synth_canon):
    layers = np.arange(N_LAYERS)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i, ct in enumerate(CANON_TYPES):
        ax = axes[i // 2][i % 2]
        for shift in range(KERNEL_SIZE):
            vals = [trailing_mean(synth_canon, f"canon_weight/canon{ct}/layer_{l}/shift_{shift}")
                    for l in range(N_LAYERS)]
            ax.plot(layers, vals, "o-", color=colors[shift], label=f"shift {shift}", lw=2)
        ax.set_title(f"Canon-{ct} weights on synthetic tasks")
        ax.set_xlabel("Layer"); ax.set_ylabel("Weight value")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, ls="--"); ax.set_xticks(layers)
    fig.suptitle("Canon learned weights on synthetic tasks (end-of-training)\n"
                 "Same pattern as text? shift_2 > others expected",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "canon_weights_synthetic.png")
    fig.savefig(path, dpi=150); plt.close(fig); print(f"Saved: {path}")


def plot_dynamics_summary_4panel(synth_canon, synth_llama, text_canon, text_llama):
    """4-panel figure: gradient + kurtosis × synthetic/text for direct comparison."""
    layers = np.arange(N_LAYERS)
    gk = grad_keys(); kk = kurtosis_keys()
    c_blue, c_red = "#2166ac", "#d6604d"

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    panels = [
        (axes[0, 0], [trailing_mean(synth_llama,k) for k in gk],
                     [trailing_mean(synth_canon,k) for k in gk],
                     "Gradient contribution — Synthetic"),
        (axes[0, 1], [trailing_mean(text_llama, k) for k in gk],
                     [trailing_mean(text_canon, k) for k in gk],
                     "Gradient contribution — Text"),
        (axes[1, 0], [trailing_mean(synth_llama,k) for k in kk],
                     [trailing_mean(synth_canon,k) for k in kk],
                     "Kurtosis — Synthetic"),
        (axes[1, 1], [trailing_mean(text_llama, k) for k in kk],
                     [trailing_mean(text_canon, k) for k in kk],
                     "Kurtosis — Text"),
    ]
    for ax, llama_v, canon_v, title in panels:
        ax.plot(layers, llama_v, "o-", color=c_blue, label="Llama", lw=2)
        ax.plot(layers, canon_v, "s-", color=c_red,  label="Canon", lw=2)
        ax.set_title(title, fontsize=10); ax.set_xlabel("Layer")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, ls="--"); ax.set_xticks(layers)
    axes[0,0].set_ylabel("Gradient contribution")
    axes[1,0].set_ylabel("Kurtosis")
    fig.suptitle("Training dynamics: synthetic tasks vs text modeling\n"
                 "Do the same mechanisms (gradient uniformity, kurtosis suppression) appear on synthetic tasks?",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "dynamics_4panel_synth_vs_text.png")
    fig.savefig(path, dpi=150); plt.close(fig); print(f"Saved: {path}")


# ── Exp 11: Kurtosis temporal evolution ───────────────────────────────────────
# Runs a single seed (s1) with 1000 samples to see if Canon kurtosis diverges
# from Llama immediately (structural) or builds up over training (learned).
PROBE_LAYERS = [0, 2, 4, 6]   # representative subset to keep the plot readable

def fetch_kurtosis_trajectory(run_id, samples=1000):
    """Fetch kurtosis time-series for PROBE_LAYERS."""
    keys = [f"outlier_features/kurtosis/layer_{l}" for l in range(N_LAYERS)]
    return fetch_history(run_id, keys, samples=samples)


def print_kurtosis_onset(synth_canon_traj, synth_llama_traj):
    """Print at what training step Canon and Llama kurtosis first diverge in deep layers."""
    print(f"\n{'='*72}")
    print("EXP 11: Kurtosis temporal evolution — structural or learned?")
    print("Checking L2–L7 where Canon ends up HIGHER than Llama on synthetic tasks")
    print(f"{'='*72}")
    for l in [2, 4, 6]:
        ck = f"outlier_features/kurtosis/layer_{l}"
        lk = f"outlier_features/kurtosis/layer_{l}"
        if ck not in synth_canon_traj.columns or lk not in synth_llama_traj.columns:
            print(f"  L{l}: no data"); continue
        # Align on steps
        c = synth_canon_traj[["_step", ck]].dropna().set_index("_step")[ck]
        l_series = synth_llama_traj[["_step", lk]].dropna().set_index("_step")[lk]
        common = c.index.intersection(l_series.index)
        if len(common) == 0:
            print(f"  L{l}: no common steps"); continue
        # First and last 10% of training
        n = len(common)
        early_steps = sorted(common)[:max(1, n // 10)]
        late_steps  = sorted(common)[-max(1, n // 10):]
        c_early = c.loc[early_steps].mean()
        l_early = l_series.loc[early_steps].mean()
        c_late  = c.loc[late_steps].mean()
        l_late  = l_series.loc[late_steps].mean()
        print(f"  L{l}  early steps: Canon={c_early:.1f}  Llama={l_early:.1f}  diff={c_early-l_early:+.1f}")
        print(f"       late  steps: Canon={c_late:.1f}   Llama={l_late:.1f}   diff={c_late-l_late:+.1f}")
        if c_early > l_early * 1.3:
            print(f"       → STRUCTURAL: Canon kurtosis elevated from the start at L{l}")
        elif c_late > l_late * 1.3 and c_early <= l_early * 1.1:
            print(f"       → LEARNED: kurtosis divergence builds over training at L{l}")
        else:
            print(f"       → MIXED / UNCLEAR at L{l}")


def plot_kurtosis_over_time(synth_canon_traj, synth_llama_traj,
                            text_canon_traj, text_llama_traj):
    """
    2×2 grid: rows=task domain (synth/text), cols=early layers (0,2) / deep layers (4,6).
    Each panel shows Canon vs Llama kurtosis over training steps.
    """
    c_blue, c_red = "#2166ac", "#d6604d"
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    configs = [
        # (row, col, domain_label, canon_traj, llama_traj, layers)
        (0, 0, "Synthetic — early layers (L0, L2)", synth_canon_traj, synth_llama_traj, [0, 2]),
        (0, 1, "Synthetic — deep layers (L4, L6)",  synth_canon_traj, synth_llama_traj, [4, 6]),
        (1, 0, "Text — early layers (L0, L2)",       text_canon_traj,  text_llama_traj,  [0, 2]),
        (1, 1, "Text — deep layers (L4, L6)",         text_canon_traj,  text_llama_traj,  [4, 6]),
    ]
    layer_styles = {0: "-", 2: "--", 4: "-", 6: "--"}
    layer_alpha  = {0: 1.0, 2: 0.7, 4: 1.0, 6: 0.7}

    for row, col, title, c_traj, l_traj, layers in configs:
        ax = axes[row, col]
        for l in layers:
            key = f"outlier_features/kurtosis/layer_{l}"
            if key in c_traj.columns:
                s = c_traj[["_step", key]].dropna()
                s_smooth = s.copy(); s_smooth[key] = s[key].rolling(10, min_periods=1).mean()
                ax.plot(s_smooth["_step"], s_smooth[key],
                        color=c_red, ls=layer_styles[l], alpha=layer_alpha[l], lw=1.8,
                        label=f"Canon L{l}")
            if key in l_traj.columns:
                s = l_traj[["_step", key]].dropna()
                s_smooth = s.copy(); s_smooth[key] = s[key].rolling(10, min_periods=1).mean()
                ax.plot(s_smooth["_step"], s_smooth[key],
                        color=c_blue, ls=layer_styles[l], alpha=layer_alpha[l], lw=1.8,
                        label=f"Llama L{l}")
        ax.set_title(title, fontsize=9); ax.set_xlabel("Step"); ax.set_ylabel("Kurtosis")
        ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3, ls="--")

    fig.suptitle(
        "Exp 11: Kurtosis over training — is Canon's elevated kurtosis on synthetic tasks "
        "structural (present from step 1) or learned (builds up)?",
        fontsize=10, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "kurtosis_over_time.png")
    fig.savefig(path, dpi=150); plt.close(fig); print(f"Saved: {path}")


# ── Exp 12: Per-task Canon improvement + kurtosis-layer context ────────────────
# 30M scaling law runs (5 seeds each) — we fetch just these to get per-task gaps.
LLAMA_30M_IDS = [
    "scaling_law_exps_3_llama_30m_0.4.83",
    "depo_variance_impact_investigation_2_all_tasks_seed_55_0.4.120",
    "depo_variance_impact_investigation_2_all_tasks_seed_56_0.4.121",
    "scaling_law_4_more_seeds_llama_30m_seed_57_0.4.158",
    "scaling_law_4_more_seeds_llama_30m_seed_58_0.4.159",
]
CANON_30M_IDS = [
    "scaling_law_exps_3_canon_30m_0.4.79",
    "remaining_runs_canon_30m_seed_55_0.4.128",
    "remaining_runs_canon_30m_seed_56_0.4.129",
    "scaling_law_4_more_seeds_canon_30m_seed_57_0.4.150",
    "scaling_law_4_more_seeds_canon_30m_seed_58_0.4.151",
]

TASK_METRICS_12 = {
    "depo_el":   ("evals/synthetic/depo_edges_list/hop_4/accuracy",      "Depo (el)"),
    "depo_al":   ("evals/synthetic/depo_adj_list/hop_4/accuracy",        "Depo (al)"),
    "cc_el":     ("evals/synthetic/concomp_factor_edges_list/prefix_accuracy", "Concomp (el)"),
    "cc_al":     ("evals/synthetic/concomp_factor_adj_list/prefix_accuracy",   "Concomp (al)"),
    "sp_el":     ("evals/synthetic/shortest_path_edges_list/set_accuracy",     "ShortPath (el)"),
    "sp_al":     ("evals/synthetic/shortest_path_adj_list/set_accuracy",       "ShortPath (al)"),
    "bfs_el":    ("evals/synthetic/bfs_edges_list/set_recall",                 "BFS (el)"),
    "bfs_al":    ("evals/synthetic/bfs_adj_list/set_recall",                   "BFS (al)"),
}


def fetch_30m_per_task_summary():
    """Fetch per-task eval metrics for 30M Canon and Llama, return mean per arch."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from utils import fetch_runs_results
    all_metric_keys = [v[0] for v in TASK_METRICS_12.values()]
    run_ids = {"llama": LLAMA_30M_IDS, "canon": CANON_30M_IDS}
    df = fetch_runs_results(run_ids, api_key=WANDB_API_KEY,
                            entity=ENTITY, project=PROJECT)
    id_cols = {"run_id", "run_name", "label", "n_params"}
    group_cols = ["label"]
    metric_cols = [c for c in df.columns if c not in id_cols]
    avg = df.groupby(group_cols, as_index=False)[metric_cols].mean(numeric_only=True)
    return avg


def print_per_task_gaps(avg_30m, synth_canon):
    """Print per-task Canon-Llama improvement alongside deep-layer kurtosis."""
    print(f"\n{'='*72}")
    print("EXP 12: Per-task Canon improvement at 30M + deep-layer kurtosis context")
    print(f"{'='*72}")
    canon_row = avg_30m[avg_30m["label"] == "canon"].iloc[0] if "canon" in avg_30m["label"].values else None
    llama_row = avg_30m[avg_30m["label"] == "llama"].iloc[0] if "llama" in avg_30m["label"].values else None
    if canon_row is None or llama_row is None:
        print("  No data."); return

    print(f"  {'Task':>20}  {'Canon':>8}  {'Llama':>8}  {'Δ Canon':>10}")
    gaps = {}
    for key, (metric, label) in TASK_METRICS_12.items():
        c_val = canon_row.get(metric, float("nan"))
        l_val = llama_row.get(metric, float("nan"))
        delta = c_val - l_val
        gaps[label] = delta
        print(f"  {label:>20}  {c_val:>8.4f}  {l_val:>8.4f}  {delta:>+10.4f}")

    # Deep-layer kurtosis (L4-L7) for Canon on synthetic tasks
    kk = [f"outlier_features/kurtosis/layer_{l}" for l in [4, 5, 6, 7]]
    deep_kurt = np.nanmean([trailing_mean(synth_canon, k) for k in kk])
    print(f"\n  Canon deep-layer kurtosis (L4–L7 avg): {deep_kurt:.2f}")
    print(f"  (Llama deep-layer kurtosis L4–L7: ~{2.0:.1f} — Canon is {deep_kurt/2.0:.1f}× higher)")

    # Qualitative assessment
    delta_vals = list(gaps.values())
    task_labels = list(gaps.keys())
    best_task  = task_labels[int(np.argmax(delta_vals))]
    worst_task = task_labels[int(np.argmin(delta_vals))]
    print(f"\n  Best Canon improvement:  {best_task}  (Δ={max(delta_vals):+.4f})")
    print(f"  Worst Canon improvement: {worst_task}  (Δ={min(delta_vals):+.4f})")


def plot_per_task_improvement(avg_30m, synth_canon):
    """Bar chart of per-task Canon improvement + kurtosis profile overlay."""
    canon_row = avg_30m[avg_30m["label"] == "canon"].iloc[0] if "canon" in avg_30m["label"].values else None
    llama_row = avg_30m[avg_30m["label"] == "llama"].iloc[0] if "llama" in avg_30m["label"].values else None
    if canon_row is None or llama_row is None:
        print("  No 30M data for per-task plot."); return

    labels, deltas, canon_vals, llama_vals = [], [], [], []
    for key, (metric, label) in TASK_METRICS_12.items():
        c = canon_row.get(metric, float("nan"))
        l = llama_row.get(metric, float("nan"))
        labels.append(label)
        canon_vals.append(c)
        llama_vals.append(l)
        deltas.append(c - l)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: absolute values Canon vs Llama per task
    ax = axes[0]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, llama_vals, w, label="Llama", color="#2166ac", alpha=0.8)
    ax.bar(x + w/2, canon_vals, w, label="Canon", color="#d6604d", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Primary metric"); ax.set_title("Per-task: Canon vs Llama at 30M")
    ax.legend(); ax.grid(axis="y", alpha=0.3, ls="--")

    # Right: Canon improvement (delta) per task, colored by sign
    ax = axes[1]
    colors = ["#d6604d" if d > 0 else "#2166ac" for d in deltas]
    bars = ax.bar(x, deltas, color=colors, alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Canon − Llama (higher = Canon better)")
    ax.set_title("Exp 12: Canon improvement per task (30M, 5-seed avg)\n"
                 "Deep-layer kurtosis elevation (Canon > Llama L2–L7) — does it match?")
    ax.grid(axis="y", alpha=0.3, ls="--")

    # Annotate the kurtosis elevation in a text box
    kk = [f"outlier_features/kurtosis/layer_{l}" for l in [4, 5, 6, 7]]
    deep_kurt = np.nanmean([trailing_mean(synth_canon, k) for k in kk])
    ax.text(0.98, 0.97,
            f"Canon deep-layer\nkurtosis L4–L7: {deep_kurt:.1f}\n(Llama ≈ 2.0)",
            transform=ax.transAxes, fontsize=8, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))

    fig.suptitle("Exp 12: Which tasks benefit most from Canon? "
                 "Does this match the kurtosis pattern?",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "per_task_canon_improvement.png")
    fig.savefig(path, dpi=150); plt.close(fig); print(f"Saved: {path}")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Fetching dynamics_on_synthetic runs (averaged, 500 samples) ===")
    print("\n[Synth Canon — averaged over 3 seeds]")
    synth_canon = avg_histories(SYNTH_CANON_IDS, ALL_DYNAMICS_KEYS, samples=500)
    print("\n[Synth Llama — averaged over 3 seeds]")
    synth_llama = avg_histories(SYNTH_LLAMA_IDS, LLAMA_DYNAMICS_KEYS, samples=500)
    print("\n[Text Canon — single reference seed]")
    text_canon = fetch_history(TEXT_CANON_ID, ALL_DYNAMICS_KEYS, samples=500)
    print("\n[Text Llama — single reference seed]")
    text_llama = fetch_history(TEXT_LLAMA_ID, LLAMA_DYNAMICS_KEYS, samples=500)

    # Existing end-of-training analysis
    print_loss_comparison(SYNTH_CANON_IDS, SYNTH_LLAMA_IDS)
    print_gradient_comparison(synth_canon, synth_llama, text_canon, text_llama)
    print_kurtosis_comparison(synth_canon, synth_llama, text_canon, text_llama)
    print_canon_weight_comparison(synth_canon, text_canon)

    plot_gradient_comparison(synth_canon, synth_llama, text_canon, text_llama)
    plot_kurtosis_comparison(synth_canon, synth_llama, text_canon, text_llama)
    plot_rms_ratio(synth_canon)
    plot_canon_weights(synth_canon)
    plot_dynamics_summary_4panel(synth_canon, synth_llama, text_canon, text_llama)

    # ── Exp 11: Kurtosis temporal evolution ───────────────────────────────────
    print("\n=== Exp 11: Fetching kurtosis trajectories (1000 samples, seed 1 only) ===")
    synth_canon_traj = fetch_kurtosis_trajectory(SYNTH_CANON_IDS[0], samples=1000)
    synth_llama_traj = fetch_kurtosis_trajectory(SYNTH_LLAMA_IDS[0], samples=1000)
    text_canon_traj  = fetch_kurtosis_trajectory(TEXT_CANON_ID,      samples=1000)
    text_llama_traj  = fetch_kurtosis_trajectory(TEXT_LLAMA_ID,       samples=1000)
    print(f"  synth_canon rows={len(synth_canon_traj)}  text_canon rows={len(text_canon_traj)}")

    print_kurtosis_onset(synth_canon_traj, synth_llama_traj)
    plot_kurtosis_over_time(synth_canon_traj, synth_llama_traj,
                            text_canon_traj,  text_llama_traj)

    # ── Exp 12: Per-task improvement + kurtosis context ───────────────────────
    print("\n=== Exp 12: Fetching 30M scaling law runs for per-task breakdown ===")
    avg_30m = fetch_30m_per_task_summary()
    print_per_task_gaps(avg_30m, synth_canon)
    plot_per_task_improvement(avg_30m, synth_canon)

    print("\nDone.")
