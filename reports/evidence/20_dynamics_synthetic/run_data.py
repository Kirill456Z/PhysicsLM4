"""
Slide "Mechanisms: Gradient Smoothing & Representation Upscaling" — synthetic version.

The deck currently shows these two end-of-training mechanism plots for *text*
modeling only (reports/new_paper_plan/plots/grad_profile.png, rms_ratio.png):

  1. Gradient smoothing  — per-layer gradient contribution (Llama vs Canon-ABCD).
     Llama spikes at shallow layers and decays; Canon leaves a flatter profile.
  2. Representation upscaling — Canon conv output/input RMS ratio (Canon-A / Canon-C).
     The conv amplifies the residual stream (ratio > 1), strongest in deep layers.

This script reproduces the *same two plots* on the 8-task synthetic mix, averaging
over the 3 dynamics seeds so the synthetic curves are directly comparable to the
text panels. Run from the repo root:

    uv run python reports/evidence/20_dynamics_synthetic/run_data.py

Runs (30M Canon-ABCD / Llama, trained on the 8-task mix, 3 seeds each):
  Canon: dynamics_on_synthetic_canon_s{1..3}_0.4.{196..198}
  Llama: dynamics_on_synthetic_llama_s{1..3}_0.4.{199..201}
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

# ── Run IDs (3 seeds each) ──────────────────────────────────────────────────────
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

N_LAYERS = 8
CANON_TYPES = ["A", "B", "C", "D"]
KERNEL_SIZE = 4

# Match the colors used in the text-modeling slide plots.
C_LLAMA = "#2166ac"
C_CANON = "#d6604d"
# Per-position colors/markers for the RMS-ratio plot. Only A (pre-attention) and
# C (pre-FFN) log the rms_ratio metric, matching the text-modeling slide.
RMS_STYLES = [
    ("A", "#d6604d", "s-"),
    ("C", "#f4a582", "^-"),
]
RMS_TYPES = [ct for ct, _, _ in RMS_STYLES]

# ── Metric key helpers ──────────────────────────────────────────────────────────
def grad_key(layer):       return f"grad_contrib/layer_{layer}"
def rms_key(ctype, layer): return f"canon_align/rms_ratio/canon{ctype}/layer_{layer}"

GRAD_KEYS = [grad_key(l) for l in range(N_LAYERS)]
RMS_KEYS  = [rms_key(ct, l) for ct in CANON_TYPES for l in range(N_LAYERS)]

# ── Fetch helpers ───────────────────────────────────────────────────────────────
def fetch_history(run_id, keys, samples=500):
    api = wandb.Api(api_key=WANDB_API_KEY)
    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
    hist = run.history(samples=samples)
    if hist.empty:
        return hist
    if "_step" not in hist.columns:
        hist = hist.reset_index().rename(columns={"index": "_step"})
    keep = ["_step"] + [k for k in keys if k in hist.columns]
    return hist[keep].sort_values("_step").reset_index(drop=True)


def trailing_mean(hist, key, n=100):
    """End-of-training value: mean of the trailing n logged points."""
    if key not in hist.columns:
        return float("nan")
    s = hist[key].dropna()
    return float(s.iloc[-min(n, len(s)):].mean()) if not s.empty else float("nan")


def avg_histories(run_ids, keys, samples=500):
    """Fetch multiple seeds and return a single step-aligned, averaged DataFrame."""
    hists = []
    for rid in run_ids:
        print(f"  fetching {rid}…")
        hists.append(fetch_history(rid, keys, samples=samples))
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

# ── Plot 1: gradient smoothing (per-layer gradient contribution) ────────────────
def plot_grad_profile(synth_canon, synth_llama):
    layers = np.arange(N_LAYERS)
    llama_v = [trailing_mean(synth_llama, grad_key(l)) for l in range(N_LAYERS)]
    canon_v = [trailing_mean(synth_canon, grad_key(l)) for l in range(N_LAYERS)]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(layers, llama_v, "o-", color=C_LLAMA, label="Llama", lw=2, ms=7)
    ax.plot(layers, canon_v, "s-", color=C_CANON, label="Canon", lw=2, ms=7)
    ax.set_xlabel("layer"); ax.set_ylabel("gradient contribution (final)")
    ax.set_title("Canon smooths gradient flow: a flatter per-layer gradient\n"
                 "profile than Llama's (synthetic 8-task mix)")
    ax.legend(); ax.grid(True, alpha=0.3, ls="--"); ax.set_xticks(layers)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "grad_profile_synth.png")
    fig.savefig(path, dpi=150); plt.close(fig); print(f"Saved: {path}")


# ── Plot 2: representation upscaling (Canon conv output/input RMS ratio) ─────────
def plot_rms_ratio(synth_canon):
    layers = np.arange(N_LAYERS)
    fig, ax = plt.subplots(figsize=(7, 5))
    for ct, color, style in RMS_STYLES:
        vals = [trailing_mean(synth_canon, rms_key(ct, l)) for l in range(N_LAYERS)]
        ax.plot(layers, vals, style, color=color, label=f"Canon-{ct}", lw=2, ms=7)
    ax.axhline(1.0, color="gray", ls=":", lw=1.0, alpha=0.8)
    ax.set_xlabel("layer"); ax.set_ylabel("output/input RMS ratio (final)")
    ax.set_title("Canon amplifies the residual stream, most in deep layers\n"
                 "(synthetic 8-task mix)")
    ax.legend(); ax.grid(True, alpha=0.3, ls="--"); ax.set_xticks(layers)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "rms_ratio_synth.png")
    fig.savefig(path, dpi=150); plt.close(fig); print(f"Saved: {path}")


# ── Diagnostics ─────────────────────────────────────────────────────────────────
def print_grad_table(synth_canon, synth_llama):
    print(f"\n{'='*56}")
    print("Gradient contribution per layer (synthetic, trailing avg)")
    print(f"{'Layer':>6}  {'Llama':>10}  {'Canon':>10}")
    print(f"{'='*56}")
    lv = [trailing_mean(synth_llama, grad_key(l)) for l in range(N_LAYERS)]
    cv = [trailing_mean(synth_canon, grad_key(l)) for l in range(N_LAYERS)]
    for l in range(N_LAYERS):
        print(f"  L{l:>3}  {lv[l]:>10.4f}  {cv[l]:>10.4f}")
    print(f"\n  Deep/shallow (L7/L0)  Llama={lv[-1]/lv[0]:.2f}×  Canon={cv[-1]/cv[0]:.2f}×")


def print_rms_table(synth_canon):
    print(f"\n{'='*56}")
    print("Canon RMS ratio per layer (synthetic, trailing avg)")
    header = "  ".join(f"{'Canon-'+ct:>10}" for ct in RMS_TYPES)
    print(f"{'Layer':>6}  {header}")
    print(f"{'='*56}")
    for l in range(N_LAYERS):
        vals = "  ".join(f"{trailing_mean(synth_canon, rms_key(ct, l)):>10.4f}" for ct in RMS_TYPES)
        print(f"  L{l:>3}  {vals}")


# ── Main ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Synth Canon (averaged over 3 seeds) ===")
    synth_canon = avg_histories(SYNTH_CANON_IDS, GRAD_KEYS + RMS_KEYS, samples=500)
    print("=== Synth Llama (averaged over 3 seeds) ===")
    synth_llama = avg_histories(SYNTH_LLAMA_IDS, GRAD_KEYS, samples=500)

    print_grad_table(synth_canon, synth_llama)
    print_rms_table(synth_canon)

    plot_grad_profile(synth_canon, synth_llama)
    plot_rms_ratio(synth_canon)

    print("\nDone.")
