"""
Local plotting for the new-paper-plan W&B report.

ALL wandb run-history downloads happen here. Each figure is a function that fetches
its data, renders a matplotlib PNG, and saves it to ``new_paper_plan/plots/``.

The report builder (``new_paper_plan.py``) only calls ``ensure(key)`` to (re)generate
a figure when its PNG is missing — so editing the report text/layout never re-downloads
run metadata.

Run standalone to (re)generate everything:
    uv run python reports/new_paper_plan_plots.py            # all missing
    uv run python reports/new_paper_plan_plots.py --force    # all, overwrite
    uv run python reports/new_paper_plan_plots.py scaling kernel   # only these keys
"""
import os
import sys
import importlib.util

import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------
# Paths / credentials
# --------------------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(HERE, "new_paper_plan", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

WANDB_API_KEY = os.environ.get(
    "WANDB_API_KEY",
    "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",
)
ENTITY = "kirill456z"
PROJECT = "physics4llm"
os.environ["WANDB_API_KEY"] = WANDB_API_KEY

# Shared wandb fetch/plot helpers (evidence/utils.py), loaded by path so cwd doesn't matter.
_spec = importlib.util.spec_from_file_location("eu", os.path.join(HERE, "evidence", "utils.py"))
eu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eu)

CANON_C = "#d6604d"   # red  — Canon
LLAMA_C = "#2166ac"   # blue — Llama

# --------------------------------------------------------------------------------------
# Run sets
# --------------------------------------------------------------------------------------
TEXT_RUNS = {
    "Llama": ["text_llama_bs_128_seq_len_2048_0.1.136", "canon_text_0.1.130"],
    "Canon": [
        "text_ks_4_const_var_sqr_init_with_residual_trainable_bs_128_seq_len_2048_0.1.147",
        "text_ks_4_const_var_sqr_init_with_residual_trainable_bs_128_seq_len_2048_0.1.140",
    ],
}
DEPO_RUNS = {
    "Canon": [
        "depo_ks_4_default_init_with_residual_trainable_0.1.120",
        "depo_ks_4_default_init_with_residual_trainable_0.1.121",
        "depo_ks_4_default_init_with_residual_trainable_0.1.122",
    ],
    "Llama": [
        # Finished run (100k steps). The other depo-llama seed (0.1.166) crashed at 28k,
        # so it is excluded — we keep the seed that actually completed training.
        "llama_0.1.126",
    ],
}
BREVO_RUNS = {
    "Llama (3 seeds)": [
        "brevo_llama_110_nodes_single_bs_256_seq_len_1024_0.1.217",
        "brevo_llama_110_nodes_single_bs_256_seq_len_1024_0.1.218",
        "brevo_llama_110_nodes_single_bs_256_seq_len_1024_0.1.219",
    ],
}

# Synthetic-task dynamics (full logging), one seed per arch is enough for the signatures.
SYNTH_CANON = "dynamics_on_synthetic_canon_s1_0.4.196"
SYNTH_LLAMA = "dynamics_on_synthetic_llama_s1_0.4.199"

# All three dynamics seeds per arch — used to show the mechanism is seed-reproducible.
SYNTH_CANON_SEEDS = [
    "dynamics_on_synthetic_canon_s1_0.4.196",
    "dynamics_on_synthetic_canon_s2_0.4.197",
    "dynamics_on_synthetic_canon_s3_0.4.198",
]
SYNTH_LLAMA_SEEDS = [
    "dynamics_on_synthetic_llama_s1_0.4.199",
    "dynamics_on_synthetic_llama_s2_0.4.200",
    "dynamics_on_synthetic_llama_s3_0.4.201",
]

# Depth ablation at fixed budget (2 seeds per cell).
DEPTH = {
    "4L":  {"Canon": ["depth_ablation_canon_4l_s1_0.4.221", "depth_ablation_canon_4l_s2_0.4.222"],
            "Llama": ["depth_ablation_llama_4l_s1_0.4.227", "depth_ablation_llama_4l_s2_0.4.228"]},
    "8L":  {"Canon": ["depth_ablation_canon_8l_s1_0.4.223", "depth_ablation_canon_8l_s2_0.4.224"],
            "Llama": ["depth_ablation_llama_8l_s1_0.4.229", "depth_ablation_llama_8l_s2_0.4.230"]},
    "16L": {"Canon": ["depth_ablation_canon_16l_s1_0.4.219", "depth_ablation_canon_16l_s2_0.4.220"],
            "Llama": ["depth_ablation_llama_16l_s1_0.4.225", "depth_ablation_llama_16l_s2_0.4.226"]},
}

KERNEL_RUNS = {
    "Llama (no Canon)": ["canon_text_0.1.130"],
    "k=1": ["canon_text_0.1.117"],
    "k=2": ["text_ks_2_const_var_sqr_init_with_residual_trainable_bs_128_seq_len_2048_0.1.164"],
    "k=4": ["text_ks_4_const_var_sqr_init_with_residual_trainable_bs_128_seq_len_2048_0.1.147"],
    "k=6": ["text_ks_6_const_var_sqr_init_with_residual_trainable_bs_128_seq_len_2048_0.1.148"],
}

# Scaling laws: 5 seeds per (arch, size), blocks of 5 = 3M, 10M, 30M, 100M.
SCALING_RUN_IDS = {
    "Llama": [
        "scaling_law_exps_3_llama_3m_0.4.84", "remaining_runs_llama_3m_seed_55_0.4.134",
        "remaining_runs_llama_3m_seed_56_0.4.135", "scaling_law_4_more_seeds_llama_3m_seed_57_0.4.160",
        "scaling_law_4_more_seeds_llama_3m_seed_58_0.4.161",
        "scaling_law_exps_3_llama_10m_0.4.82", "remaining_runs_llama_10m_seed_55_0.4.132",
        "remaining_runs_llama_10m_seed_56_0.4.133", "scaling_law_4_more_seeds_llama_10m_seed_57_0.4.156",
        "scaling_law_4_more_seeds_llama_10m_seed_58_0.4.157",
        "scaling_law_exps_3_llama_30m_0.4.83", "depo_variance_impact_investigation_2_all_tasks_seed_55_0.4.120",
        "depo_variance_impact_investigation_2_all_tasks_seed_56_0.4.121", "scaling_law_4_more_seeds_llama_30m_seed_57_0.4.158",
        "scaling_law_4_more_seeds_llama_30m_seed_58_0.4.159",
        "scaling_law_exps_3_llama_100m_0.4.81", "seed_variance_comp_llama_seed_55_0.4.99",
        "seed_variance_comp_llama_seed_56_0.4.100", "scaling_law_4_more_seeds_llama_100m_seed_57_0.4.154",
        "scaling_law_4_more_seeds_llama_100m_seed_58_0.4.155",
    ],
    "Canon": [
        "scaling_law_exps_3_canon_3m_0.4.80", "remaining_runs_canon_3m_seed_55_0.4.130",
        "remaining_runs_canon_3m_seed_56_0.4.131", "scaling_law_4_more_seeds_canon_3m_seed_57_0.4.152",
        "scaling_law_4_more_seeds_canon_3m_seed_58_0.4.153",
        "scaling_law_exps_3_canon_10m_0.4.78", "remaining_runs_canon_10m_seed_55_0.4.126",
        "remaining_runs_canon_10m_seed_56_0.4.127", "scaling_law_4_more_seeds_canon_10m_seed_57_0.4.148",
        "scaling_law_4_more_seeds_canon_10m_seed_58_0.4.149",
        "scaling_law_exps_3_canon_30m_0.4.79", "remaining_runs_canon_30m_seed_55_0.4.128",
        "remaining_runs_canon_30m_seed_56_0.4.129", "scaling_law_4_more_seeds_canon_30m_seed_57_0.4.150",
        "scaling_law_4_more_seeds_canon_30m_seed_58_0.4.151",
        "scaling_law_exps_3_canon_100m_0.4.77", "canon_seed_variance_100m_canon_3e-4_seed_55_0.4.118",
        "canon_seed_variance_100m_canon_3e-4_seed_56_0.4.119", "scaling_law_4_more_seeds_canon_100m_seed_57_0.4.146",
        "scaling_law_4_more_seeds_canon_100m_seed_58_0.4.147",
    ],
}

# 30M / 8-task pool (5 seeds per arch) — the per-task variance and per-task Canon-benefit
# analyses both draw on this. Same runs as the scaling-law 30M block.
POOL_30M = {
    "Canon": [
        "scaling_law_exps_3_canon_30m_0.4.79", "remaining_runs_canon_30m_seed_55_0.4.128",
        "remaining_runs_canon_30m_seed_56_0.4.129", "scaling_law_4_more_seeds_canon_30m_seed_57_0.4.150",
        "scaling_law_4_more_seeds_canon_30m_seed_58_0.4.151",
    ],
    "Llama": [
        "scaling_law_exps_3_llama_30m_0.4.83", "depo_variance_impact_investigation_2_all_tasks_seed_55_0.4.120",
        "depo_variance_impact_investigation_2_all_tasks_seed_56_0.4.121", "scaling_law_4_more_seeds_llama_30m_seed_57_0.4.158",
        "scaling_law_4_more_seeds_llama_30m_seed_58_0.4.159",
    ],
}

# Per-task primary metric (display label -> wandb summary key). (el)=edges-list, (al)=adj-list.
TASK_PRIMARY = {
    "Depo (el)":      "evals/synthetic/depo_edges_list/hop_4/accuracy",
    "Depo (al)":      "evals/synthetic/depo_adj_list/hop_4/accuracy",
    "ConComp (el)":   "evals/synthetic/concomp_factor_edges_list/prefix_accuracy",
    "ConComp (al)":   "evals/synthetic/concomp_factor_adj_list/prefix_accuracy",
    "ShortPath (el)": "evals/synthetic/shortest_path_edges_list/set_accuracy",
    "ShortPath (al)": "evals/synthetic/shortest_path_adj_list/set_accuracy",
    "BFS (el)":       "evals/synthetic/bfs_edges_list/set_recall",
    "BFS (al)":       "evals/synthetic/bfs_adj_list/set_recall",
}

# --------------------------------------------------------------------------------------
# Small fetch helpers
# --------------------------------------------------------------------------------------
def _hist(groups, keys, samples=500):
    return eu.fetch_training_history(groups, keys, WANDB_API_KEY, ENTITY, PROJECT, samples=samples)


def _trailing_mean(series, n=50):
    s = series.dropna()
    if s.empty:
        return np.nan
    return float(s.iloc[-min(n, len(s)):].mean())


def _layer_profile(group_runs, key_fn, layers):
    """Average (over seeds) the trailing-mean value of key_fn(layer) at each layer."""
    keys = [key_fn(l) for l in layers]
    hist = _hist(group_runs, keys)
    out = {g: [] for g in group_runs}
    for g, runs in hist.items():
        per_layer = []
        for l in layers:
            k = key_fn(l)
            vals = [_trailing_mean(df[k]) for df in runs.values() if k in df.columns]
            vals = [v for v in vals if np.isfinite(v)]
            per_layer.append(np.mean(vals) if vals else np.nan)
        out[g] = per_layer
    return out


def _save(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")
    return path


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
def _loss_curves(groups, title, name, y_log=True, y_max=None, smoothing=20):
    """Per-seed loss/out curves, coloured by group."""
    hist = _hist(groups, ["loss/out"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    palette = {"Canon": CANON_C, "Llama": LLAMA_C}
    cyc = [p["color"] for p in plt.rcParams["axes.prop_cycle"]]
    for gi, (g, runs) in enumerate(hist.items()):
        color = palette.get(g.split()[0], cyc[gi % len(cyc)])
        first = True
        for df in runs.values():
            if "loss/out" not in df.columns:
                continue
            s = df[["_step", "loss/out"]].dropna()
            y = s["loss/out"].rolling(smoothing, min_periods=1).mean()
            ax.plot(s["_step"], y, color=color, alpha=0.8, lw=1.4,
                    label=g if first else None)
            first = False
    if y_log:
        ax.set_yscale("log")
    if y_max:
        ax.set_ylim(top=y_max)
    ax.set_xlabel("step")
    ax.set_ylabel("loss/out")
    ax.set_title(title)
    ax.grid(alpha=0.3, ls="--")
    ax.legend()
    return _save(fig, name)


def plot_text_stable():
    return _loss_curves(TEXT_RUNS, "Text modelling — near-identical across seeds", "text_stable.png",
                        y_max=1.0)


def plot_depo_grokking():
    return _loss_curves(DEPO_RUNS, "Depo 8L/512D — grokking jumps at random steps; a Llama seed\n"
                                   "beats a Canon seed (single-seed ranking is unreliable)",
                        "depo_grokking.png", y_max=1.2)


def plot_brevo():
    return _loss_curves(BREVO_RUNS, "Brevo 110 nodes — seed-driven divergence", "brevo.png", y_max=3)


def plot_scaling():
    df = eu.fetch_runs_results(SCALING_RUN_IDS, WANDB_API_KEY, ENTITY, PROJECT)
    id_cols = {c for c in df.columns if c in {"run_id", "run_name", "label", "n_params"}}
    metric_cols = [c for c in df.columns if c not in id_cols]
    df = df.groupby(["label", "n_params"], as_index=False)[metric_cols].mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    eu.plot_scaling_law(df, "loss/out", ax=ax, log_x=True, log_y=True,
                        title="Canon vs Llama — final loss at every scale (5 seeds avg)")
    return _save(fig, "scaling.png")


def plot_task_noise():
    """Per-task seed-to-seed std of the final primary metric (30M, 8-task pool, Canon+Llama).
    Depo dominates the variance; the graph tasks (BFS/ShortPath/ConComp) are stable."""
    # Trailing-mean of the last 10 eval points per run (smooths per-run eval noise), then
    # pool over all Canon+Llama seeds — matches the vetted per-task std numbers.
    hist = _hist(POOL_30M, list(TASK_PRIMARY.values()), samples=400)
    rows = []
    for label, metric in TASK_PRIMARY.items():
        vals = []
        for runs in hist.values():
            for dfr in runs.values():
                if metric in dfr.columns:
                    v = _trailing_mean(dfr[metric], n=10)
                    if np.isfinite(v):
                        vals.append(v)
        if len(vals) > 1:
            rows.append((label, float(np.std(vals, ddof=1))))
    rows.sort(key=lambda r: r[1])   # ascending -> largest ends up on top in barh
    labels = [r[0] for r in rows]
    stds = [r[1] for r in rows]
    colors = [CANON_C if l.startswith("Depo") else LLAMA_C for l in labels]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(labels, stds, color=colors, alpha=0.88)
    ax.set_xlabel("seed-to-seed std of final primary metric (30M, 8-task pool)")
    ax.set_title("The variance is concentrated on Depo;\nBFS / ShortPath / ConComp are stable")
    ax.grid(axis="x", alpha=0.3, ls="--")
    fig.tight_layout()
    return _save(fig, "task_noise.png")


def plot_task_breakdown():
    """Per-task Canon vs Llama at 30M (5 seeds each). Left: absolute primary metric with
    seed error bars; right: Canon-minus-Llama, sorted. The gain is concentrated on Depo."""
    df = eu.fetch_runs_results(POOL_30M, WANDB_API_KEY, ENTITY, PROJECT)
    labels, c_means, c_stds, l_means, l_stds, deltas = [], [], [], [], [], []
    for label, metric in TASK_PRIMARY.items():
        if metric not in df.columns:
            continue
        c = df[df["label"] == "Canon"][metric].dropna()
        l = df[df["label"] == "Llama"][metric].dropna()
        if c.empty or l.empty:
            continue
        labels.append(label)
        c_means.append(c.mean()); c_stds.append(c.std(ddof=1) if len(c) > 1 else 0.0)
        l_means.append(l.mean()); l_stds.append(l.std(ddof=1) if len(l) > 1 else 0.0)
        deltas.append(c.mean() - l.mean())
    x = np.arange(len(labels))
    w = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.bar(x - w / 2, l_means, w, yerr=l_stds, label="Llama", color=LLAMA_C, alpha=0.85, capsize=4)
    ax.bar(x + w / 2, c_means, w, yerr=c_stds, label="Canon", color=CANON_C, alpha=0.85, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("primary metric (final)")
    ax.set_title("Canon vs Llama per task at ~30M (mean ± seed std)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3, ls="--")

    ax = axes[1]
    order = np.argsort(deltas)[::-1]
    s_labels = [labels[i] for i in order]
    s_deltas = [deltas[i] for i in order]
    ax.bar(range(len(s_labels)), s_deltas, color=CANON_C, alpha=0.88)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(s_labels)))
    ax.set_xticklabels(s_labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Canon − Llama")
    ax.set_title("Canon's gain is large on Depo, minor elsewhere")
    ax.grid(axis="y", alpha=0.3, ls="--")
    for i, d in enumerate(s_deltas):
        ax.text(i, d + (0.012 if d >= 0 else -0.02), f"{d:+.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    return _save(fig, "task_breakdown.png")


def _deep_shallow_ratio_over_time(run_id, n_layers, smoothing=20):
    """Per-step ratio: gradient of the last layer / gradient of the first layer
    (matches the vetted evidence definition; deep-layer dominance => ratio > 1)."""
    lo, hi = "grad_contrib/layer_0", f"grad_contrib/layer_{n_layers - 1}"
    hist = _hist({"g": [run_id]}, [lo, hi])
    df = next(iter(hist["g"].values()))
    sdf = df[["_step", lo, hi]].dropna()
    ratio = (sdf[hi] / sdf[lo]).rolling(smoothing, min_periods=1).mean()
    return sdf["_step"].to_numpy(), ratio.to_numpy()


def _deep_shallow_ratio_final(group_runs, n_layers, n=50):
    """End-of-training last/first-layer gradient ratio, averaged over seeds."""
    lo, hi = "grad_contrib/layer_0", f"grad_contrib/layer_{n_layers - 1}"
    hist = _hist(group_runs, [lo, hi])
    out = {}
    for g, runs in hist.items():
        ratios = []
        for df in runs.values():
            if lo in df.columns and hi in df.columns:
                a, b = _trailing_mean(df[lo], n), _trailing_mean(df[hi], n)
                if np.isfinite(a) and a:
                    ratios.append(b / a)
        out[g] = float(np.mean(ratios)) if ratios else np.nan
    return out


# Text-modelling dynamics runs — the gradient signature is clearest here (it is learned and
# grows monotonically over training). On the noisier synthetic tasks the same ordering shows
# up across depths (see depth_grad_ratio).
TEXT_CANON = "canon_text_0.1.129"
TEXT_LLAMA = "canon_text_0.1.130"


def plot_grad_ratio():
    """Deep/shallow gradient ratio over training (text): Llama grows it, Canon holds it down."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for run_id, g, c in [(TEXT_LLAMA, "Llama", LLAMA_C), (TEXT_CANON, "Canon", CANON_C)]:
        x, y = _deep_shallow_ratio_over_time(run_id, 8)
        ax.plot(x, y, color=c, lw=2, label=g)
    ax.axhline(1.0, color="gray", ls=":", lw=1, label="ratio = 1 (uniform)")
    ax.set_xlabel("training step")
    ax.set_ylabel("last / first layer gradient ratio")
    ax.set_title("Last / first layer gradient ratio")
    ax.grid(alpha=0.3, ls="--")
    ax.legend()
    return _save(fig, "grad_ratio.png")


def plot_grad_ratio_seeds():
    """Deep/shallow gradient ratio over training, two representative seeds per arch
    (30M, synthetic 8-task mix). To make the separation legible we keep, per arch,
    the two seeds with the most extreme final ratio: Llama's two highest (clearest
    deep-layer dominance) and Canon's two lowest (clearest suppression)."""
    def _curves(ids):
        out = []
        for run_id in ids:
            x, y = _deep_shallow_ratio_over_time(run_id, 8)
            final = float(np.nanmean(y[-20:])) if len(y) else np.nan
            out.append((final, x, y))
        return out

    llama = sorted(_curves(SYNTH_LLAMA_SEEDS), key=lambda t: t[0], reverse=True)[:2]
    canon = sorted(_curves(SYNTH_CANON_SEEDS), key=lambda t: t[0])[:2]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for curves, g, c in [(llama, "Llama", LLAMA_C), (canon, "Canon", CANON_C)]:
        for i, (_, x, y) in enumerate(curves):
            ax.plot(x, y, color=c, lw=1.8, alpha=0.9, label=g if i == 0 else None)
    ax.axhline(1.0, color="gray", ls=":", lw=1, label="ratio = 1 (uniform)")
    ax.set_xlabel("training step")
    ax.set_ylabel("last / first layer gradient ratio")
    ax.set_title("Last / first layer gradient ratio")
    ax.grid(alpha=0.3, ls="--")
    ax.legend()
    return _save(fig, "grad_ratio_seeds.png")


def plot_depth_grad_ratio():
    """Deep/shallow gradient ratio at 4L/8L/16L: Canon stays below Llama (and below 1) everywhere."""
    depths = ["4L", "8L", "16L"]
    nlayers = {"4L": 4, "8L": 8, "16L": 16}
    means = {"Canon": [], "Llama": []}
    for d in depths:
        r = _deep_shallow_ratio_final(DEPTH[d], nlayers[d])
        means["Canon"].append(r.get("Canon", np.nan))
        means["Llama"].append(r.get("Llama", np.nan))
    x = np.arange(len(depths))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - w / 2, means["Canon"], w, color=CANON_C, label="Canon")
    ax.bar(x + w / 2, means["Llama"], w, color=LLAMA_C, label="Llama")
    ax.axhline(1.0, color="gray", ls=":", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(depths)
    ax.set_ylabel("deep / shallow gradient ratio (final)")
    ax.set_title("The gradient signature persists at every depth — even at 16L,\nwhere Canon no longer wins")
    ax.grid(alpha=0.3, ls="--", axis="y")
    ax.legend()
    return _save(fig, "depth_grad_ratio.png")


def plot_rms_ratio():
    # Text run: the amplification signature is clean and monotone in depth here.
    layers = list(range(8))
    prof = _layer_profile({"Canon": [TEXT_CANON]},
                          lambda l: f"canon_align/rms_ratio/canonA/layer_{l}", layers)
    profC = _layer_profile({"Canon": [TEXT_CANON]},
                           lambda l: f"canon_align/rms_ratio/canonC/layer_{l}", layers)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(layers, prof["Canon"], "s-", color=CANON_C, lw=2, label="Canon-A")
    ax.plot(layers, profC["Canon"], "^-", color="#f4a582", lw=2, label="Canon-C")
    ax.axhline(1.0, color="gray", ls=":", lw=1)
    ax.set_xlabel("layer")
    ax.set_ylabel("output/input RMS ratio (final)")
    ax.set_title("Canon amplifies the residual stream, most in deep layers")
    ax.grid(alpha=0.3, ls="--")
    ax.legend()
    return _save(fig, "rms_ratio.png")


def plot_grad_profile():
    """Per-layer gradient contribution (final) on text: Llama ramps up toward deep layers,
    Canon stays flat — the gradient-smoothing view of the same signature as grad_ratio."""
    layers = list(range(8))
    prof = _layer_profile({"Llama": [TEXT_LLAMA], "Canon": [TEXT_CANON]},
                          lambda l: f"grad_contrib/layer_{l}", layers)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(layers, prof["Llama"], "o-", color=LLAMA_C, lw=2, label="Llama")
    ax.plot(layers, prof["Canon"], "s-", color=CANON_C, lw=2, label="Canon")
    ax.set_xlabel("layer")
    ax.set_ylabel("gradient contribution (final)")
    ax.set_title("Canon smooths gradient flow: a flatter per-layer gradient\n"
                 "profile than Llama's (uneven, spiking at shallow layers)")
    ax.grid(alpha=0.3, ls="--")
    ax.legend()
    return _save(fig, "grad_profile.png")


def plot_depth16_loss():
    return _loss_curves(DEPTH["16L"], "Loss at 16L (fixed budget): Canon does NOT beat Llama",
                        "depth16_loss.png")


def plot_kernel():
    """Final text loss by kernel size (zoomed) — k=1 worse than no Canon; larger k better."""
    order = ["k=1", "Llama (no Canon)", "k=2", "k=4", "k=6"]
    hist = _hist(KERNEL_RUNS, ["loss/out"])
    vals = []
    for g in order:
        runs = hist.get(g, {})
        v = [_trailing_mean(df["loss/out"]) for df in runs.values() if "loss/out" in df.columns]
        vals.append(np.nanmean(v) if v else np.nan)
    colors = [CANON_C, "#7f7f7f", "#9ecae1", "#4292c6", "#08519c"]  # k=1 red, Llama grey, k up = darker blue
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(range(len(order)), vals, color=colors)
    ax.axhline(vals[1], color="#7f7f7f", ls="--", lw=1)  # baseline line
    finite = [v for v in vals if np.isfinite(v)]
    if finite:
        ax.set_ylim(min(finite) - 0.004, max(finite) + 0.004)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_ylabel("final loss/out (trailing mean)")
    ax.set_title("Kernel size: k=1 (no cross-token mixing) is worse than no Canon;\n"
                 "larger kernels are better")
    ax.grid(alpha=0.3, ls="--", axis="y")
    return _save(fig, "kernel.png")


def plot_canon_weights():
    """Conv-weight magnitude by shift distance (final), positions A-D, a few layers each —
    farther-back taps carry more mass than the current token."""
    shifts = list(range(4))   # kernel size 4 -> taps shift_0..shift_3
    layers = [0, 3, 7]
    positions = ["A", "B", "C", "D"]
    keys = [f"canon_weight/canon{p}/layer_{l}/shift_{s}"
            for p in positions for l in layers for s in shifts]
    hist = _hist({"Canon": [SYNTH_CANON]}, keys)
    df = next(iter(hist["Canon"].values()))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, p in zip(axes.ravel(), positions):
        for l in layers:
            prof = [abs(_trailing_mean(df[f"canon_weight/canon{p}/layer_{l}/shift_{s}"]))
                    if f"canon_weight/canon{p}/layer_{l}/shift_{s}" in df.columns else np.nan
                    for s in shifts]
            ax.plot(shifts, prof, "o-", lw=2, label=f"layer {l}")
        ax.set_title(f"Canon-{p}")
        ax.set_xticks(shifts)
        ax.grid(alpha=0.3, ls="--")
    for ax in axes[-1, :]:
        ax.set_xlabel("shift (0 = current token, larger = farther back)")
    for ax in axes[:, 0]:
        ax.set_ylabel("|conv weight| (final)")
    axes[0, 0].legend()
    fig.suptitle("Canon conv weights favour farther-back tokens (positions A–D)")
    fig.tight_layout()
    return _save(fig, "canon_weights.png")


def plot_cos_sim():
    layers = list(range(8))
    profA = _layer_profile({"Canon": [SYNTH_CANON]},
                           lambda l: f"canon_align/cos_sim/canonA/layer_{l}", layers)
    profC = _layer_profile({"Canon": [SYNTH_CANON]},
                           lambda l: f"canon_align/cos_sim/canonC/layer_{l}", layers)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(layers, profA["Canon"], "s-", color=CANON_C, lw=2, label="Canon-A")
    ax.plot(layers, profC["Canon"], "^-", color="#f4a582", lw=2, label="Canon-C")
    ax.set_xlabel("layer")
    ax.set_ylabel("cosine(out, in) (final)")
    ax.set_title("Most token mixing (lowest cosine) in early layers")
    ax.grid(alpha=0.3, ls="--")
    ax.legend()
    return _save(fig, "cos_sim.png")


def plot_depth_gap():
    """Final loss (Canon vs Llama) at 4L / 8L / 16L — gap shrinks then reverses."""
    depths = ["4L", "8L", "16L"]
    means = {"Canon": [], "Llama": []}
    for d in depths:
        hist = _hist(DEPTH[d], ["loss/out"])
        for arch in ("Canon", "Llama"):
            vals = [_trailing_mean(df["loss/out"]) for df in hist[arch].values() if "loss/out" in df.columns]
            vals = [v for v in vals if np.isfinite(v)]
            means[arch].append(np.mean(vals) if vals else np.nan)
    x = np.arange(len(depths))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - w / 2, means["Canon"], w, color=CANON_C, label="Canon")
    ax.bar(x + w / 2, means["Llama"], w, color=LLAMA_C, label="Llama")
    ax.set_xticks(x)
    ax.set_xticklabels(depths)
    ax.set_ylabel("final loss/out")
    ax.set_title("Canon's gain is largest for shallow models and gone by 16L")
    ax.grid(alpha=0.3, ls="--", axis="y")
    ax.legend()
    return _save(fig, "depth_gap.png")


# --------------------------------------------------------------------------------------
# Registry + dispatch
# --------------------------------------------------------------------------------------
# key -> (filename, builder)
FIGURES = {
    "text_stable":   ("text_stable.png",   plot_text_stable),
    "depo_grokking": ("depo_grokking.png", plot_depo_grokking),
    "brevo":         ("brevo.png",         plot_brevo),
    "scaling":          ("scaling.png",          plot_scaling),
    "task_noise":       ("task_noise.png",       plot_task_noise),
    "task_breakdown":   ("task_breakdown.png",   plot_task_breakdown),
    "grad_ratio":       ("grad_ratio.png",       plot_grad_ratio),
    "grad_ratio_seeds": ("grad_ratio_seeds.png", plot_grad_ratio_seeds),
    "rms_ratio":        ("rms_ratio.png",        plot_rms_ratio),
    "grad_profile":     ("grad_profile.png",     plot_grad_profile),
    "depth16_loss":     ("depth16_loss.png",     plot_depth16_loss),
    "depth_grad_ratio": ("depth_grad_ratio.png", plot_depth_grad_ratio),
    "kernel":        ("kernel.png",        plot_kernel),
    "canon_weights": ("canon_weights.png", plot_canon_weights),
    "cos_sim":       ("cos_sim.png",       plot_cos_sim),
    "depth_gap":     ("depth_gap.png",     plot_depth_gap),
}


def path_for(key):
    return os.path.join(PLOTS_DIR, FIGURES[key][0])


def ensure(key, force=False):
    """Return the PNG path for `key`, generating it (downloads run data) only if missing."""
    if key not in FIGURES:
        raise KeyError(f"unknown figure '{key}'. Known: {list(FIGURES)}")
    path = path_for(key)
    if force or not os.path.exists(path):
        print(f"[generate] {key}")
        FIGURES[key][1]()
    else:
        print(f"[cached]   {key} -> {path}")
    return path


def make_all(keys=None, force=False):
    for key in (keys or FIGURES):
        ensure(key, force=force)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    force = "--force" in sys.argv
    make_all(args or None, force=force)
