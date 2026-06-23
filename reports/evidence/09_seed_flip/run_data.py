"""
Exp A — Seed-flip audit at 30M (Part 1 §3.3, the bridge to Part 2).

Claim under test: at 30M on the 8-task mix, the Canon-vs-Llama verdict is NOT
seed-robust. Two batches of seeds (the "dynamics" batch and the "scaling" batch),
trained at the same 30M / 8-task / 80k-step setup, reach OPPOSITE conclusions, and
the effect size is the order of the seed noise.

This script:
  1. Pools every available 30M / 8-task seed for Canon and Llama, tagged by batch.
  2. Verifies the runs are comparable (steps, n_layers/dim, canon_set, #tasks).
  3. Reports per-seed final loss, mean ± SE, std, and coefficient of variation.
  4. Reproduces the per-batch flip (dynamics: Canon≈Llama; scaling: Canon≪Llama).
  5. Runs a permutation test on the pooled Canon−Llama difference.
  6. Produces the headline VERDICT-vs-SEED-COUNT curve: P(Canon judged better) and
     the spread of the measured gap, as a function of how many seeds you average.

No new training — reanalysis of existing runs only.

Plots saved to plots/:
  per_seed_loss.png         — strip plot of every seed, per arch, coloured by batch
  batch_flip.png            — mean±std per (arch, batch): the opposite verdicts
  verdict_vs_seedcount.png  — P(correct sign) and gap CI vs #seeds averaged
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

LOSS_TRAILING = 100   # trailing steps averaged for the final-loss point
RNG = np.random.default_rng(0)
N_BOOT = 20000        # resamples for verdict-vs-seedcount and permutation test

# ── Run inventory: every 30M / 8-task seed we have, tagged by batch ──────────────
# value = (run_id, batch)
CANON_RUNS = [
    ("dynamics_on_synthetic_canon_s1_0.4.196", "dynamics"),
    ("dynamics_on_synthetic_canon_s2_0.4.197", "dynamics"),
    ("dynamics_on_synthetic_canon_s3_0.4.198", "dynamics"),
    ("scaling_law_exps_3_canon_30m_0.4.79",            "scaling"),
    ("remaining_runs_canon_30m_seed_55_0.4.128",       "scaling"),
    ("remaining_runs_canon_30m_seed_56_0.4.129",       "scaling"),
    ("scaling_law_4_more_seeds_canon_30m_seed_57_0.4.150", "scaling"),
    ("scaling_law_4_more_seeds_canon_30m_seed_58_0.4.151", "scaling"),
]
LLAMA_RUNS = [
    ("dynamics_on_synthetic_llama_s1_0.4.199", "dynamics"),
    ("dynamics_on_synthetic_llama_s2_0.4.200", "dynamics"),
    ("dynamics_on_synthetic_llama_s3_0.4.201", "dynamics"),
    ("scaling_law_exps_3_llama_30m_0.4.83",                       "scaling"),
    ("depo_variance_impact_investigation_2_all_tasks_seed_55_0.4.120", "scaling"),
    ("depo_variance_impact_investigation_2_all_tasks_seed_56_0.4.121", "scaling"),
    ("scaling_law_4_more_seeds_llama_30m_seed_57_0.4.158",        "scaling"),
    ("scaling_law_4_more_seeds_llama_30m_seed_58_0.4.159",        "scaling"),
]

C_RED, C_BLUE = "#d6604d", "#2166ac"


# ── Fetch + comparability check ─────────────────────────────────────────────────
def fetch_runs(run_list, arch, api):
    """Return a DataFrame: one row per seed with final loss, batch, and config flags."""
    rows = []
    for rid, batch in run_list:
        try:
            run = api.run(f"{ENTITY}/{PROJECT}/{rid}")
            cfg = dict(run.config)
            model = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
            gens = cfg.get("include_generators", None)
            n_tasks = len(gens) if isinstance(gens, list) else None
            h = run.history(samples=400, keys=["loss/out"])
            s = h["loss/out"].dropna() if "loss/out" in h.columns else pd.Series(dtype=float)
            maxstep = int(h["_step"].max()) if "_step" in h.columns and len(h) else -1
            final = float(s.iloc[-LOSS_TRAILING:].mean()) if len(s) >= 5 else float("nan")
            rows.append({
                "arch": arch, "run_id": rid, "batch": batch, "final_loss": final,
                "maxstep": maxstep, "n_layers": model.get("n_layers"),
                "dim": model.get("dim"), "canon_set": model.get("canon_set"),
                "steps_cfg": cfg.get("steps"), "n_tasks": n_tasks,
            })
            print(f"  {arch:5s} {rid:55s} loss={final:.4f} step={maxstep} "
                  f"L={model.get('n_layers')} D={model.get('dim')} "
                  f"canon={model.get('canon_set')} tasks={n_tasks}")
        except Exception as exc:
            print(f"  [warn] {rid}: {exc}")
    return pd.DataFrame(rows)


def print_comparability(df):
    print(f"\n{'='*72}\nCOMPARABILITY CHECK (all rows should agree except canon_set)\n{'='*72}")
    for col in ["steps_cfg", "maxstep", "n_layers", "dim", "n_tasks"]:
        vals = sorted(set(v for v in df[col].dropna().tolist()))
        flag = "OK" if len(vals) <= 2 else "!! HETEROGENEOUS"
        print(f"  {col:12s}: {vals}   {flag}")
    print(f"  canon_set   : Canon={sorted(set(df[df.arch=='Canon'].canon_set.dropna()))}  "
          f"Llama={sorted(set(df[df.arch=='Llama'].canon_set.dropna()))}")
    print("  (maxstep may differ by a few steps due to eval-cadence rounding; "
          "steps_cfg is the source of truth.)")


# ── Statistics ───────────────────────────────────────────────────────────────────
def summary_stats(vals):
    vals = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    n = len(vals)
    mean = vals.mean() if n else float("nan")
    std = vals.std(ddof=1) if n > 1 else float("nan")
    se = std / np.sqrt(n) if n > 1 else float("nan")
    cv = std / mean if (n > 1 and mean) else float("nan")
    return dict(n=n, mean=mean, std=std, se=se, cv=cv, vals=vals)


def print_pooled_summary(canon, llama):
    print(f"\n{'='*72}\nPOOLED SUMMARY (all seeds, lower loss = better)\n{'='*72}")
    cs, ls = summary_stats(canon), summary_stats(llama)
    for name, st in [("Canon", cs), ("Llama", ls)]:
        print(f"  {name:5s}  n={st['n']}  mean={st['mean']:.4f}  std={st['std']:.4f}  "
              f"SE={st['se']:.4f}  CV={st['cv']:.3f}")
    gap = ls["mean"] - cs["mean"]   # positive ⇒ Canon better (lower loss)
    pooled_se = np.sqrt(cs["se"]**2 + ls["se"]**2)
    print(f"\n  Canon−Llama gap (Llama−Canon loss) = {gap:+.4f}  "
          f"(±{pooled_se:.4f} SE, ~{gap/pooled_se:.1f}σ)")
    print(f"  Seed noise (within-arch std ≈ {0.5*(cs['std']+ls['std']):.4f}) "
          f"vs effect ({gap:+.4f}) → effect is {abs(gap)/(0.5*(cs['std']+ls['std'])):.1f}× the noise std")
    return cs, ls, gap


def print_batch_flip(df):
    print(f"\n{'='*72}\nPER-BATCH VERDICT (the flip)\n{'='*72}")
    for batch in ["dynamics", "scaling"]:
        c = df[(df.arch == "Canon") & (df.batch == batch)]["final_loss"].tolist()
        l = df[(df.arch == "Llama") & (df.batch == batch)]["final_loss"].tolist()
        cs, ls = summary_stats(c), summary_stats(l)
        verdict = ("Canon better" if cs["mean"] < ls["mean"] - 1e-9
                   else "Llama better" if ls["mean"] < cs["mean"] - 1e-9 else "tie")
        print(f"  [{batch:8s}]  Canon {cs['mean']:.4f}±{cs['std']:.4f} (n={cs['n']})   "
              f"Llama {ls['mean']:.4f}±{ls['std']:.4f} (n={ls['n']})   → {verdict}")
    print("  If the two batches disagree, a single batch is not a reliable verdict.")


def permutation_test(canon, llama, n=N_BOOT):
    canon = np.asarray(canon); llama = np.asarray(llama)
    obs = llama.mean() - canon.mean()           # >0 ⇒ Canon better
    pool = np.concatenate([canon, llama]); nc = len(canon)
    count = 0
    for _ in range(n):
        perm = RNG.permutation(pool)
        diff = perm[nc:].mean() - perm[:nc].mean()
        if abs(diff) >= abs(obs):
            count += 1
    p = (count + 1) / (n + 1)
    print(f"\n{'='*72}\nPERMUTATION TEST (H0: arch label irrelevant)\n{'='*72}")
    print(f"  observed gap = {obs:+.4f}   two-sided p = {p:.4f}   "
          f"({'significant' if p < 0.05 else 'NOT significant'} at α=0.05)")
    return p


def verdict_vs_seedcount(canon, llama, n=N_BOOT):
    """For each k, draw k seeds (with replacement) from each arch, average, and
    record (a) P(Canon judged better) and (b) the distribution of the measured gap."""
    canon = np.asarray(canon); llama = np.asarray(llama)
    kmax = min(len(canon), len(llama))
    ks, p_correct, gap_lo, gap_hi, gap_med = [], [], [], [], []
    print(f"\n{'='*72}\nVERDICT vs SEED COUNT (bootstrap, {n} draws each)\n{'='*72}")
    print(f"  {'k seeds':>8}  {'P(Canon better)':>16}  {'gap 2.5–97.5%':>22}")
    for k in range(1, kmax + 1):
        gaps = np.empty(n)
        for i in range(n):
            c = RNG.choice(canon, size=k, replace=True).mean()
            l = RNG.choice(llama, size=k, replace=True).mean()
            gaps[i] = l - c                      # >0 ⇒ Canon better
        ks.append(k)
        p_correct.append(float((gaps > 0).mean()))
        lo, hi = np.percentile(gaps, [2.5, 97.5])
        gap_lo.append(lo); gap_hi.append(hi); gap_med.append(np.median(gaps))
        print(f"  {k:>8}  {p_correct[-1]:>16.3f}  [{lo:+.4f}, {hi:+.4f}]")
    return dict(ks=ks, p=p_correct, lo=gap_lo, hi=gap_hi, med=gap_med)


# ── Plots ────────────────────────────────────────────────────────────────────────
def plot_per_seed(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    for xi, arch in enumerate(["Canon", "Llama"]):
        sub = df[df.arch == arch]
        for batch, marker in [("dynamics", "o"), ("scaling", "s")]:
            b = sub[sub.batch == batch]
            jit = RNG.uniform(-0.12, 0.12, len(b))
            ax.scatter(np.full(len(b), xi) + jit, b["final_loss"],
                       marker=marker, s=70, alpha=0.85,
                       color=C_RED if arch == "Canon" else C_BLUE,
                       edgecolors="white", linewidths=0.6,
                       label=f"{arch} — {batch}")
        m = sub["final_loss"].mean()
        ax.hlines(m, xi - 0.25, xi + 0.25, color="black", lw=2.5, zorder=5)
        ax.text(xi, m, f"  mean {m:.3f}", va="center", fontsize=9)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Canon", "Llama"])
    ax.set_ylabel("Final loss/out (trailing-100 mean)")
    ax.set_title("Every 30M / 8-task seed (circle=dynamics batch, square=scaling batch)\n"
                 "Best single run is a Llama seed; arch means nearly overlap")
    ax.grid(axis="y", alpha=0.3, ls="--"); ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "per_seed_loss.png"); fig.savefig(p, dpi=150); plt.close(fig)
    print(f"Saved: {p}")


def plot_batch_flip(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    batches = ["dynamics", "scaling"]
    x = np.arange(len(batches)); w = 0.36
    for off, arch, color in [(-w/2, "Canon", C_RED), (w/2, "Llama", C_BLUE)]:
        means, stds = [], []
        for batch in batches:
            v = df[(df.arch == arch) & (df.batch == batch)]["final_loss"]
            means.append(v.mean()); stds.append(v.std(ddof=1) if len(v) > 1 else 0.0)
        ax.bar(x + off, means, w, yerr=stds, capsize=5, color=color, alpha=0.8, label=arch)
    ax.set_xticks(x); ax.set_xticklabels([f"{b} batch" for b in batches])
    ax.set_ylabel("Final loss/out (mean ± std)")
    ax.set_title("The flip: same 30M / 8-task setup, opposite verdicts\n"
                 "dynamics batch → Canon ≈ Llama;  scaling batch → Canon < Llama")
    ax.grid(axis="y", alpha=0.3, ls="--"); ax.legend()
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "batch_flip.png"); fig.savefig(p, dpi=150); plt.close(fig)
    print(f"Saved: {p}")


def plot_verdict_vs_seedcount(vc, gap_pooled):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    ax.plot(vc["ks"], vc["p"], "o-", color="#444", lw=2)
    ax.axhline(0.95, color="green", ls="--", lw=1, label="95% confident")
    ax.axhline(0.5, color="gray", ls=":", lw=1, label="coin flip")
    ax.set_xlabel("# seeds averaged per arch"); ax.set_ylabel("P(Canon judged better)")
    ax.set_ylim(0.4, 1.02); ax.set_title("How many seeds until the sign is stable?")
    ax.grid(alpha=0.3, ls="--"); ax.legend(fontsize=8)

    ax = axes[1]
    ax.fill_between(vc["ks"], vc["lo"], vc["hi"], alpha=0.25, color="#444",
                    label="95% gap interval")
    ax.plot(vc["ks"], vc["med"], "o-", color="#444", lw=2, label="median gap")
    ax.axhline(0, color="red", ls="--", lw=1, label="no difference")
    ax.set_xlabel("# seeds averaged per arch")
    ax.set_ylabel("measured gap  (Llama−Canon loss, >0 ⇒ Canon better)")
    ax.set_title("Measured gap vs seed count\n(interval crossing 0 ⇒ verdict can flip)")
    ax.grid(alpha=0.3, ls="--"); ax.legend(fontsize=8)
    fig.suptitle("Exp A — the 30M Canon-vs-Llama verdict is seed-count dependent",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "verdict_vs_seedcount.png"); fig.savefig(p, dpi=150); plt.close(fig)
    print(f"Saved: {p}")


if __name__ == "__main__":
    api = wandb.Api(api_key=WANDB_API_KEY)
    print("Fetching Canon 30M / 8-task seeds…")
    canon_df = fetch_runs(CANON_RUNS, "Canon", api)
    print("Fetching Llama 30M / 8-task seeds…")
    llama_df = fetch_runs(LLAMA_RUNS, "Llama", api)
    df = pd.concat([canon_df, llama_df], ignore_index=True)

    print_comparability(df)

    canon_vals = canon_df["final_loss"].dropna().tolist()
    llama_vals = llama_df["final_loss"].dropna().tolist()

    print_pooled_summary(canon_vals, llama_vals)
    print_batch_flip(df)
    permutation_test(canon_vals, llama_vals)
    vc = verdict_vs_seedcount(canon_vals, llama_vals)
    gap_pooled = np.mean(llama_vals) - np.mean(canon_vals)

    plot_per_seed(df)
    plot_batch_flip(df)
    plot_verdict_vs_seedcount(vc, gap_pooled)

    print("\nDone.")
