# All Findings with Evidence

For each finding: the claim, the experiments that prove it, and the source documents it
was first articulated in. Run IDs are W&B internal IDs (`entity=kirill456z`,
`project=physics4llm`). Evidence scripts live under `reports/evidence/`.

---

## A. Benchmark reliability (the methodological story)

### A.1 Synthetic tasks exhibit grokking to varying degrees, and grokking is very bad for reproducibility

**Claim.** Different tasks have different tendencies to grok: on grokking-prone tasks
(Depo especially) the *same* architecture trained with different seeds gives wildly
different training curves — a sudden solve after a long plateau on some seeds, permanent
stagnation on others — while text modeling and the stable tasks (BFS, ShortPath, ConComp)
are reproducible. Per-task seed std at 30M: Depo 0.21–0.29 vs 0.012–0.055 for every
other task; text final-loss spread <0.002. Single runs are therefore unreliable proxies
and many seeds are needed.

#### Experiments that prove this

**Exp F1 — direct seed-variance comparison** (`reports/evidence/01_seed_variance/`)

The primary evidence. Plots five groups:

- *Text modeling (stable):* 4 seeds across Llama and Canon, showing near-identical
  loss curves (spread <0.002).
  - Runs: `text_llama_bs_128_seq_len_2048_0.1.136`,
    `text_ks_4_const_var_sqr_init_with_residual_trainable_bs_128_seq_len_2048_0.1.140`,
    `text_ks_4_const_var_sqr_init_with_residual_trainable_bs_128_seq_len_2048_0.1.147`,
    `canon_text_0.1.130`

- *Depo 8L/512D (noisy):* 3 Canon + 1 Llama seed diverging sharply.
  - Runs: `depo_ks_4_default_init_with_residual_trainable_0.1.120`,
    `depo_ks_4_default_init_with_residual_trainable_0.1.121`,
    `depo_ks_4_default_init_with_residual_trainable_0.1.122`,
    `llama_0.1.126`

- *Depo 12L/768D (grokking):* one Canon seed grokked after a long plateau; the other
  stagnated permanently — the clearest illustration of the phenomenon.
  - Runs: `depo_ks_4_default_init_with_residual_trainable_8_hops_100_nodes_bs_128_seq_len_1024_0.1.155`,
    `depo_ks_4_default_init_with_residual_trainable_8_hops_100_nodes_bs_128_seq_len_1024_0.1.156`

- *Brevo 110 nodes (noisy):* 3 Llama seeds diverging (same instability pattern as Depo).
  - Runs: `brevo_llama_110_nodes_single_bs_256_seq_len_1024_0.1.217`,
    `brevo_llama_110_nodes_single_bs_256_seq_len_1024_0.1.218`,
    `brevo_llama_110_nodes_single_bs_256_seq_len_1024_0.1.219`

Plots produced: `text_stable.png`, `depo_8l_noisy.png`, `depo_12l_grokking.png`,
`brevo_noisy.png`, `comparison_stable_vs_noisy.png`.

**Exp Q1 — per-task seed-std ranking at 30M** (`reports/evidence/12_variance_sources/`,
plot `q1_per_task_noise.png`)

Quantifies which task drives the aggregate noise. Uses the full 30M / 8-task pool
(5 Llama + 5 Canon seeds), computes the seed-to-seed std of each task's final primary
metric. Result: Depo (adj) std 0.285, Depo (edges) 0.208; the next noisiest task
(BFS edges) is 0.055; most tasks are ~0.02. Confirms that Depo is the dominant and
nearly exclusive noise source.

- Llama pool runs: `scaling_law_exps_3_llama_30m_0.4.83`,
  `depo_variance_impact_investigation_2_all_tasks_seed_55_0.4.120`,
  `depo_variance_impact_investigation_2_all_tasks_seed_56_0.4.121`,
  `scaling_law_4_more_seeds_llama_30m_seed_57_0.4.158`,
  `scaling_law_4_more_seeds_llama_30m_seed_58_0.4.159`
- Canon pool runs: `scaling_law_exps_3_canon_30m_0.4.79`,
  `remaining_runs_canon_30m_seed_55_0.4.128`,
  `remaining_runs_canon_30m_seed_56_0.4.129`,
  `scaling_law_4_more_seeds_canon_30m_seed_57_0.4.150`,
  `scaling_law_4_more_seeds_canon_30m_seed_58_0.4.151`

#### Key numbers (verified from W&B)

| Group | Seed std |
|---|---|
| Depo adj-list | 0.285 |
| Depo edges-list | 0.208 |
| BFS edges-list (next noisiest) | 0.055 |
| BFS adj / ConComp / ShortPath | 0.012–0.04 |
| Text modeling | <0.002 |

#### Source documents

- **`reports/FINDINGS.md` — F1** — canonical description with run IDs and key numbers.
- **`reports/march_09.ipynb`** (Phase 1) — earliest exploration of text stability vs Depo
  variance; kernel-size and initialization runs also analyzed here.
- **`reports/march_22.ipynb`** (Phase 1) — Brevo variance; three seeds shown diverging.
- **`reports/midterm_pres_repo.ipynb`** (Phase 1, midterm) — slides showing the
  grokking phenomenon (the Depo 12L plateau-then-solve curve); the instability was
  first formally presented here.
- **`reports/RESEARCH_CHECKPOINT.md`** — F1 row in the key-findings table; summary of
  the Phase 1 experiments.
- **`reports/QUESTIONS.md` Q B1 + Q B2** — question-driven framing: "which task causes
  the noise?" answered with the per-task std table.

---

### A.2 Task mixing smooths training and aggregate scores but does not remediate the variance on the volatile task

**Claim.** Mixing a grokking-prone task (Depo) with stable ones makes the aggregate
score reproducible (cross-seed std 0.087 → 0.006 as tasks go from 2 → 8), but Depo
itself is *not* stabilized — its hop-4 accuracy goes from 0.93 when trained alone to
0.05 in the 8-task mix. Mixing dilutes/suppresses the hard task rather than fixing
its variance.

#### Experiments that prove this

**Exp F5 — task-mixing sweep** (`reports/evidence/05_task_mixing_variance/`)

Five mixing levels (1 through 8 tasks), 3 seeds each (all 30M Llama, lr=3e-4). Measures
aggregate loss std as a function of task count. Showed the 8× aggregate-std drop
(0.087 → 0.011). Provides the raw data for the "mixing reduces variance" claim and the
later floor-effect reanalysis.

- 1 task (depo edges only): `seed_variance_exps_remaining_only_depo_edges_55_0.4.189`,
  `seed_variance_exps_remaining_only_depo_edges_56_0.4.190`,
  `seed_variance_exps_remaining_only_depo_edges_57_0.4.191`
- 2 tasks (depo ×2): `seed_variance_exps_remaining_only_depo_55_0.4.186`,
  `seed_variance_exps_remaining_only_depo_56_0.4.187`,
  `seed_variance_exps_remaining_only_depo_57_0.4.188`
- 4 tasks (all edges_list): `seed_variance_exps_remaining_only_edges_list_55_0.4.192`,
  `seed_variance_exps_remaining_only_edges_list_56_0.4.193`,
  `seed_variance_exps_remaining_only_edges_list_57_0.4.194`
- 6 tasks (no depo): `depo_variance_impact_investigation_2_no_depo_seed_54_0.4.122`,
  `depo_variance_impact_investigation_2_no_depo_seed_55_0.4.123`,
  `depo_variance_impact_investigation_2_no_depo_seed_56_0.4.124`
- 8 tasks: `depo_variance_impact_investigation_2_all_tasks_seed_55_0.4.120`,
  `depo_variance_impact_investigation_2_all_tasks_seed_56_0.4.121`,
  `scaling_law_4_more_seeds_llama_30m_seed_57_0.4.158`

**Exp Q2 + Q3 — floor-effect analysis** (`reports/evidence/12_variance_sources/`,
plots `q2_aggregate_smoothness.png` and `q3_depo_in_isolation_vs_mix.png`)

Uses the same depo-only and 8-task runs to answer two sharper questions:

- *Q2:* does mixing make the aggregate training trajectory smoother? Yes — the 8-task
  loss band is narrow and well-behaved; the depo-only band is wide and jumpy.
- *Q3:* does mixing stabilize *Depo's own performance*, or does Depo just crater to
  a floor? Depo (edges) hop-4 accuracy: 0.93 (depo-only) vs 0.045 (8-task mix).
  The variance collapses because the model reliably fails, not because mixing teaches
  Depo more stably.

**Exp A2 — variance recompute** (`reports/evidence/10_variance_recompute/`,
plots `f5_per_task_variance.png`, `f7_cv.png`)

A post-pivot reanalysis (completed June 19) that puts the honest framing in numbers:
tracks Depo (edges) hop-4 accuracy std across the four mixing levels that contain it
(1, 2, 4, 8 tasks). Confirms the aggregate-loss std drops are partly mechanical (more
tasks → mean is an average of more independent random variables) while the *fixed-task*
(Depo) std only "collapses" at 8 tasks because Depo accuracy craters to 0.045 — a floor
effect, not genuine stabilization.

Uses the same runs as F5 above; no new training required.

#### Key numbers (verified from W&B)

| # tasks | Agg loss mean | Agg loss std | Depo h4 acc (depo-only vs mix) |
|---|---|---|---|
| 1 (depo edges) | 0.243 | 0.045 | 0.93 (trained alone) |
| 2 (depo ×2) | 0.269 | 0.087 | — |
| 4 (edges only) | 0.199 | 0.014 | — |
| 8 (all tasks) | 0.194 | 0.011 | 0.045 (floored) |

Aggregate std drops 8× (0.087 → 0.011), but Depo accuracy simultaneously drops 20×
(0.93 → 0.045) — the variance is reduced by suppressing the hard task, not by
stabilizing it.

#### Source documents

- **`reports/FINDINGS.md` — F5** — original description (aggregate variance drop) and
  post-pivot reframe (floor-effect caveat added after Exp A2).
- **`reports/MISSING_RUNS_AFTER_PIVOT.md` — Exp A2** — write-up of the reanalysis,
  verified results, and narrative impact.
- **`reports/RESEARCH_CHECKPOINT.md`** — F5 row in the key-findings table; Exp A2
  summary under "Phase 4 — Post-pivot reanalysis."
- **`reports/QUESTIONS.md` Q B3** — question-driven framing: "does task mixing reduce
  variance / smooth training?" with the honest floor-effect answer.
- **`reports/april_20.ipynb`** (Phase 2) — exploratory analysis of task mixing and
  the LR sweep that preceded the systematic F5 sweep.
