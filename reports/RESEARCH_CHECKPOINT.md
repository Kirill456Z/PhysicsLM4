# Research Checkpoint
**Session:** benchmark-variance-canon-evaluation  
**Repo:** `/Users/kirill.zemlianskii/WorkspacePersonal/PhysicsLM4`  
**WandB:** entity=`kirill456z`, project=`physics4llm`  
**Python env:** `uv run python ...` from repo root (pyproject.toml at root)  
**WandB API key:** `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`

---

## What This Project Is

A university lab research project investigating the **Canon layer** — an architectural
modification to transformers proposed in the paper *Physics of Language Models Part 4*
(Allen-Zhu & Li, Meta). The Canon layer is a learned 1D causal convolution inserted at
one or more positions within each transformer block:
- **A** — pre-attention
- **B** — post-attention  
- **C** — pre-FFN
- **D** — post-FFN

The original paper shows Canon improves transformer performance on synthetic reasoning
benchmarks but leaves key questions open. This project addresses them.

**Three open questions we address:**
1. *Mechanism*: Why does Canon work? What changes in training dynamics?
2. *Configuration*: Which Canon hyperparameters matter (kernel size, init, position, layer depth)?
3. *Scale + reliability*: Is the improvement consistent across model sizes, and how do we
   measure this without being fooled by benchmark seed variance?

---

## The Synthetic Benchmarks

Evaluation uses synthetic tasks from the Physics4LLM paper that each probe a specific
transformer capability. All tasks are graph/sequence reasoning problems:

| Task | What it tests | Metric |
|------|--------------|--------|
| Depo | Multi-hop knowledge retrieval (circular permutation) | hop_k/accuracy |
| Brevo | Reasoning breadth (BFS over a DAG) | loss/out |
| Concomp factor | Connected component factorization | prefix_accuracy |
| Shortest path | Shortest path on graphs | set_accuracy / is_correct_path |
| BFS | Breadth-first search | set_recall |

Each task has two encoding variants: `edges_list` and `adj_list`. The standard evaluation
setup used in the scaling law experiments is the **8-task mix** (all 5 tasks × both encodings
minus brevo, which is excluded due to extreme instability).

**Critical discovery we made:** These benchmarks have very high seed-to-seed variance —
the same architecture with different random seeds can produce wildly different training
curves (grokking on some seeds, permanent stagnation on others). Text modeling is stable.
Task mixing reduces the *aggregate* variance, but only partly fixes the problem: at 30M on
the 8-task mix the Canon-vs-Llama verdict is still **statistically null and seed-fragile**
(Exp A: gap +0.015, p=0.22, flips between seed batches). The real lesson is that
**aggregate single-scale numbers on these benchmarks are not trustworthy** — they hide
seed noise, ceiling effects (BFS/SP), and floor effects (Depo collapses in the 8-task mix).

---

## What Has Been Done

### Phase 1 — Pre-midterm (March–April)
Training on text modeling and early Depo/Brevo tasks. Results in `reports/main_midterm.tex`,
analyzed in notebooks `reports/march_09.ipynb`, `reports/match_16.ipynb`, `reports/march_22.ipynb`.

Key experiments completed:
- Demonstrated Depo/Brevo variance vs. text stability across seeds
- Canon-ABCD vs. Llama baseline on text modeling (main performance comparison)
- **Kernel size ablation** (k∈{1,2,4,6}) on text modeling
- **Initialization ablation** (zeros / default / const_var / const_var_sqrt) on text modeling
- **Position ablation** (AC vs. ABCD) on text modeling
- **Layer depth ablation** (all layers / first only / last only) on text modeling
- **Normalization type ablation** (Pre-LN / Post-LN / Peri-LN) on text modeling
- **Training dynamics** on Canon vs. Llama: learned canon weights, RMS ratio, gradient
  contributions per layer, activation kurtosis, cosine similarity between canon output and input

### Phase 2 — Post-midterm (April–June)
- **Task mixing variance reduction**: compared 1-task through 8-task mixes, 3 seeds each,
  to quantify how task diversity reduces seed variance
- **Scaling laws**: Canon vs. Llama at 3M, 10M, 30M, 100M parameters, 5 seeds each,
  trained on the 8-task mixed synthetic benchmark

### Phase 3 — Post-checkpoint (June 18–19)
- **Exp 1 — Dynamics on synthetic tasks**: 3 seeds each of 30M Canon-ABCD and Llama on
  the 8-task mixed benchmark, with full dynamics logging. Runs:
  `dynamics_on_synthetic_canon_s{1,2,3}_0.4.{196,197,198}`,
  `dynamics_on_synthetic_llama_s{1,2,3}_0.4.{199,200,201}`.
- **Exp 2 — Canon stability on depo-only**: 5 seeds each of 30M Canon-ABCD and Llama on
  depo×2 (depo_edges_list + depo_adj_list). Runs:
  `canon_stability_depo_canon_s{1..5}_0.4.{202..206}`,
  `canon_stability_depo_llama_s{1..5}_0.4.{207..211}`.
- **Exps 11, 12, 5, 7 — Free analyses (no new training)**: kurtosis temporal evolution
  (added to `07_dynamics_on_synthetic/`), per-task Canon improvement + kurtosis correlation
  (added to `07_dynamics_on_synthetic/`), gradient ratio and canon weight trajectories over
  time (added to `04_training_dynamics/`), per-task breakdown bar chart (added to
  `06_scaling_laws/`). All plots saved to their respective `plots/` subdirectories.

### Phase 5 — Depth follow-up (June 20–21)
- **Exp 9 repowered — 16L with 5 seeds** (`reports/evidence/15_depth_ablation_repowered/`):
  Added 3 new 16L seeds (Canon + Llama) to the original 2-seed depth ablation.
  **Result: 16L reversal is confirmed real.** Canon 16L mean=0.163 vs Llama 16L mean=0.129,
  gap=−0.034, permutation p=0.897. Not a sampling artifact. §A4 mechanism stays
  descriptive/correlational. Gradient-ratio suppression is consistent at every depth
  (Canon 0.55/0.55/0.41 vs Llama 2.39/0.97/0.71 at 4L/8L/16L).
- **Depth equivalence experiment** (`reports/evidence/16_depth_equivalence/`):
  Tested whether Canon-ABCD NL ≈ Llama-(N+k)L for constant k ("logical depth" hypothesis).
  New runs: Canon 4L/448D + Canon 10L/448D + Llama 10L/448D + Llama 12L/448D (3 seeds each).
  **Result: hypothesis rejected.** Equivalent Llama depths: Canon 4L→~5.6L (+1.6), Canon 8L→~9.9L
  (+1.9), Canon 10L→~9.1L (−0.9), Canon 16L→~9.2L (−6.8). The Canon loss curve is **flat
  from 10L to 16L** (~0.163–0.164) while the Llama curve keeps improving (10L 0.142, 16L 0.129).
  Canon's processing saturates around 8–10L; adding more Canon layers beyond that yields no
  gain. The 16L reversal is a saturation effect, not a simple "too much depth" story.
  New finding (**F9**): Canon has a performance plateau above ~8L at the 30M scale.

### Phase 4 — Post-pivot reanalysis (June 19)
Narrative pivoted to **Reliability × Conditionality** (see `reports/PAPER_OUTLINE.md`).
Three free reanalyses run and verified against wandb:
- **Exp A — Seed-flip audit** (`reports/evidence/09_seed_flip/`): pooled all 8 Canon + 8
  Llama 30M / 8-task seeds. **Aggregate verdict is NULL**: Canon 0.1752 vs Llama 0.1905,
  gap +0.0153 (~1.3σ), permutation p=0.22. The per-batch verdict flips (dynamics batch →
  Llama better; scaling batch → Canon better). Verdict-vs-seed-count: 1 seed → 67%
  "correct" sign, 8 seeds → only 91%; CI still includes 0. This is the Part 1 climax.
- **Exp A2 — Variance recompute** (`reports/evidence/10_variance_recompute/`):
  F7 raw std drops 46% (Canon 0.039 vs Llama 0.072) but **CV barely moves (0.166 vs 0.181,
  8%)** — the variance reduction is a mean effect, not an independent mechanism. F5: the
  fixed-task (Depo) seed std only collapses at 8 tasks because Depo accuracy **craters to
  0.045 (floor effect)**, not because training genuinely stabilises the hard task.
- **Exp A3 — Per-scale + per-task seed-robustness** (`reports/evidence/11_scale_task_robustness/`):
  Applied the Exp A permutation test at each scale (5 seeds each) and to Depo hop-4
  accuracy across the 30M 8+8 pool.
  - *Scale robustness:* significant at **3M** (p<0.0001, gap +0.090), **10M** (p=0.0006,
    gap +0.048), and **100M** (p=0.025, gap +0.023); null at 30M (p=0.069, gap +0.015,
    confirming Exp A).
  - *Depo accuracy at 30M:* edges_list Δ=+0.110 (p=0.030), adj_list Δ=+0.167 (p=0.006)
    — **both significant**. Canon's Depo gain is seed-robust even where the aggregate is
    null; aggregation hides the effect by mixing in ceiling tasks (BFS, SP).

---

## Key Findings

| ID | Finding | Key numbers (verified from wandb) |
|----|---------|----------------------------------|
| F1 | Synthetic benchmarks have high seed variance; text modeling is stable | Depo std=0.111, text std=0.007 across seeds |
| F2 | Canon improves text modeling; Canon-AC ≈ Canon-ABCD | Canon 0.781 vs Llama 0.793; AC=0.782 |
| F3 | k=1 is harmful; all inits beat baseline; last-layer-only can be worse than Llama | k=1:0.797 > Llama:0.793 > k=6:0.778; last-only:0.799 |
| F4 | Two separable mechanisms: universal gradient uniformity + domain-specific feature specialisation | Gradient ratio: text Canon 0.70×, synth Canon 0.44× (vs Llama 2.80×/1.72×); kurtosis: text ↓10×, synth ↑10× |
| F5 | Task mixing reduces *aggregate* seed variance; per-task it partly reflects a floor effect | agg std 0.087→0.011; but Depo-h4 std only collapses at 8-task because Depo acc craters to 0.045 |
| F6 | Canon scales better at small scale; gap shrinks with size and the 30M point is within noise | +0.086 at 3M, +0.023 at 100M; Depo +0.71 at 30M but 30M aggregate is null (see F8) |
| F7 | Canon's lower depo×2 variance is largely a mean effect, not an independent mechanism | raw std −46% (0.039 vs 0.072) but CV ≈ equal (0.166 vs 0.181, −8%) |
| **F8** | **At 30M on the 8-task mix the Canon-vs-Llama aggregate verdict is null and seed-fragile** | **gap +0.015 (~1.3σ), permutation p=0.22; flips by seed batch; needs >8 seeds for a stable sign** |
| **F9** | **Canon's performance plateaus above ~8L at 30M; the 16L reversal is a saturation effect, not a sampling artifact** | Canon 4L→0.249, 8L→0.146, 10L→0.164, 16L→0.163 (flat 10–16L); Llama 10L→0.142, 16L→0.129 (keeps improving); depth-equivalence offset non-constant (std=3.5L) |

**F4 full detail:**
- **Gradient uniformity (universal):** Both text and synthetic tasks. Canon holds gradient ratio at ~0.7× (text) / ~0.44× (synth) throughout training. Llama develops 2.8× (text) / 1.72× (synth) deep-layer dominance by mid-training. Neither model is uniform at step 1 (~0.5× for both); Llama rises steeply to ~2.86× by step 50k, Canon peaks at ~0.85× then decays to 0.68×.
- **Canon weight inductive bias (structural):** Farther-shift dominance (shift_2 > shift_0) present from the very first checkpoint (~step 500). Pattern sharpens slightly over training. Canon is structurally biased toward long-range integration from initialisation.
- **Feature specialisation on text (kurtosis suppression):** Llama L1 kurtosis = 157, Canon L1 = 18. Canon uniformly suppresses outlier features on text.
- **Feature specialisation on synthetic tasks (kurtosis elevation):** Canon kurtosis in deep layers L2–L7 is 8–12× *higher* than Llama (Canon L4 ≈ 14.7, Llama L4 ≈ 2.0). This is partly structural (small Canon > Llama gap from step 500) and mostly learned (gap grows 10× over training). The elevation reflects feature sharpening beneficial for discrete token discrimination. Tasks benefiting most (Depo +0.57–0.71) require exactly this — tasks near ceiling (BFS, SP) gain minimally.

---

## File Map

### Core documentation
| File | Purpose |
|------|---------|
| `reports/PAPER_OUTLINE.md` | **Canonical narrative** (Reliability × Conditionality) — read before drafting |
| `reports/FINDINGS.md` | Findings F1–F8 with wandb run names and verified numbers (F8 + F5/F7 reframes pending writeback) |
| `reports/MISSING_RUNS_AFTER_PIVOT.md` | **Canonical** missing-runs list, re-prioritized for the pivot |
| `reports/MISSING_EXPS.md` | Superseded for ordering; retains write-ups of completed Exps 1,2,5,7,11,12 |
| `reports/GUIDENCE.md` | Paper writing best practices — **read before writing any LaTeX** |
| `reports/RESEARCH_CHECKPOINT.md` | This file |
| `reports/main_midterm.tex` | Phase 1 midterm report (complete LaTeX, compiled to PDF) |

### Evidence system (the main artifact of Phase 2 analysis)
Each subfolder has a `run_data.py` that fetches wandb data, prints a verification summary
with actual numbers, and saves matplotlib plots to its `plots/` subdirectory.

| Path | Finding | What it produces |
|------|---------|-----------------|
| `reports/evidence/run_all.py` | — | **Entrypoint**: runs all 10 scripts |
| `reports/evidence/utils.py` | — | Shared wandb fetch/plot utilities |
| `reports/evidence/01_seed_variance/` | F1 | 5 plots: text stable, depo 8L noisy, depo 12L grokking, brevo noisy, comparison |
| `reports/evidence/02_canon_performance/` | F2 | 2 plots: canon vs llama, AC vs ABCD |
| `reports/evidence/03_canon_ablations/` | F3 | 4 plots: 2×2 panel + individual ablation plots |
| `reports/evidence/04_training_dynamics/` | F4 (text) | 5 plots: gradient uniformity, kurtosis, canon weights, RMS ratio, cosine similarity (text runs only) |
| `reports/evidence/05_task_mixing_variance/` | F5 | 3 plots: bar chart, std vs tasks, training curves |
| `reports/evidence/06_scaling_laws/` | F6 | 3 plots: loss scaling, per-task metrics, avg metric scaling |
| `reports/evidence/07_dynamics_on_synthetic/` | F4 (synth) | 5 plots: gradient/kurtosis 4-panel synth vs text, RMS ratio synth, canon weights synth |
| `reports/evidence/08_canon_stability/` | F7 | 3 plots: variance curves, bar chart, depo hop-4 accuracy |
| `reports/evidence/09_seed_flip/` | F8 | 3 plots: per-seed loss, batch flip, verdict-vs-seed-count (Exp A) |
| `reports/evidence/10_variance_recompute/` | F5/F7 | 2 plots: per-task variance (floor effect), CV bars (Exp A2) |
| `reports/evidence/11_scale_task_robustness/` | F6/F8 | 3 plots: scale p-values, verdict-vs-seedcount by scale, Depo accuracy robustness (Exp A3) |
| `reports/evidence/15_depth_ablation_repowered/` | F9 | 4 plots: loss comparison + gap + p-values (5 seeds at 16L), grad ratio, Depo accuracy, training curves |
| `reports/evidence/16_depth_equivalence/` | F9 | 3 plots: Canon vs Llama depth curves + equivalence arrows, depth offset bar chart, training curves |

Run the evidence system:
```bash
WANDB_API_KEY="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" \
  uv run python reports/evidence/run_all.py

# Run a single script
WANDB_API_KEY="..." uv run python reports/evidence/04_training_dynamics/run_data.py

# Run one finding via entrypoint (pass folder prefix)
WANDB_API_KEY="..." uv run python reports/evidence/run_all.py 06
```

### Notebooks (exploratory/historical, not the source of truth)
| File | Phase | Contents |
|------|-------|---------|
| `reports/march_09.ipynb` | 1 | Text stability, Depo variance, kernel size, init, dynamics |
| `reports/match_16.ipynb` | 1 | Layer ablation, norm type ablation |
| `reports/march_22.ipynb` | 1 | Brevo variance |
| `reports/april_20.ipynb` | 2 | LR sweep on mixed tasks, task mixing comparisons |
| `reports/evidence/run_vairance.ipynb` | 2 | Variance reduction analysis (now in `05_task_mixing_variance/`) |
| `reports/evidence/canon_scaling_laws.ipynb` | 2 | Scaling law analysis (now in `06_scaling_laws/`) |

### Configs
| Path | Purpose |
|------|---------|
| `recipe_stashes/` | 314 YAML configs for every run ever launched (one per codebase version) |

---

## Wandb Run ID Notes

Some runs were renamed in wandb so their display name differs from the internal run ID.
`api.run("entity/project/ID")` requires the internal ID, not the display name.
The evidence scripts already use the correct IDs. Key mismatches:

| Display name | Internal run ID |
|---|---|
| `text_llama_bs_128_seq_len_2048_0.1.130` | `canon_text_0.1.130` |
| `text_ks_4_default_init_with_residual_trainable_bs_128_seq_len_2048_0.1.129` | `canon_text_0.1.129` |
| `text_ks_1_default_init_with_residual_trainable_bs_128_seq_len_2048_0.1.117` | `canon_text_0.1.117` |
| `depo_ks_4_default_init_with_residual_trainable__8_hops_100_nodes_..._0.1.12X` | `depo_ks_4_default_init_with_residual_trainable_0.1.12X` |
| `depo_llama_8_hops_100_nodes_bs_256_seq_len_768_0.1.126` | `llama_0.1.126` |

To look up any run's internal ID by display name:
```python
api.runs('kirill456z/physics4llm', filters={'displayName': '<display_name>'}, per_page=1)[0].id
```

Most post-Phase-1 runs (scaling law, variance) have matching display name and internal ID.

---

## Final Goal

Write a short LaTeX research paper in `reports/main.tex`.
Target: as short as possible while complete (estimated 6–8 pages with figures).
**Always consult `reports/GUIDENCE.md` before writing LaTeX.**
**The full narrative outline lives in `reports/PAPER_OUTLINE.md` — read it before drafting.**

---

## Paper Narrative (PIVOTED June 19 — Reliability × Conditionality)

> Full section-by-section outline, figure plan, and findings→section map:
> **`reports/PAPER_OUTLINE.md`**. This is the canonical narrative; the summary below
> is the elevator version. The earlier "two mechanisms / Canon is great" framing is
> retired (see "What changed" at the end of this section).

### Thesis (one sentence)
Validating an architectural change on synthetic reasoning benchmarks is threatened in
two compounding ways — **seed variance corrupts the measurement** and **aggregation hides
where the effect lives** — and we show, using Canon as a running case study, how to get a
verdict you can trust and what that verdict turns out to be. Methodology-anchored paper,
Canon as the running example.

### Two parts, one paper (the bridge)
- **Part 1 — Can we trust the measurement?** Seed variance is real and task-specific
  (F1); task mixing shrinks the *aggregate* variance (F5) and Canon's lower depo×2 spread
  turns out to be mostly a mean effect (F7, CV ≈ equal). The climax (Exp A / F8): pooling
  **all 8+8 seeds**, the 30M / 8-task Canon-vs-Llama verdict is **statistically null**
  (gap +0.015, ~1.3σ, permutation p=0.22) and **flips by seed batch** (dynamics → Llama
  better; scaling → Canon better). Conclusion: a single aggregate number at a single scale
  is the wrong object of study — it hides seed noise, ceilings, and floors.
- **Part 2 — When and why does Canon help?** Since the 30M aggregate is null, the burden
  shifts to showing *where* a robust effect exists and *why*. Ablations on the stable
  signal (F2, F3); **scale-local + per-task evidence with seed-robustness checked at each
  scale** (F6 + Exp A3 + Exp B); mechanism = gradient-ratio control, depth-tested
  (F4-relabeled + Exp 9). The honest verdict: Canon's benefit is large and robust at
  *small scale* and on *Depo specifically*, and dissolves into noise in the 30M aggregate.

### Key reframes the data forces (verified, vs. FINDINGS.md as written)
- **NEW (F8): the 30M aggregate is null.** This is now the pivot's centrepiece. The
  mechanism runs (F4) live at 30M — where Canon does *not* measurably win in aggregate —
  so §4.3 must study "what Canon does to dynamics," not "why Canon wins at 30M."
- **"Canon reduces variance 46% (independent fix)" → mostly a mean effect.** Verified CV:
  Canon 0.166 vs Llama 0.181 (only 8% lower); raw std −46% follows the lower mean.
- **"Task mixing reduces variance 8×" → partly a floor effect per task.** Verified: Depo-h4
  seed std only collapses at 8 tasks because Depo accuracy craters to 0.045. Mixing
  stabilises the *aggregate*, sometimes by suppressing the hard task, not by solving it.
- **"Gains concentrate in Depo (mechanism)" → confounded with ceiling/floor.** Llama is at
  the floor on Depo, ceiling on BFS/SP; the aggregate loss is dominated by easy tasks, so
  a huge Depo accuracy gain coexists with a null aggregate. Exp B (de-ceiling) needed.
- **"Gradient uniformity" → "gradient-ratio control (suppresses deep-layer dominance)."**
  Canon's synth ratio 0.44× is *further* from uniform than Llama's 1.72×; the universal
  pattern is a downward shift, not uniformity.
- **Kurtosis "feature specialization mechanism" → domain-dependent descriptive
  observation.** Canon lowers kurtosis on text, raises it on synthetic; no causal claim.
- **"Canon consistently better 3M–100M" → conditionally better, small-scale-local.** Gap
  shrinks +0.086→+0.023 and is already null by 30M; the 30M scaling point is within noise.

### Implications
- **Methodological (primary contribution):** Physics4LLM-style benchmarks demand
  multi-seed + multi-task + multi-scale evaluation; report verdict stability vs seed count;
  never trust a single aggregate number — it hides seed noise, ceiling (BFS/SP) and floor
  (Depo-in-mix) effects. We demonstrate a verdict that *flips* under all three.
- **On Canon:** the benefit is real but conditional — robust at small scale and on
  multi-hop retrieval (Depo), null in the 30M aggregate, mechanistically associated with
  gradient-flow control (causality pending Exp 9).

### What changed from the pre-pivot narrative
- Reliability promoted from a supporting section to **Part 1 / the lead**.
- **The 30M aggregate "Canon wins" claim is retired** — verified null (F8). The paper no
  longer asserts Canon beats Llama at 30M; it asserts the verdict is unmeasurable there.
- "Two mechanisms" → **one mechanism** (gradient-ratio control) + one demoted descriptive
  observation (kurtosis).
- "Canon is consistently better" → **"Canon is conditionally better (small-scale / Depo)."**
- The seed-flip and the null aggregate are the paper's central evidence, not liabilities.
- Exp A / A2 **DONE** (verified). New experiments: **A3 (scale/per-task seed-robustness,
  reanalysis), B (de-ceiling), 9 (depth)** — see `reports/MISSING_RUNS_AFTER_PIVOT.md`.

---

## Why Missing Experiments Are Needed

**Canonical list: `reports/MISSING_RUNS_AFTER_PIVOT.md`** (re-prioritized for the pivot).
The older `reports/MISSING_EXPS.md` is superseded for ordering but retains the detailed
write-ups of completed Exps 1, 2, 5, 7, 11, 12.

**DONE (Phase 4, verified):**
- **A — Seed-flip audit** (`09_seed_flip/`): 30M aggregate null (p=0.22), flips by batch.
- **A2 — Variance recompute** (`10_variance_recompute/`): F7 CV ≈ equal; F5 floor effect.
- **A3 — Scale + per-task robustness** (`11_scale_task_robustness/`): significant at 3M
  (p<0.0001), 10M (p=0.0006), 100M (p=0.025); null at 30M aggregate (confirms A). Depo
  accuracy at 30M significant (edges p=0.030, adj p=0.006). Part 2 positive claims established.

**DONE (Phase 5, verified):**
- **Exp 9 repowered** (`15_depth_ablation_repowered/`): 16L reversal confirmed with 5 seeds
  (p=0.897, Canon 16L 0.163 vs Llama 16L 0.129). Not a sampling artifact.
- **Depth equivalence** (`16_depth_equivalence/`): logical-depth hypothesis rejected. Canon
  curve flat from 10L–16L (~0.163); depth offset non-constant (std=3.5L). New finding F9.

**All load-bearing experiments complete.** Completeness / reviewer-proofing only:
- **#3** multi-seed ablations on text — needed if adjacent ablation orderings are asserted.
- **#4** A/B/C/D positions — upgrades "AC≈ABCD" to individual position attribution.
- **#6** larger kernels — configs created June 20.
- **D** larger scale, **C** gradient intervention, **8** fixed-weight Canon — all optional.
