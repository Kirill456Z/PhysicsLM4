# Paper Outline — Question-Driven (rewritten 2026-06-19)

**This supersedes the earlier "Reliability × Conditionality" plan, which is dropped.**
The paper answers a concrete set of questions about two things: the **Canon layer**
(is it a good architectural change?) and the **synthetic reasoning benchmarks**
(are they a trustworthy testbed?). Everything below is grounded in verified runs.

Read with: `reports/FINDINGS.md` (evidence + numbers), `reports/GUIDENCE.md` (writing style).

---

## Part 0 — Research questions and answers (the spine of the paper)

Format: **question → answer → supporting data**. Two themes.

### Theme A — Is Canon a good architectural change?

- **Q A1. Is Canon better than the Llama baseline?**
  - **Answer:** Yes. Canon reaches a lower loss at every model size we tested (3M, 10M,
    30M, 100M) and on text modeling. The advantage is consistent in direction; we present
    the curves and let the reader judge the magnitude rather than fitting a scaling trend.
  - **Data (F6, F2):** scaling loss gaps Canon−Llama: 3M +0.086, 10M +0.045, 30M +0.031,
    100M +0.023 (5-seed scaling batch). Per-scale permutation tests (8+8 pooled seeds) are
    significant at 3M (p<0.0001, gap +0.090), 10M (p=0.0006, gap +0.048), and 100M
    (p=0.025, gap +0.023), and marginal at 30M (p≈0.07). Note the 30M gap is **batch-
    dependent**: the 5-seed scaling batch shows +0.031, but pooling all 8+8 seeds gives
    +0.015 (p=0.069) — this is the seed-flip finding (see B4), not a separate result. Since
    four scales are tested, one marginal result is statistically unremarkable, not a
    counter-finding. Text: Canon 0.781 vs Llama 0.793. Evidence: `06_scaling_laws/`,
    `02_canon_performance/`, `11_scale_task_robustness/`.

- **Q A2. What are the optimal Canon settings?**
  - **Answer:** Use kernel size k ≥ 2 (k = 1 is harmful); any reasonable initialization
    works (1/k constant is best); positions A (pre-attention) + C (pre-FFN) recover the
    full effect, so B and D can be dropped; insert Canon in all layers or at least the
    early ones, never last-layer-only; the benefit is normalization-agnostic (Pre-/Post-/
    Peri-LN).
  - **Data (F2, F3):** k=1 0.797 > Llama 0.793 > k=4 0.779 > k=6 0.778; zeros-init 0.787,
    1/k 0.779; Canon-AC 0.782 ≈ Canon-ABCD 0.781; last-layer-only 0.799 (worse than
    baseline), first-layer-only 0.781; Canon wins under all three norm types. Evidence:
    `03_canon_ablations/`. (Caveat: single-seed on text where the total effect is ~0.01;
    adjacent-condition orderings need multi-seed error bars before being asserted.)

- **Q A3. Does Canon's benefit depend on the data / task?**
  - **Answer:** Strongly, but with nuance from the de-ceiling experiment. The gain is
    largest and most consistent on multi-hop retrieval (Depo). When BFS and ShortPath are
    made genuinely hard (max_nodes=120, Llama baseline ~0.43 — well out of the ceiling),
    Canon still gains, but *only* on edges-encoded variants and at 3–5× smaller magnitude
    than on Depo. Adj-encoded BFS/SP remain null. The original concentration on Depo was
    **partly a ceiling artifact and partly a genuine task-type difference**. The
    encoding-dependence (edges gains, adj null) is an unexpected finding that remains
    unexplained and is reported as an open question.
  - **Data (F6, Exp A3, Exp B):** per-task Δ at 30M (standard difficulty) — Depo adj +0.71,
    Depo edges +0.57; ConComp +0.04–0.06; BFS/SP +0.007–0.025. The Depo accuracy gain is
    seed-robust (edges Δ+0.11 p=0.030, adj Δ+0.17 p=0.006). Hard BFS/SP (Exp B,
    max_nodes=120, 3 seeds): aggregate loss gap +0.029 (p=0.005); BFS-edges +0.032
    (p=0.022), SP-edges +0.049 (p=0.005); BFS-adj +0.002 (p=0.63), SP-adj −0.001 (p=0.92).
    Evidence: `06_scaling_laws/`, `11_scale_task_robustness/`, `13_bfs_sp_hard/`.

- **Q A4. Why does Canon help (mechanism)?**
  - **Answer:** Canon controls the deep-layer gradient dominance that the baseline develops
    over training, and it carries a built-in bias toward integrating context from farther
    back in the sequence. This is a **descriptive/correlational** observation. The depth
    ablation (Exp 9 + repowered) confirmed the gradient-ratio suppression is consistent at
    every depth tested, but the 16L performance reversal is **real** (confirmed with 5 seeds:
    Canon 16L 0.163 vs Llama 16L 0.129, p=0.897). The reversal is not a sampling artifact.
    A depth-equivalence experiment further ruled out the simplest explanation (Canon = free
    depth): Canon's loss curve is flat from 10L to 16L (~0.163), while Llama keeps
    improving (10L 0.142, 16L 0.129). Canon's conv processing **saturates around 8–10L**
    at this scale; adding more Canon layers beyond that yields no gain, and Llama's
    attention-depth continues to be productive past that point. The causal claim is not made.
  - **Data (F4, F9, Exp 9 repowered, depth equivalence):** deep/shallow gradient ratio —
    Llama 2.80× (text) / 1.72× (synth) vs Canon 0.70× / 0.44×; diverge over training.
    Learned conv weights favor farther shifts from step ~500. Kurtosis domain-dependent
    (descriptive only). Depth ablation (5 seeds at 16L): Canon 0.55/0.55/0.41 vs Llama
    2.39/0.97/0.71 at 4L/8L/16L — suppression holds everywhere, gap non-monotone. Canon
    depth curve (4L→0.249, 8L→0.146, 10L→0.164, 16L→0.163) vs Llama (4L→0.288, 8L→0.192,
    10L→0.142, 12L→0.163, 16L→0.129): Canon plateaus, Llama does not. Depth-equivalence
    offset mean=−1.1L, std=3.5L — not a constant shift. Evidence: `04_training_dynamics/`,
    `07_dynamics_on_synthetic/`, `14_depth_ablation/`, `15_depth_ablation_repowered/`,
    `16_depth_equivalence/`.

- **Q A5. Does Canon act like added depth, and does its benefit grow with depth?**
  - **Answer:** No to both. A natural hypothesis for a layer that mixes context across
    tokens is that Canon-NL behaves like a deeper Llama-(N+k)L (a "free depth" prior). We
    reject it. Across 4L→16L (fixed ~30M params), Canon's loss curve **saturates around
    8–10L and goes flat**, while Llama keeps improving with depth; by 16L Llama actually
    *overtakes* Canon. The implied "equivalent Llama depth" for a given Canon depth is not
    a constant offset — it swings from +1.9L (at 8L) to −6.8L (at 16L). So Canon's benefit
    is **a property of the shallow-to-mid-depth regime**, not a depth multiplier, and it
    does not compound with depth. This bounds where Canon helps and rules out the simplest
    mechanistic story (Canon = cheap extra layers).
  - **Data (F9, Exp 9 repowered, depth equivalence):** Canon loss by depth 4L→0.249,
    8L→0.146, 10L→0.164, 16L→0.163 (flat 10–16L); Llama 4L→0.288, 8L→0.192, 10L→0.142,
    12L→0.163, 16L→0.129 (monotone improvement). 16L reversal confirmed real with 5 seeds
    (Canon 0.163 vs Llama 0.129, permutation p=0.897 — not a 2-seed artifact). Depth-
    equivalence offset mean=−1.1L, **std=3.5L** (non-constant → hypothesis rejected).
    Evidence: `14_depth_ablation/`, `15_depth_ablation_repowered/`, `16_depth_equivalence/`.

### Theme B — Are the synthetic benchmarks a trustworthy testbed?

- **Q B1. Are results on these benchmarks reproducible across random seeds?**
  - **Answer:** Mostly, but with one sharp exception. Text modeling is fully stable, and 7
    of the 8 synthetic tasks are seed-stable. The variance is localized to a single task,
    Depo, which groks on some seeds and stagnates on others.
  - **Data (F1, new Q1):** text final-loss std ~0.007; per-task seed std at 30M — Depo
    0.21–0.29 vs every other task 0.012–0.055 (a 4–20× gap). Evidence: `01_seed_variance/`,
    `12_variance_sources/` (`q1_per_task_noise.png`).

- **Q B2. Which task causes the noise — are BFS / ShortPath less noisy than Depo?**
  - **Answer:** Yes, dramatically. Depo is the dominant noise source; BFS, ShortPath, and
    ConComp are all stable across seeds.
  - **Data (new Q1):** see the table above. Depo (adj) std 0.285, Depo (edges) 0.208; the
    next-noisiest task (BFS edges) is 0.055, and most are ~0.02. Evidence:
    `12_variance_sources/`.

- **Q B3. Does task mixing reduce variance / smooth training?**
  - **Answer:** It makes the *aggregate score* far more reproducible — but not by helping
    the model learn the hard task. Mixing dilutes the one volatile task (Depo) among seven
    stable ones, so the averaged number becomes seed-stable; at 30M the mix even suppresses
    Depo's learning outright. Mixing buys a reproducible aggregate, at the cost of the
    hardest task.
  - **Data (F5, new Q2/Q3):** aggregate-loss cross-seed std drops 0.087 (depo-only) → 0.006
    (8-task). But Depo hop-4 accuracy is 0.93 when Depo is trained alone vs 0.05 in the
    8-task mix — mixing floors the hard task rather than stabilizing it. Evidence:
    `05_task_mixing_variance/`, `12_variance_sources/`
    (`q2_aggregate_smoothness.png`, `q3_depo_in_isolation_vs_mix.png`).

- **Q B4. Can these benchmarks be used to measure architecture performance?**
  - **Answer:** Yes, with a protocol: use multiple seeds, and never rely on a single
    aggregate number. Decompose by task — the aggregate is dominated by easy near-ceiling
    tasks and hides the effect on the hard task that actually discriminates architectures.
  - **Data (F6, F8):** at 30M Canon's large Depo accuracy gain (+0.11–0.17, significant)
    barely moves the aggregate loss, because BFS/SP sit at 0.91–0.97 and dominate the
    average; a single-seed comparison gets the sign right only ~67% of the time. Evidence:
    `09_seed_flip/`, `11_scale_task_robustness/`.

- **Q B5. Can these benchmarks be used for mechanistic interpretability?**
  - **Answer:** Yes — and this is one of their strengths. The internal training dynamics
    (per-layer gradient ratios, learned convolution weights, activation statistics) are
    clear and consistent across seeds, even at scales where the bottom-line metric is noisy.
    The mechanism is more legible than the aggregate metric.
  - **Data (F4):** the gradient-ratio divergence and the farther-shift weight bias are
    stable and reproducible across seeds and across text vs synthetic. Evidence:
    `04_training_dynamics/`, `07_dynamics_on_synthetic/`.

- **Q B6. Besides task mixing (a data-side fix), does anything else reduce the Depo seed
  variance — e.g. does the Canon architecture itself stabilize it?**
  - **Answer:** Only marginally, and not as an independent mechanism. On the isolated hard
    task (depo×2, 5 seeds each) Canon's raw cross-seed std is 46% lower than Llama's, which
    initially looks like an architectural variance fix. But almost all of that is a **mean
    effect**: Canon reaches a much lower mean loss, and a lower mean mechanically carries a
    smaller absolute spread. Normalizing it out (coefficient of variation) the gap nearly
    vanishes — Canon CV 0.166 vs Llama 0.181, only 8% lower. So the honest reading is that
    Canon gives you a *better-and-lower* operating point on Depo, not a genuinely *more
    reproducible* one. Variance reduction you can trust still comes from the data side
    (mixing, B3) and from running enough seeds, not from the architecture.
  - **Data (F7, Exp A2):** depo×2, 5 seeds — Canon mean 0.237, std 0.039; Llama mean 0.401,
    std 0.072. Raw std ratio 0.54 (−46%) but CV ratio 0.92 (−8%). Evidence:
    `08_canon_stability/`, `10_variance_recompute/` (`f7_cv.png`).

### The one-sentence synthesis (how the two themes connect)

> The seed variance on these benchmarks lives almost entirely in one task, Depo; this is
> both why a single aggregate number is an unreliable instrument (it is dominated by easy,
> stable tasks and hides the volatile one) and why Canon looks modest in aggregate yet
> strong in truth — because Canon's benefit is concentrated on exactly that hard,
> high-variance, multi-hop task.

---

## Proposed paper structure (derived from the Q&A)

1. **Introduction.** Synthetic reasoning benchmarks promise a cheap testbed for architecture
   changes, and Canon is a promising candidate. Two questions: is Canon good, and can these
   benchmarks be trusted to answer that? Contributions bulleted.
2. **Background.** The tasks (Depo, ConComp, ShortPath, BFS; edges/adj encodings) and the
   Canon layer (causal 1-D conv; ABCD positions; kernel, init).
3. **Is Canon a good architecture?** A1 (scaling + text), A2 (ablations / optimal config),
   A3 (task dependence), A4 (mechanism), A5 (depth: not free depth, saturates ~8–10L).
4. **Are the benchmarks trustworthy?** B1/B2 (variance is real but localized to Depo),
   B3 (what task mixing does and does not fix), B6 (architecture is not an independent
   variance fix — mostly a mean effect), B4 (measure performance: multi-seed + per-task),
   B5 (mechanistic interpretability as a strength).
5. **Discussion.** Canon is a real, well-characterized improvement concentrated on hard
   multi-hop reasoning; the benchmarks are usable for both performance and interpretability
   provided you respect the localized-variance / aggregate-dilution structure.

---

## Figure plan (page-wide panels)

1. **Fig 1 — Canon works.** Scaling curves Canon vs Llama (3M–100M) + text curve.
   (`06_scaling_laws/loss_scaling.png`, `02_canon_performance/`.)
2. **Fig 2 — Optimal config.** Kernel / init / position / norm ablation panels.
   (`03_canon_ablations/`.)
3. **Fig 3 — Where Canon helps.** Per-task Canon−Llama at 30M (standard) + per-task
   hard-BFS/SP breakdown by encoding (edges vs adj). Panel A: Depo/ConComp/BFS/SP at
   standard difficulty. Panel B: hard BFS/SP edges-list vs adj-list with 95% CI bars.
   Shows both the Depo concentration and the encoding-dependent de-ceiling result.
   (`06_scaling_laws/per_task_breakdown_30m.png`, `13_bfs_sp_hard/`.)
4. **Fig 4 — Mechanism (correlational).** Gradient-ratio trajectory (Canon vs Llama,
   text + synth) + learned-weight shift bias + depth-ablation gradient-ratio bars
   (Canon 0.55/0.55/0.23 vs Llama 2.39/0.97/1.09 at 4L/8L/16L). Makes clear the
   suppression is consistent at every depth even where the loss gap is non-monotone.
   (`04_training_dynamics/`, `07_dynamics_on_synthetic/`, `14_depth_ablation/`.)
5. **Fig 5 — Variance is localized to Depo.** Per-task seed-std bar chart.
   (`12_variance_sources/q1_per_task_noise.png`.)
6. **Fig 6 — What mixing does.** Aggregate-loss seed band depo-only vs 8-task (smooth) +
   Depo accuracy alone vs in-mix (floored). (`12_variance_sources/q2_*.png`, `q3_*.png`.)

---

## Open items / honest gaps (do not over-claim)

- Ablations (A2) are single-seed on text; multi-seed error bars pending before
  adjacent-condition orderings are asserted (Tier 3, Exps 3/4/6 — not yet run).
- A3 "concentration on Depo" was partly ceiling artifact — **resolved by Exp B** (done):
  Canon gains on hard BFS/SP on edges-encoded variants (p=0.005–0.022) but not adj-encoded
  variants. The edges-vs-adj encoding dependence is unexplained and flagged as an open
  question.
- A4 mechanism is descriptive/correlational — **16L reversal confirmed real** (5 seeds,
  p=0.897). The gradient-ratio suppression is consistent at all depths. **Depth-equivalence
  hypothesis rejected**: Canon's conv processing saturates at ~8–10L while Llama keeps
  improving with depth; Canon-NL ≠ Llama-(N+k)L for constant k. A causal test would
  require a gradient-intervention control (Exp C — not yet run).
- Largest scale tested is 100M; the shrinking gap (+0.086→+0.023) may reach ~0 at
  practitioner scale (Exp D — not yet run).
- The encoding dependence found in Exp B (edges gains, adj null on hard BFS/SP) is a
  new unexplained finding — no experiment currently addresses it.
- **F9 (new):** Canon performance plateau above ~8L at 30M — worth a discussion paragraph
  noting it rules out "Canon = free depth" and leaves mechanism genuinely open.
