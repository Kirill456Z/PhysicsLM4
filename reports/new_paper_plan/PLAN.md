# Paper Plan — Canon, Horizontal Information Flow, and What Synthetic Benchmarks Are Good For

> This is a **writing plan**, not the paper. Each section says (a) what the prose argues
> and (b) exactly which experiment/plot to show. Keep the actual paper short (~6–8 pp).
> Lead framing (per decision): **Canon mechanism paper** — horizontal information flow is
> the headline; benchmark volatility is the motivation; the methodological lesson is the
> closing takeaway.
>
> **Direction notes (locked):**
> - **No seed-flip / null-verdict thread.** Canon is simply **better than Llama at every
>   scale** — show it with the scaling plot from `evidence/canon_scaling_laws.ipynb`.
> - **No statistical-significance machinery.** No p-values, permutation tests, σ. The loss
>   plots make the comparison obvious; we trust the plots.

---

## Thesis (one sentence)

Architecture research is expensive, so the field reaches for synthetic reasoning
benchmarks — but a *single* training run is an unreliable guide because some tasks grok
(the curve can jump at a random step), so you must look across seeds and scales; do that and
Canon is clearly better than the baseline everywhere — and, more usefully, the *internal*
mechanism it induces is reproducible run-to-run, which is where these benchmarks really earn
their keep: using Canon as a case study, that mechanism turns out to be **horizontal
(cross-token) information flow**, not the depth/representation reorganization it also
exhibits, which is only a correlate.

## Narrative arc (each beat necessitates the next)

1. Transformer architecture research is expensive → synthetic benchmarks are proposed as
   cheap proxies. **(Why we care.)**
2. But a single run is a poor guide: these tasks **grok**, and grokking-prone tasks (Depo,
   Brevo) jump at a random step, so one seed can rank two architectures the wrong way around;
   BFS, Shortest-Path and ConComp are stable. **(The pitfall in the proxy.)**
3. Look across seeds and scales and the picture is clean: **Canon reaches lower loss than the
   baseline at every model size (3M–100M).** **(The proxy still works if used correctly.)**
4. The more valuable use: the **mechanism** an architecture induces is reproducible across
   seeds even where individual curves wiggle. So we use the benchmarks to ask *how* Canon
   differs. **(The pivot to mechanism.)**
5. Canon shows two things at once: a **depth/representation reorganization signature** (it
   reshapes gradients and representation scale across layers, the way scaling a model up
   would) and **horizontal information flow** (its causal conv mixes tokens). We disentangle
   them. **(The scientific core.)**
6. The reorganization signature is a **correlate**: in a fixed-budget setting where Canon
   does *not* help (deep + narrow, L=16), the signature is still fully present. Horizontal
   flow is the **driver**: k=1 (no token mixing) is worse than the baseline, Canon's conv
   weights and cosine geometry point to farther tokens, and the gain is largest exactly
   where attention is scarce (shallow models). **(The verdict.)**
7. Meta-lesson: synthetic benchmarks are unreliable read one-seed-at-a-time, but valuable for
   *understanding* architectures — lead with mechanism. **(Why anyone should care beyond Canon.)**

---

## Section-by-section

### Title (pick at draft time)
- "Horizontal Information Flow Explains the Canon Layer"
- "What Canon Adds Is Cross-Token Mixing, Not Scaling: A Mechanistic Case Study on Synthetic Reasoning Benchmarks"
- "Read the Mechanism, Not the Single-Seed Score"

### Abstract (plain language, ~1 number max)
Architecture research is expensive; synthetic reasoning benchmarks are pitched as a cheap
substitute. We show a single training run is a poor guide, because some tasks *grok*: the
loss can suddenly drop at a random step, so one seed can rank two architectures the wrong way
around. Averaging over seeds and model sizes removes the ambiguity — the Canon layer reaches
lower loss than a strong baseline at every scale we test. More useful still, the internal
changes an architecture makes to training are reproducible even when the final score is
noisy. Using Canon as a case study, we use this to ask *why* it helps. Canon both reshapes
the model's gradients and representations across layers (resembling the effect of making the
model bigger) and mixes information across tokens. We show the first is only a side effect —
it persists in a setting where Canon stops helping — and the gain comes from cross-token
mixing. The broader lesson: don't trust a single-seed benchmark score, but do use these
benchmarks to study how architectures behave.

### 1. Introduction
- **Importance:** architecture iteration on transformers is gated by training cost; the
  Physics-of-LM synthetic benchmarks are proposed as controlled, cheap proxies.
- **Gap:** a single run can mislead — these tasks grok.
- **Objective:** evaluate correctly (across seeds/scales), then use the benchmarks for what
  they are best at: understanding mechanism. Canon is the running example.
- **Contributions (bulleted):**
  1. We show synthetic-benchmark training groks on some tasks (Depo, Brevo) and is stable on others (BFS/SP/ConComp), so single-seed convergence curves are unreliable for ranking.
  2. Evaluated across seeds and scales, Canon reaches lower loss than the baseline at every model size (3M–100M).
  3. The mechanism Canon induces is seed-stable; we disentangle two candidate explanations and show **horizontal information flow** drives the gain, while the depth/representation reorganization is a correlate.
  4. We argue synthetic benchmarks are unreliable read one-seed-at-a-time but valuable for mechanistic study.

### 2. Background and Setup
- Canon layer: learned 1D causal conv at positions A/B/C/D, kernel size $k$; Llama baseline.
- Tasks: Depo (multi-hop retrieval), BFS, Shortest-Path, ConComp; edges/adj encodings; metric.
- Training config (30M default = 8L/512D, k=4, ABCD, 80k steps, 8-task mix); scales 3M–100M.
- *No figure* (optionally a small hand-drawn Canon-block schematic — see Figure plan F0).

### 3. A single run is unreliable: these tasks grok (motivation)
**Argues:** the proxy can't be trusted one-seed-at-a-time, and the cause is grokking
concentrated in a couple of task types.
- **Plot A** text loss across seeds — near-identical curves (stable reference).
- **Plot B** **Depo 8L512D — grokking jumps at different steps per seed** (the midterm panel:
  3 Canon + Llama seeds). Grokking is visible; a Llama seed can beat a Canon seed purely by
  when it groks. *This is the motivation figure — single-seed ranking is unreliable.*
- **Plot C** Depo 12L768D and/or Brevo — same instability at another size/task; contrast with
  the stable tasks (BFS/SP/ConComp) in prose.
- → Figure 1.

### 4. Evaluated correctly, Canon is better at every scale
**Argues:** average over seeds and sweep scale and the comparison is unambiguous.
- **Plot D** scaling plot from `evidence/canon_scaling_laws.ipynb` — loss vs. params (log-log),
  Canon below Llama at 3M / 10M / 30M / 100M (5 seeds averaged per point).
- (Report version) per-scale loss-curve panels (Canon vs. Llama at each scale) as live panels.
- (Optional) per-task gain concentrates on Depo (multi-hop retrieval) — `06/per_task_breakdown_30m`.
- → Figure 2. No significance tests; the curves separate cleanly.

### 5. The mechanism is stable where the single-seed score is not (the pivot)
**Argues (short, ~1 paragraph, no new figure):** across seeds whose individual curves wiggle,
the directional internal signatures are reproducible (gradient redistribution, long-range
weight bias, RMS amplification). One or a few seeds already exposes *how* two architectures
differ — this is the benchmarks' real strength. Sets up Section 6.

### 6. What Canon does: reorganization signature vs. horizontal information flow (core)
**Argues:** Canon shows two things at once; we separate cause from correlate.

**6.1 Canon reshapes gradients and representations across layers (the "looks-like-scaling" signature).**
- **Plot E** gradient contribution per layer, Canon vs. Llama — Llama grows deep-layer
  gradient dominance over training; Canon keeps it flat/suppressed.
- **Plot F** Canon output/input RMS ratio — Canon amplifies the residual stream, most in deep
  layers near the head.
- Frame: these are the kind of changes that scaling a model up also produces → tempting to
  say "Canon buys you a bigger/deeper model." Two hypotheses: (H1) reorganization, (H2) flow.
- → Figure 3.

**6.2 The reorganization signature is correlational, not causal (the kill).**
- **Plot G** depth ablation at fixed budget — loss curves at L=16: Canon does *not* beat Llama
  there (deep + narrow). 
- **Plot H** gradient contribution at L=16 — the suppression signature is **still fully
  present** even though Canon no longer wins.
- The signature persists where the benefit disappears → it cannot be what drives the gain.
- (Discussion support) depth-equivalence rejected "Canon = free depth" — reinforces "not
  scaling." Mention in text.
- → Figure 4. *(Note: this is the fixed-budget depth tradeoff, a controlled mechanistic
  probe — distinct from the model-size scaling in §4 where Canon wins. Frame explicitly so
  the two don't read as contradictory.)*

**6.3 Horizontal information flow is the driver (the positive case).**
- **Plot I** kernel-size ablation — **k=1 (no cross-token mixing) is worse than the baseline**;
  loss improves monotonically with kernel size. Cross-token mixing is the necessary ingredient.
- **Plot J** learned conv weights favor farther-back tokens (shift weights: shift_2 ≳ shift_0)
  from the first checkpoint; cosine similarity (Canon output vs. input) is lowest in early
  layers and at A/C — the conv is doing long-range integration.
- **Plot K** depth ablation loss — the gain is **largest at shallow depth** (4L/8L) and gone
  at 16L. Interpretation: shallow models have few attention layers → little native horizontal
  flow → Canon's mixing fills the gap; deep models already integrate across tokens.
- → Figure 5.

### 7. Discussion / Implications
- **On Canon:** the gain is horizontal information flow; the cross-layer reorganization is a
  reproducible *correlate*, not the cause; benefit is largest where attention is scarce
  (shallow models) and on retrieval-heavy tasks (Depo).
- **Methodological (the meta-lesson):** synthetic benchmarks mislead read one-seed-at-a-time
  (grokking), but are excellent for *understanding* — average over seeds/scales for any
  ranking claim, and lead with mechanism.
- **Limitations / future work:** mechanism evidence is observational, not a direct
  intervention (propose a gradient-flow / fixed-weight Canon intervention); the
  edges-vs-adj encoding dependence is unexplained; depth-equivalence rules out the simplest
  "free depth" account.

---

## Figure plan (target 5 figures)

| Fig | Source | Message |
|----|--------|---------|
| F0 (opt.) | hand-drawn | Canon block + the two hypotheses (reorg vs. flow) |
| F1 | text-stable loss, Depo 8L512D grokking panel, Depo-12L/Brevo | Single-seed curves are unreliable; cause is grokking |
| F2 | `canon_scaling_laws.ipynb` loss-vs-params (+ per-scale panels) | Canon is lower-loss at every scale |
| F3 | grad-contrib per layer, RMS ratio (Canon vs Llama) | Canon's reorganization signature (looks like scaling) |
| F4 | depth L=16 loss + grad-contrib | Signature persists at 16L while Canon doesn't win → correlate |
| F5 | kernel-size, conv weights + cosine, shallow-depth gain | Horizontal flow is the driver |

## Findings → section map
- F1 (grokking / single-seed unreliable) → §3
- F6 scaling laws (Canon better at every scale) → §4
- mechanism stability → §5
- F4 (grad contribution, RMS, conv weights, cosine) → §6.1, §6.3
- depth ablation (16L: signature persists, no win; shallow gain) + depth-equivalence → §6.2, §6.3
- F3 (k=1 harmful, kernel monotone) → §6.3

## Dropped (per direction notes)
- **Seed-flip / 30M null verdict** (`09_seed_flip`) — removed entirely.
- **Per-scale/per-task permutation tests** (`11_scale_task_robustness`), **variance recompute /
  CV** (`10_variance_recompute`) — significance machinery, dropped.
- **BFS/SP de-ceiling** (`13_bfs_sp_hard`) — encoding-dependent; appendix at most.
- **Task-mixing variance / floor effect** (`05`, `12` q2/q3) — at most one motivating sentence.
- **Init / norm-type / A-B-C-D position ablations** — appendix "best config" note; main text keeps only kernel size.

## Report (`reports/report.ipynb`) — build notes
- Hosted `wr.Report` like `midterm_pres_repo.ipynb`: live `wr.LinePlot` panels via
  `RunsetsFactory` + `plots.py`; blocks in the §3→§6 order above.
- §4 scaling: render the `canon_scaling_laws.ipynb` matplotlib loss-vs-params plot inline in
  the notebook (faithful to source) **and** add per-scale live loss panels to the report.
- Reuse the midterm's "Depo 8L512D — grokking jumps at different steps per seed" panel for §3.
- Reuse the midterm dynamics panels (grad_contrib, canon_rms_ratio/residual_rms, canon
  weights, cos_sim) for §6.1/§6.3; add 4L/8L/16L depth loss + grad panels for §6.2/§6.3.
