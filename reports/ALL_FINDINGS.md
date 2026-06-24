# All Findings — Consolidated

A single-place list of the interesting findings gathered from `RESEARCH_CHECKPOINT.md`,
`FINDINGS.md`, `QUESTIONS.md` (`PAPER_OUTLINE.md`), `MISSING_RUNS_AFTER_PIVOT.md`, the
slides, and the midterm report. Restructured around your comments: grouped into the results
sections they'd map to, framed around absolute effect sizes rather than significance tests.

Entity `kirill456z` · project `physics4llm`. Regenerate plots with
`WANDB_API_KEY=... uv run python reports/evidence/run_all.py`.

---

## A. Benchmark reliability (the methodological story)

1. **Synthetic tasks exhibit grokking to varying degrees, and grokking is very bad for reproducibility** (F1, `01_seed_variance/`, `12_variance_sources/`, midterm). Different tasks have different tendencies to grok: on grokking-prone tasks (Depo especially) the *same* architecture trained with different seeds gives wildly different curves — a sudden solve after a long plateau on some seeds, permanent stagnation on others — while text modeling and the stable tasks (BFS, ShortPath, ConComp) are reproducible.
2. **Task mixing smooths training and aggregate scores but does not remediate the variance on the volatile task** (F5, `05_task_mixing_variance/`, `10_variance_recompute/`). Mixing a grokking-prone task (Depo) with stable ones makes the aggregate score reproducible (cross-seed std 0.087→0.006), but Depo itself is not stabilized — its hop-4 accuracy goes from 0.93 trained alone to 0.05 in the 8-task mix. Mixing dilutes/suppresses the hard task rather than fixing its variance.

## B. Is Canon a good architecture? (performance)

1. **Canon improves text modeling and Canon-AC ≈ Canon-ABCD** (F2, `02_canon_performance/`). Final loss Canon 0.781 vs Llama 0.793; positions A (pre-attention) + C (pre-FFN) recover essentially the entire gain — B and D are droppable.
2. **Canon is better than Llama at every scale tested (3M–100M)** (F6, `06_scaling_laws/`). Absolute loss gap: +0.090 at 3M, +0.048 at 10M, ~+0.03 at 30M, +0.023 at 100M. The advantage is consistent in direction across all scales; the gap shrinks with size.
3. **Canon's benefit is concentrated on a specific task type — multi-hop retrieval (Depo) — and is encoding-dependent** (F6, Exp B, `06_scaling_laws/`, `13_bfs_sp_hard/`). Per-task Δ at 30M: Depo adj +0.71, Depo edges +0.57, vs ConComp +0.04–0.06 and BFS/SP near zero. On de-ceiled (hard) BFS/SP, Canon gains on edges-encoded variants but not adj-encoded ones — reported in results without accenting it.

## C. Canon ablations — optimal configuration

A single results section with subsections; together these identify the best Canon config.

1. **Kernel size: larger is better, 1 < 2 < 4 < 6** (F3, `03_canon_ablations/`). Cross-token mixing is the essential ingredient; performance improves monotonically with kernel size.
2. **Initialization: all schemes beat the baseline, including zero-init; 1/k constant is best** (F3, `03_canon_ablations/`). The architectural benefit isn't init-dependent — even zeros-init beats Llama.
3. **Layer placement: all-layers is best; first-layer-only captures most of it; last-layer-only is worse than baseline** (F3, `03_canon_ablations/`). Inserting Canon only after the final block hurts.
4. **Canon's benefit is normalization-agnostic** (F3, midterm, `03_canon_ablations/`). It beats the matched Llama baseline under Pre-LN, Post-LN, and Peri-LN.

## D. Mechanism (training dynamics — descriptive/correlational)

*Disclaimer on robustness: even where final performance varied substantially from run to
run, the directional mechanistic effects below stayed consistent across seeds. Some
internal statistics did fluctuate run-to-run without a clear correlation to performance,
but the core signatures (gradient-ratio suppression, long-range weight bias, RMS
amplification) were reproducible.*

1. **Canon suppresses the deep-layer gradient dominance the baseline develops over training** (F4, `04_training_dynamics/`, `07_dynamics_on_synthetic/`). Llama grows a 2.8× (text) / 1.72× (synth) deep/shallow gradient ratio; Canon holds ~0.7× / 0.44×. Both start ~0.5× at step 5k and diverge — a learned divergence, not an init effect. The suppression is consistent at every depth tested (Canon 0.55/0.55/0.41 vs Llama 2.39/0.97/0.71 at 4L/8L/16L), reinforcing it as a robust correlate (`14_depth_ablation/`).
2. **The universal pattern is gradient *suppression*, not uniformity** (pivot reframe). Canon's synthetic ratio (0.44×) is *further* from uniform (1.0×) than Llama's (1.72×); the shift is downward, not toward uniformity.
3. **Canon is structurally biased toward long-range context from initialization** (F4, `04_training_dynamics/`). Learned conv weights favor farther shifts (shift_2 > shift_0) from the very first checkpoint (~step 500); the pattern only sharpens over training.
4. **Canon amplifies representation magnitude, most strongly in deep layers** (midterm). Canon output/input RMS ratio > 1 for all types (strongest for A and C near the LM head), which may explain why Canon overtakes the baseline only later in training as the residual stream compounds.
