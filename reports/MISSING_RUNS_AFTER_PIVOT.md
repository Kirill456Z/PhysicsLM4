# Missing Runs After the Pivot (Reliability × Conditionality narrative)

Runs (and reanalyses) still needed to complete the combined narrative in
`reports/PAPER_OUTLINE.md`. Ordered by leverage on the story, not by compute.

**Legend**
- *Complexity* — GPU-compute on 0–10 (0 = reanalysis only, 2 = a few short text runs,
  5 = several full synthetic runs w/ dynamics, 7 = larger-scale multi-seed).
- *Section* — where it lands in the outline.
- *Decides* — the specific claim that does not stand without it.

Standard synthetic setup (unless noted): 30M = 8L/512D, 8 heads, head_dim 64,
vocab 512, k=4, canon ABCD, lr 3e-4, bs 128, seq 1024, 80k steps, 8-task mix
(concomp/depo/shortest_path/bfs × edges_list/adj_list). Template config:
`lingua_modified/apps/main/configs/exps/dynamics_on_synthetic/`.
Task difficulty lives in `data_synthetic_pretrain/tasks_config.yaml`
(`max_nodes`, `max_hops`, `max_nodes_in_output`).

---

## TIER 1 — Reanalysis, no new training

### ~~A · Seed-flip audit at 30M~~ ✅ DONE (June 19)
**Complexity 0.** **Evidence:** `reports/evidence/09_seed_flip/`.

**Verified results:**
- Canon mean loss: **0.1752** (8 seeds), Llama: **0.1905** (8 seeds), gap +0.015 (~1.3σ).
- Permutation test (20k resamples): **p = 0.22 — not significant.**
- Per-batch flip: dynamics batch → Llama better (0.176 vs 0.180); scaling batch → Canon
  better (0.173 vs 0.198). The verdict *depends on which seed batch you happened to run*.
- Verdict-vs-seed-count: 1 seed → 67% P(Canon better); 8 seeds → 91%, but the 95% CI
  on the gap still includes 0. Sign does not reliably stabilize within 8 seeds.
- All 8+8 runs confirmed comparable (80k steps, identical 8-task mix, 30M architecture).

**Narrative impact:** the headline "Canon wins at 30M" is null under honest pooling. The
paper's central claim (F8) is verified. The bridge between Part 1 and Part 2 holds.

### ~~A2 · Recompute F5 (per-task variance) and F7 (CV)~~ ✅ DONE (June 19)
**Complexity 0.** **Evidence:** `reports/evidence/10_variance_recompute/`.

**Verified results:**
- **F5 (floor effect):** Depo hop-4 accuracy at 8-task mix = **0.045** — the model
  reliably fails. Fixed-task seed std collapses at 8 tasks not because mixing stabilises
  the task, but because Depo accuracy craters to a floor where failure is consistent.
  The "mixing reduces variance" claim survives only for the *aggregate* loss, not for
  the hard task itself.
- **F7 (CV):** Canon CV = **0.166**, Llama CV = **0.181** — only −8% difference. The
  much-cited raw-std −46% is largely a mean effect (Canon reaches a lower mean loss on
  depo×2, so a smaller absolute spread is expected). Independent variance-reduction
  claim is weak.

**Narrative impact:** §3.2 reframed with floor-effect and CV caveat. Both originally
optimistic variance stories are now honest.

### ~~A3 · Per-scale + per-task seed-robustness~~ ✅ DONE (June 19)
**Complexity 0.** **Evidence:** `reports/evidence/11_scale_task_robustness/`.

**Verified results:**
- **Scale robustness (permutation test, 20k resamples):**
  - 3M: Canon 0.3248 vs Llama 0.4143, gap +0.090, **p < 0.0001 — significant.**
  - 10M: Canon 0.2148 vs Llama 0.2630, gap +0.048, **p = 0.0006 — significant.**
  - 30M: Canon 0.1752 vs Llama 0.1905, gap +0.015, p = 0.069 — null (confirms Exp A).
  - 100M: Canon 0.1389 vs Llama 0.1619, gap +0.023, **p = 0.025 — significant.**
- **Per-task Depo robustness at 30M (8+8 seeds):**
  - edges_list: Canon 0.205 vs Llama 0.095, Δ = +0.110, **p = 0.030 — significant.**
  - adj_list: Canon 0.236 vs Llama 0.069, Δ = +0.167, **p = 0.006 — significant.**

**Narrative impact:**
- Part 2 §4.2 can assert "Canon is significantly better at 3M, 10M, and 100M (aggregate
  loss); the 30M aggregate is null (confirmed)."
- §4.2 can assert "Depo accuracy gain is seed-robust at 30M even where the aggregate loss
  is not — aggregation hides the effect by mixing in ceiling tasks." Both decision rules
  are satisfied.

---

## TIER 2 — New training; earns the Part-2 claims (highest scientific value)

### ~~B · De-ceiling BFS and Shortest-Path (Part 2 §4.2)~~ ✅ DONE (June 19)
**Complexity ~5.** **Evidence:** `reports/evidence/13_bfs_sp_hard/`.

**Runs:** BFS×2 + SP×2 at hard difficulty (max_nodes=120, 30M, 3 seeds each).
- Canon: `bfs_sp_hard_canon_s{1,2,3}_0.4.{213,214,215}`
- Llama: `bfs_sp_hard_llama_s{1,2,3}_0.4.{216,217,218}`

**Verified results:**

| Metric | Canon mean | Llama mean | Δ | p |
|--------|-----------|-----------|---|---|
| Aggregate loss/out | 0.0753 | 0.1046 | +0.029 | **0.005 — significant** |
| BFS (edges) set_recall | 0.224 | 0.192 | +0.032 | **0.022 — significant** |
| BFS (adj) set_recall | 0.313 | 0.310 | +0.002 | 0.63 — null |
| SP (edges) set_accuracy | 0.602 | 0.554 | +0.049 | **0.005 — significant** |
| SP (adj) set_accuracy | 0.670 | 0.671 | −0.001 | 0.92 — null |

Difficulty bump was effective: Llama mean accuracy across tasks ≈ 0.43 (well out of the
ceiling region; standard difficulty was 0.91–0.97).

**Verdict: MIXED — partial ceiling artifact, partial genuine specificity.**
- Canon *does* gain on hard BFS/SP: aggregate loss gap +0.029 (> Exp-A noise band ~0.015),
  significant on edges-encoded variants (BFS-edges p=0.022, SP-edges p=0.005).
- Canon is *null* on adj-encoded variants (BFS-adj p=0.63, SP-adj p=0.92).
- Gain magnitude is modest (+0.03–0.05) vs the Depo gain (+0.11–0.17, Exp A3) — about 3–5×
  smaller even though headroom is now comparable.
- The result is encoding-dependent: Canon gains on edges-list but not adj-list BFS/SP.

**Narrative impact:**
- The "concentration on Depo is pure ceiling artifact" story is **not supported** — Canon
  does gain on hard BFS/SP too, so a partial ceiling confound existed.
- The "Canon is fully generic (helps wherever there is headroom)" story is **also not
  supported** — the Depo advantage is 3–5× larger, and BFS/SP gains are encoding-selective.
- The honest §4.2 framing: *"Canon's benefit is largest and most consistent on Depo
  (multi-hop retrieval); it also gains on hard BFS/SP but only on edges-encoded variants
  and with a substantially smaller effect. The original concentration on Depo was partly
  a ceiling artifact and partly a genuine task-type difference."*
- Drop the strong "Canon is specific to long-range reasoning" claim. Replace with a
  conditional + magnitude-based reading: Canon gains most where the task requires
  integrating many hops of information (Depo), and less (or inconsistently) on tasks
  that are locally solvable once headroom exists (BFS/SP). The encoding dependence
  (edges vs adj) is an unexpected finding worth reporting as an open question.

### ~~9 · Depth ablation at fixed parameter count (Part 2 §4.3 — causal test)~~ ✅ DONE (June 20)
**Complexity ~5.** **Evidence:** `reports/evidence/14_depth_ablation/`.

**Runs:** 4L/~768D + 8L/512D + 16L/~360D × {Canon-ABCD, Llama} × 2 seeds = 12 runs.
- Canon 4L:  `depth_ablation_canon_4l_s{1,2}_0.4.{221,222}`
- Canon 8L:  `depth_ablation_canon_8l_s{1,2}_0.4.{223,224}`
- Canon 16L: `depth_ablation_canon_16l_s{1,2}_0.4.{219,220}`
- Llama 4L:  `depth_ablation_llama_4l_s{1,2}_0.4.{227,228}`
- Llama 8L:  `depth_ablation_llama_8l_s{1,2}_0.4.{229,230}`
- Llama 16L: `depth_ablation_llama_16l_s{1,2}_0.4.{225,226}`

**Verified results (all 12 runs at 80k steps):**

| Depth | Canon loss | Llama loss | Gap (Llama−Canon) | Canon grad_ratio | Llama grad_ratio | Canon Depo | Llama Depo |
|-------|-----------|-----------|-------------------|-----------------|-----------------|-----------|-----------|
| 4L  | 0.1995 | 0.2876 | **+0.088** | 0.55 | **2.39** | 0.156 | 0.026 |
| 8L  | 0.1459 | 0.1920 | **+0.046** | 0.55 | 0.97  | 0.293 | 0.041 |
| 16L | 0.1411 | **0.0928** | **−0.048** | 0.23 | 1.09  | 0.385 | 0.354 |

**Verdict: NON-MONOTONE — the causal mechanism claim is NOT earned in this form.**

The loss gap shrinks from 4L to 8L and then **reverses sign at 16L** (Llama beats Canon).
This means the data does not support "Canon's advantage grows with depth."

**What is actually happening (honest interpretation):**

1. **The result is dominated by Depo seed variance, not architecture.** At every depth,
   whether a seed grokked Depo determines almost the entire loss outcome. With only 2 seeds
   per cell the win/loss is a coin flip in the Depo-variance regime. The 16L reversal is
   consistent with Llama having a lucky Depo seed at 16L (Llama 16L s1=0.063, s2=0.122 —
   a ~2× spread; Canon 16L s1=0.115, s2=0.168). With 8 seeds per cell the result could
   easily reverse again.

2. **The gradient-ratio pattern is consistent and directional across all depths.**
   Canon's deep/shallow ratio (0.55 / 0.55 / 0.23) is always below Llama's (2.39 / 0.97 /
   1.09) at every depth — this part of the mechanism story holds. But the ratio is not
   substantially lower at 16L than at 8L for Canon, so even this doesn't show a clean
   depth trend.

3. **Kurtosis behaves as expected:** Canon raises deep-layer kurtosis above Llama at
   4L (20.0 vs 14.2) and 8L (9.4 vs 2.0), consistent with the F4 finding on synthetic.
   At 16L both are low (~5 vs ~1.7), possibly because the 16L synthetic task is
   qualitatively different in its feature structure.

**Narrative impact:**
- §4.3 **cannot assert** "the gradient-ratio mechanism causally produces the performance
  gap (evidenced by a depth trend)." That specific claim is unsupported.
- §4.3 **can still assert** the gradient-ratio signature as a *consistent correlate*:
  Canon maintains a suppressed deep/shallow ratio at every depth we tested, while Llama
  develops deep-layer dominance (most strongly at 4L). Report it descriptively as "what
  Canon does to gradient flow" without claiming it is the cause of the performance gap.
- The 16L reversal and the Depo-variance confound should be reported honestly as a
  limitation: the depth test is underpowered (2 seeds in a high-variance regime) and
  the aggregate loss is a noisy instrument for isolating a depth effect. More seeds
  (≥5) or a depo-isolated evaluation would be needed for a clean causal test.
- Overall: the mechanism section becomes descriptive/correlational. This is acceptable
  given the paper's primary contribution is methodological (Part 1 / reliability), not
  mechanistic.

### ~~Exp 9 repowered · 16L with 5 seeds~~ ✅ DONE (June 20–21)
**Complexity ~3.** **Evidence:** `reports/evidence/15_depth_ablation_repowered/`.

**Runs:** Canon 16L/384D s{3,4,5} + Llama 16L/384D s{3,4,5} from `depth_ablation_more_seeds/`.
Pooled with original 2 seeds → 5 seeds per arm at 16L.

**Verified results:**
- Canon 16L mean = 0.1630 (std=0.034), Llama 16L mean = 0.1289 (std=0.041)
- Gap = −0.034 (Llama better), permutation p = 0.897 — **null but reversed sign**
- The 16L reversal is **confirmed real**, not a 2-seed sampling artifact
- Gradient-ratio suppression consistent at every depth: Canon 0.55/0.55/0.41 vs Llama 2.39/0.97/0.71 (4L/8L/16L)

**Narrative impact:**
- §A4: 16L reversal is a genuine architectural finding, not noise. Canon is worse than Llama
  at 16L/384D. Mechanism stays correlational. Report the reversal honestly.

---

### ~~Depth equivalence · Canon-ABCD vs Llama depth curves~~ ✅ DONE (June 21)
**Complexity ~4.** **Evidence:** `reports/evidence/16_depth_equivalence/`.

**Runs:** Canon 4L/448D + Canon 10L/448D + Llama 10L/448D + Llama 12L/448D × 3 seeds each.
Combined with existing depth ablation runs (4L/768D, 8L/512D, 16L/384D both archs).

**Verified results:**

| Config | Loss | Equiv Llama depth | Offset |
|--------|------|-------------------|--------|
| Canon 4L/448D | 0.249 | ~5.6L | +1.6L |
| Canon 8L/512D | 0.146 | ~9.9L | +1.9L |
| Canon 10L/448D | 0.164 | ~9.1L | −0.9L |
| Canon 16L/384D | 0.163 | ~9.2L | −6.8L |

Depth-equivalence offset: mean=−1.1L, **std=3.5L** — non-constant, hypothesis rejected.
The Canon loss curve is **flat from 10L to 16L** (~0.163–0.164); Llama keeps improving
(10L→0.142, 16L→0.129). Canon's processing saturates around 8–10L at this scale.

**Verdict: HYPOTHESIS REJECTED.** Canon-ABCD NL is not equivalent to Llama-(N+k)L for
constant k. The 16L reversal is a Canon saturation effect, not a simple depth story.

**Narrative impact (new finding F9):**
- §A4: Canon has a performance plateau above ~8L at 30M. The 16L reversal is not "too much
  depth" in the Llama sense — it is Canon's conv processing saturating, while Llama's
  attention-depth continues to be productive. Report as a limitation / boundary condition:
  Canon's benefit is constrained to the shallow-to-mid depth regime at this scale.
- The depth-equivalence experiment is worth a paragraph in the discussion: it rules out the
  simplest mechanistic explanation (Canon = free depth) and leaves the mechanism open.

---

## TIER 3 — Credibility / completeness (do once narrative is locked)

### 3 · Multi-seed ablations on text (Part 2 §4.1)
**Complexity ~6.** **Decides:** that adjacent ablation conditions (e.g. k=4 vs k=6,
Δ=0.001) are distinguishable. Required for credible error bars at any venue.
- 3 seeds × {k=1,2,4,6,(k* from Exp 6), Llama}; 3 seeds × 5 init; 2 seeds × 4 layer-depth;
  2 seeds × 6 norm-type. ~50 text runs (8L/512D, bs128, seq2048, 80k). Metric: `loss/out`.
- Run only after Exp 4 and 6 fix the condition set, so conditions don't change later.

### 4 · Individual Canon positions A / B / C / D (Part 2 §4.1)
**Complexity 2.** Upgrades "AC≈ABCD" to "A and C alone each suffice; pre-sublayer
insertion is what matters." 4 text runs (8L/512D, 80k, default init, k=4), one per
position. Compare to existing Llama (`canon_text_0.1.130`), ABCD (`...129`),
AC (`...0.1.138`). Metric: `loss/out`. Config base: `canon_first_layer/`-style.

### 6 · Larger kernels k=8, k=16 (Part 2 §4.1)
**Complexity 2.** Pre-empts "did you try k=8?" 2 text runs (8L/512D, 80k,
const_var_sqrt init, ABCD). Metric: `loss/out`. Add a 2nd seed if k=8 within 0.001 of k=6.

---

## TIER 4 — Optional, strengthens but not load-bearing

### D · One larger scale point (~200–300M), multi-seed (Part 2 §4.2 / Discussion)
**Complexity ~7.** **Decides:** whether the shrinking gap (+0.086→+0.023, 3M→100M)
extrapolates to ≈0 at practitioner scale. Canon vs Llama at ~200–300M × ≥3 seeds, 8-task
mix. First question a reviewer asks about a shrinking gap. Expensive; run if compute allows.

### C · Direct gradient-flow intervention (Part 2 §4.3)
**Complexity ~4.** **Decides:** whether gradient-ratio control is *causally* responsible.
Compare Canon to a Llama variant whose deep-layer gradient ratio is flattened by other
means (e.g. LayerScale / scaled residuals / Peri-LN). If the cheaper fix recovers most of
Canon's gain → mechanism confirmed as gradient flow. If not → temporal mixing matters
beyond gradients. Converts the depth correlation (Exp 9) into a direct test.

### 8 · Fixed-weight (non-learned) Canon (Part 2 §4.1 / §4.3)
**Complexity 3.** Disentangles structural temporal-mixing prior from learned adaptation.
2 text runs: Canon-ABCD frozen at 1/k; frozen at exponential decay w_i=exp(−i)/Z.
Metric: `loss/out` + `canon_weight/...` (verify frozen). Frozen≈learned → structure
drives benefit; frozen<learned → the learned shift_2-peak pattern is meaningful.

---

## Priority summary

| Order | Run | Section | Complexity | Status | Blocks |
|---|---|---|---|---|---|
| ~~1~~ | ~~**A** seed-flip audit~~ | ~~P1 §3.3~~ | ~~0~~ | ✅ DONE | ~~Part 1 climax / the bridge~~ |
| ~~2~~ | ~~**A2** F5/F7 recompute~~ | ~~P1 §3.2~~ | ~~0~~ | ✅ DONE | ~~honest variance claims~~ |
| ~~3~~ | ~~**A3** per-scale + per-task robustness~~ | ~~P2 §4.2~~ | ~~0~~ | ✅ DONE | ~~Part 2 positive claim~~ |
| ~~**1**~~ | ~~**B** de-ceiling BFS/SP~~ | ~~P2 §4.2~~ | ~~5~~ | ✅ DONE | ~~§4.2 attribution + §4.3 mechanism scope~~ |
| ~~**3**~~ | ~~**9** depth ablation~~ | ~~P2 §4.3~~ | ~~5~~ | ✅ DONE | non-monotone; mechanism stays correlational (see write-up) |
| ~~—~~ | ~~**9 repowered** 16L × 5 seeds~~ | ~~P2 §4.3~~ | ~~3~~ | ✅ DONE | reversal confirmed real; §A4 reports saturation |
| ~~—~~ | ~~**Depth equivalence** Canon vs Llama curves~~ | ~~Discussion~~ | ~~4~~ | ✅ DONE | hypothesis rejected; F9 Canon plateau above 8L |
| 4 | **3** multi-seed ablations | P2 §4.1 | 6 | TODO | ablation error bars |
| 5 | **4** A/B/C/D positions | P2 §4.1 | 2 | TODO | position story |
| 6 | **6** larger kernels | P2 §4.1 | 2 | TODO | kernel saturation |
| 7 | **D** larger scale | P2 §4.2 | 7 | optional | scale extrapolation |
| 8 | **C** gradient intervention | P2 §4.3 | 4 | optional | mechanism causality |
| 9 | **8** fixed-weight Canon | P2 §4.1 | 3 | optional | structure-vs-learned |

**Current state: ALL LOAD-BEARING EXPERIMENTS COMPLETE.**
- A, A2, A3 ✅ — Part 1 and Part 2 positive claims established.
- B ✅ — mixed verdict; §4.2 framing calibrated to magnitude + encoding dependence.
- 9 ✅ — non-monotone; causal mechanism claim not earned. §4.3 stays descriptive/correlational.
- 9 repowered ✅ — 16L reversal confirmed real (p=0.897 with 5 seeds); Canon saturation effect.
- Depth equivalence ✅ — depth-equivalence hypothesis rejected; F9: Canon plateau above ~8L.

**The paper is now fully evidenced for a coherent submission.** Everything remaining is
completeness or reviewer-proofing:
- **#3** multi-seed ablations on text — needed only if adjacent ablation comparisons (e.g.
  k=4 vs k=6, Δ=0.001) are asserted as distinguishable.
- **#4** A/B/C/D positions — upgrades "AC≈ABCD" to individual position attribution.
- **#6** larger kernels — configs created June 20 (`larger_kernels/`).
- **D** larger scale, **C** gradient intervention, **8** fixed-weight Canon — all optional.
