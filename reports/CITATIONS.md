# Relevant Citations

Literature relevant to the paper in `reports/main.tex` (narrative in
`reports/PAPER_OUTLINE.md`). Compiled from a deep-research literature search.

**Verification status:** arXiv IDs marked ✅ were confirmed against arXiv directly.
IDs marked ⚠️ were drawn from standard knowledge and should be **verified before
camera-ready**. The Part 4.1 ID and author list are recent — confirm both.

---

## Tier 1 — Must cite (load-bearing)

| Key | Paper | Authors | Year | ID | Supports |
|-----|-------|---------|------|----|----|
| `allen_zhu_physics4` | Physics of Language Models: Part 4.1 — Architecture Design and the Magic of Canon Layers | Allen-Zhu (Z. Allen-Zhu; verify Li co-authorship) | 2025 | ⚠️ arXiv:2512.17351 | §1, §2.2 — object of study; fix the current bare cite |
| `allen_zhu_physics42` | Physics of Language Models: Part 4.2 — Canon Layers at Scale | Allen-Zhu (Meta) | 2025/26 | repo: github.com/facebookresearch/PhysicsLM4 (no arXiv surfaced) | §4.2, §5 — **counter-claim to "benefit dissolves by 30M"; must engage** |
| `reimers2017reporting` | Reporting Score Distributions Makes a Difference: Performance Study of LSTM-networks for Sequence Tagging | Reimers, Gurevych | 2017 (EMNLP) | ✅ arXiv:1707.09861 | §3.1/§3.3 — seminal: seed alone gives significant score swings |
| `dodge2019showyourwork` | Show Your Work: Improved Reporting of Experimental Results | Dodge, Gururangan, Card, Schwartz, Smith | 2019 (EMNLP) | ✅ arXiv:1909.03004 | §3.3 — closest precedent to verdict-vs-seed-count curve |
| `bouthillier2021variance` | Accounting for Variance in Machine Learning Benchmarks | Bouthillier et al. | 2021 (MLSys) | ✅ arXiv:2103.03098 | §3.2, §5 — variance decomposition / protocol |
| `picard2021seed` | Torch.manual_seed(3407) Is All You Need: On the Influence of Random Seeds… | Picard | 2021 | ✅ arXiv:2109.08203 | §3.3 — the seed-flip climax |
| `schaeffer2023mirage` | Are Emergent Abilities of Large Language Models a Mirage? | Schaeffer, Miranda, Koyejo | 2023 (NeurIPS) | ✅ arXiv:2304.15004 | §3.3/§4.2/§5 — metric/aggregation produces the verdict |
| `power2022grokking` | Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets | Power, Burda, Edwards, Babuschkin, Misra | 2022 | ✅ arXiv:2201.02177 | §3.1, Fig 1 — already cited |

---

## Tier 2 — Strong support

### Canon prior art (conv-in-transformer) — §2.2 / §4.1
| Key | Paper | Authors | Year | ID | Note |
|-----|-------|---------|------|----|----|
| `so2021primer` | Primer: Searching for Efficient Transformers for Language Modeling | So, Mańke, Liu, Dai, Shazeer, Le | 2021 | ✅ arXiv:2109.08668 | Depthwise conv inside the block — best "Canon is not new" anchor |
| `wu2019lightconv` | Pay Less Attention with Lightweight and Dynamic Convolutions | Wu, Fan, Baevski, Dauphin, Auli | 2019 (ICLR) | ✅ arXiv:1901.10430 | Depthwise weight-shared conv competitive with attention |
| `peng2023rwkv` | RWKV: Reinventing RNNs for the Transformer Era | Peng et al. | 2023 | ✅ arXiv:2305.13048 | Token-shift = Canon special case (k=2) |
| `fu2023h3` | Hungry Hungry Hippos (H3): Towards Language Modeling with State Space Models | Fu, Dao et al. | 2022/23 (ICLR) | ✅ arXiv:2212.14052 | Short causal conv before SSM mixing |
| `gu2023mamba` | Mamba: Linear-Time Sequence Modeling with Selective State Spaces | Gu, Dao | 2023 | ✅ arXiv:2312.00752 | Short causal conv before selective SSM |

### Gradient flow / signal propagation / normalization — §4.3 / §4.1
| Key | Paper | Authors | Year | ID | Note |
|-----|-------|---------|------|----|----|
| `wang2022deepnet` | DeepNet: Scaling Transformers to 1,000 Layers | Wang et al. | 2022 | ✅ arXiv:2203.00555 | Core prior art for deep-layer gradient-dominance story |
| `kim2025periln` | Peri-LN: Revisiting Layer Normalization in the Transformer Architecture | Kim et al. | 2025 | ✅ arXiv:2502.02732 | Directly matches norm-type ablation + gradient norms |

### Outlier features / activation kurtosis — §4.3
| Key | Paper | Authors | Year | ID | Note |
|-----|-------|---------|------|----|----|
| `dettmers2022llmint8` | LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale | Dettmers, Lewis, Belkada, Zettlemoyer | 2022 | ✅ arXiv:2208.07339 | Origin of "emergent outlier feature dimensions" |
| `sun2024massive` | Massive Activations in Large Language Models | Sun, Chen, Kolter, Liu | 2024 | ✅ arXiv:2402.17762 | Most on-point for activation-kurtosis observations (Fig 6c) |

### Scaling laws — §4.2
| Key | Paper | Authors | Year | ID | Note |
|-----|-------|---------|------|----|----|
| `kaplan2020scaling` | Scaling Laws for Neural Language Models | Kaplan et al. | 2020 | ✅ arXiv:2001.08361 | Frames "architectural delta shrinks with scale" |
| `hoffmann2022chinchilla` | Training Compute-Optimal Large Language Models (Chinchilla) | Hoffmann et al. | 2022 | ✅ arXiv:2203.15556 | Standard scaling companion |

### Grokking dynamics — §3.1 / §4.3
| Key | Paper | Authors | Year | ID | Note |
|-----|-------|---------|------|----|----|
| `nanda2023progress` | Progress Measures for Grokking via Mechanistic Interpretability | Nanda, Chan, Lieberum, Smith, Steinhardt | 2023 (ICLR) | ✅ arXiv:2301.05217 | Three-phase memorize→circuit→cleanup account |

---

## Tier 3 — Optional / appendix / freshness

| Key | Paper | Authors | Year | ID | Cluster |
|-----|-------|---------|------|----|----|
| `allenzhu2024knowcap` | Physics of LM Part 3.3 — Knowledge Capacity Scaling Laws | Allen-Zhu, Li | 2024 | ✅ arXiv:2404.05405 | Physics program / scaling |
| `allenzhu2024gsm` | Physics of LM Part 2.1 — Grade-School Math and Hidden Reasoning | Allen-Zhu, Li | 2024 | ✅ arXiv:2407.20311 | Synthetic-task methodology |
| `allenzhu2023cfg` | Physics of LM Part 1 — Learning Hierarchical Language Structures | Allen-Zhu, Li | 2023 | ✅ arXiv:2305.13673 | Synthetic-data program |
| `gulati2020conformer` | Conformer: Convolution-augmented Transformer for Speech Recognition | Gulati et al. | 2020 | ✅ arXiv:2005.08100 | Conv+attention hybrid |
| `poli2023hyena` | Hyena Hierarchy: Towards Larger Convolutional Language Models | Poli et al. | 2023 (ICML) | ✅ arXiv:2302.10866 | Short depthwise conv branch |
| `liu2022omnigrok` | Omnigrok: Grokking Beyond Algorithmic Data | Liu, Michaud, Tegmark | 2022 | ✅ arXiv:2210.01117 | Grokking beyond toy tasks |
| `colas2018seeds` | How Many Random Seeds? Statistical Power Analysis in Deep RL | Colas, Sigaud, Oudeyer | 2018 | ✅ arXiv:1806.08295 | Power analysis for seed count |
| `ulmer2022deepsig` | deep-significance: Statistical Significance Testing in the Age of Neural Networks | Ulmer, Hardmeier, Frellsen | 2022 | ✅ arXiv:2204.06815 | Tooling for permutation/significance tests |
| `henderson2017rlmatters` | Deep Reinforcement Learning That Matters | Henderson et al. | 2017 | ⚠️ arXiv:1709.06560 | Canonical seed-fragility / reproducibility |
| `xiong2020layernorm` | On Layer Normalization in the Transformer Architecture | Xiong et al. | 2020 (ICML) | ⚠️ arXiv:2002.04745 | Foundational Pre-LN vs Post-LN gradient analysis |
| `bondarenko2023quantizable` | Quantizable Transformers (outlier-free) | Bondarenko et al. | 2023 | ⚠️ arXiv:2306.12929 | Links attention behavior to outlier emergence |
| `liu2022understanding` | Towards Understanding Grokking: An Effective Theory of Representation Learning | Liu et al. | 2022 (NeurIPS) | ⚠️ arXiv:2205.10343 | Phase-diagram view of grokking |

---

## Density / thinness notes (for framing Related Work)

- **DENSE — cite carefully (derivative risk):** conv-in-transformer (Primer, LightConv,
  RWKV token-shift, H3/Mamba short-conv). Canon's depthwise scalar-shared causal conv is
  closely prefigured. Frame honestly in §2.2 — the contribution is the *verdict*, not the
  layer, so acknowledging lineage strengthens rather than weakens the paper.
- **DENSE — recent/noisy:** benchmark-pitfall preprints (2025–26). Cite the strong ones
  (Schaeffer, Bouthillier); avoid padding with surveys.
- **THIN — novelty opportunity (assert from data):**
  1. "Architectural deltas shrink with scale" — no clean seminal cite; lean on Kaplan/Chinchilla.
  2. "Aggregation over a synthetic task-mix hides ceiling/floor effects in the small-model
     regime" — essentially unclaimed; strongest novelty pocket.
  3. The seed-instability × aggregation combination applied to architecture validation on
     synthetic reasoning benchmarks — genuinely unoccupied; nearest neighbor is Schaeffer.

## Action items
- Replace the bare `allen_zhu_physics4` cite in `main.tex` with arXiv:2512.17351 (verify ID + authors).
- Add and engage Part 4.2 (Canon at Scale) as the counter-claim in §4.2/§5.
- Verify the four ⚠️ IDs before insertion.
