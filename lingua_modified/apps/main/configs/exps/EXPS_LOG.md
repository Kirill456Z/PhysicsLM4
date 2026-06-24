# Experiment Log — Canon vs Llama Scaling Law Study

**Goal:** Quantify the performance impact of the canon layer across model sizes via scaling law comparison. For each size, train canon and llama models at the optimal LR across 3 seeds, then compare averaged performance curves.

**WandB project:** `kirill456z/physics4llm`

---

## Model Size Definitions

| Label | dim | n_layers | n_heads |
|-------|-----|----------|---------|
| 3m    | 256 |  4       |  4      |
| 10m   | 384 |  6       |  6      |
| 30m   | 512 |  8       |  8      |
| 100m  | 768 | 12       | 12      |

---

## Experiment Groups

### lr_comp & lr_comp_2 — LR search, 30m model
Tested lr ∈ {5e-5, 1e-4, 3e-4, 1e-3} for both llama and canon 30m. **Best: 3e-4.**

| Run | Model | LR |
|-----|-------|----|
| `lr_comp_llama1e-3_0.4.52` | Llama 30m | 1e-3 |
| `lr_comp_llama3e-4_0.4.53` | Llama 30m | 3e-4 |
| `lr_comp_canon1e-3_0.4.50` | Canon 30m | 1e-3 |
| `lr_comp_canon3e-4_0.4.51` | Canon 30m | 3e-4 |
| `lr_comp_2_llama1e-4_0.4.56` | Llama 30m | 1e-4 |
| `lr_comp_2_llama5e-5_0.4.57` | Llama 30m | 5e-5 |
| `lr_comp_2_canon1e-4_0.4.54` | Canon 30m | 1e-4 |
| `lr_comp_2_canon5e-5_0.4.55` | Canon 30m | 5e-5 |

### scaling_law_exps_3 — Main scaling runs, all sizes, lr=3e-4, seed=54
| Run | Model | Note |
|-----|-------|------|
| `scaling_law_exps_3_llama_3m_0.4.84` | Llama 3m | |
| `scaling_law_exps_3_llama_10m_0.4.82` | Llama 10m | |
| `scaling_law_exps_3_llama_30m_0.4.83` | Llama 30m | model.seed=52 |
| `scaling_law_exps_3_llama_100m_0.4.81` | Llama 100m | |
| `scaling_law_exps_3_canon_3m_0.4.80` | Canon 3m | |
| `scaling_law_exps_3_canon_10m_0.4.78` | Canon 10m | |
| `scaling_law_exps_3_canon_100m_0.4.77` | Canon 100m | lr_min=3e-5 |
| `scaling_law_exps_3_canon_30m_0.4.79` | Canon 30m | seed=52 |

### lr_sweeps_100m — LR search, llama 100m. **Best: 3e-4.**
`lr_sweeps_100m_llama_1e-4_0.4.95`, `lr_sweeps_100m_llama_3e-4_0.4.96`, `lr_sweeps_100m_llama_3e-5_0.4.97`, `lr_sweeps_100m_llama_9e-4_0.4.98`

### canon_lr_and_seed_variance_comp — LR search, canon 100m. **Best: 3e-4.**
`canon_lr_and_seed_variance_comp_canon_1e-4_0.4.102`, `canon_lr_and_seed_variance_comp_canon_3e-4_0.4.103`, `canon_lr_and_seed_variance_comp_canon_3e-5_0.4.104`, `canon_lr_and_seed_variance_comp_canon_9e-4_0.4.105`

### seed_variance_comp — Extra seeds, llama 100m, lr=3e-4
`seed_variance_comp_llama_seed_55_0.4.99`, `seed_variance_comp_llama_seed_56_0.4.100`

### canon_seed_variance_100m — Extra seeds, canon 100m, lr=3e-4
`canon_seed_variance_100m_canon_3e-4_seed_55_0.4.118`, `canon_seed_variance_100m_canon_3e-4_seed_56_0.4.119`

---

## Scaling Law Comparison Status

Runs for inclusion in the final comparison (3 seeds each, optimal lr):

| Size | Llama seeds | Canon seeds | Status |
|------|-------------|-------------|--------|
| 3m   | `0.4.84` (s54) | `0.4.80` (s54) | ❌ 2 seeds missing each; LR not verified for 3m |
| 10m  | `0.4.82` (s54) | `0.4.78` (s54) | ❌ 2 seeds missing each; LR not verified for 10m |
| 30m  | `0.4.83` (s52), `0.4.120` (s55)†, `0.4.121` (s56)† | `0.4.XX` (s52) | ⚠️ Llama complete; canon needs 2 more seeds |
| 100m | `0.4.81` (s54), `0.4.99` (s55), `0.4.100` (s56) | `0.4.77` (s54), `0.4.118` (s55), `0.4.119` (s56) | ✅ Complete |

† from `depo_variance_impact_investigation_2/all_tasks_seed_{55,56}` — all-tasks training, model.seed matches, functionally equivalent.

---

## remaining_runs experiment

All 12 configs in `remaining_runs/` use lr=3e-4, lr_min=3e-5, 80k steps. Each just changes `model.seed` and `model.canon_set`:

| File | Size | Arch | seed |
|------|------|------|------|
| `llama_3m_seed_55.yaml` | 3m | Llama | 55 |
| `llama_3m_seed_56.yaml` | 3m | Llama | 56 |
| `canon_3m_seed_55.yaml` | 3m | Canon | 55 |
| `canon_3m_seed_56.yaml` | 3m | Canon | 56 |
| `llama_10m_seed_55.yaml` | 10m | Llama | 55 |
| `llama_10m_seed_56.yaml` | 10m | Llama | 56 |
| `canon_10m_seed_55.yaml` | 10m | Canon | 55 |
| `canon_10m_seed_56.yaml` | 10m | Canon | 56 |
| `canon_30m_seed_55.yaml` | 30m | Canon | 55 |
| `canon_30m_seed_56.yaml` | 30m | Canon | 56 |
