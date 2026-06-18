# Findings and evidence

## Canon improves the performance of transformers be a factor of X
### Description:

Experiments to test scaling laws of canon layer. Training models of different size with different seeds and aggregating the results to observe if the improvement is consistent accross various model sizes and see how much canon scales the performance of the models

### Evidence: 
/Users/kirillzemlanskij/Workspace/PhysicsLM4/reports/evidence/canon_scaling_laws.ipynb

### Missing runs:

## Mixing different synthetic tasks into training reduces variance between runs

Training only on depo synthetic has been identified to be noisy with respect to initialization seed. Adding more tasks and encoding variations has shown to decrease the variance

### Evidence
Filter:
```
model:
  dim: 512
  n_layers: 8
  n_heads: 8
  head_dim: 64
  canon_set: ''
optim:
  lr: 3e-4
```

```
{
    '['concomp_factor_edges_list', 'concomp_factor_adj_list', 'depo_edges_list', 'depo_adj_list', 'shortest_path_edges_list', 'shortest_path_adj_list', 'bfs_edges_list', 'bfs_adj_list']': {
        '54': [],
        '55': ['depo_variance_impact_investigation_30m_all_tasks_seed_55_0.4.106', 'depo_variance_impact_investigation_30m_all_tasks_seed_55_0.4.111', 'depo_variance_impact_investigation_2_all_tasks_seed_55_0.4.120'],
        '56': ['depo_variance_impact_investigation_30m_all_tasks_seed_56_0.4.107', 'depo_variance_impact_investigation_30m_all_tasks_seed_56_0.4.112', 'depo_variance_impact_investigation_2_all_tasks_seed_56_0.4.121'],
        '57': [],
        '64': ['scaling_law_exps_3_llama_30m_0.4.83', 'scaling_law_4_more_seeds_llama_30m_seed_57_0.4.158', 'scaling_law_4_more_seeds_llama_30m_seed_58_0.4.159'],
    },
    '['concomp_factor_edges_list', 'concomp_factor_adj_list', 'shortest_path_edges_list', 'shortest_path_adj_list', 'bfs_edges_list', 'bfs_adj_list']': {
        '54': ['depo_variance_impact_investigation_30m_no_depo_seed_54_0.4.108', 'depo_variance_impact_investigation_30m_no_depo_seed_54_0.4.113', 'depo_variance_impact_investigation_2_no_depo_seed_54_0.4.122'],
        '55': ['depo_variance_impact_investigation_30m_no_depo_seed_55_0.4.109', 'depo_variance_impact_investigation_30m_no_depo_seed_55_0.4.114', 'depo_variance_impact_investigation_2_no_depo_seed_55_0.4.123'],
        '56': ['depo_variance_impact_investigation_30m_no_depo_seed_56_0.4.110', 'depo_variance_impact_investigation_30m_no_depo_seed_56_0.4.115', 'depo_variance_impact_investigation_2_no_depo_seed_56_0.4.124'],
        '57': [],
        '64': [],
    },
    '['concomp_factor_edges_list', 'depo_edges_list', 'shortest_path_edges_list', 'bfs_edges_list']': {
        '54': [],
        '55': ['seed_variance_exps_remaining_only_edges_list_55_0.4.192'],
        '56': ['seed_variance_exps_remaining_only_edges_list_56_0.4.193'],
        '57': ['seed_variance_exps_remaining_only_edges_list_57_0.4.194'],
        '64': [],
    },
    '['depo_edges_list', 'depo_adj_list']': {
        '54': [],
        '55': ['seed_variance_exps_remaining_only_depo_55_0.4.186'],
        '56': ['seed_variance_exps_remaining_only_depo_56_0.4.187'],
        '57': ['seed_variance_exps_remaining_only_depo_57_0.4.188'],
        '64': [],
    },
    '['depo_edges_list']': {
        '54': [],
        '55': ['seed_variance_exps_remaining_only_depo_edges_55_0.4.189'],
        '56': ['seed_variance_exps_remaining_only_depo_edges_56_0.4.190'],
        '57': ['seed_variance_exps_remaining_only_depo_edges_57_0.4.191'],
        '64': [],
    },
}
```

### Missing runs
- **depo_only / 2-enc**: no runs exist at all. Need depo_edges_list + depo_adj_list runs with seeds 52–56 to directly compare one- vs two-encoding depo training.
- **depo_only / 1-enc with varied seeds**: the early depo runs only cover model_seed 51 and 52 (and differ in other configs too). Need clean depo-only runs at seeds 53, 54, 55 matching the same hyperparameter setup as the all_tasks comparison group to properly measure depo-only variance.
- **all_tasks / 1-enc with more seeds**: only seeds 52, 53, 54 exist; adding seeds 55–56 would allow a direct apples-to-apples seed variance comparison with the 2-enc group.

## Various canon ablations from p1. (before midterm) (to be separated)
### Evidence
described in 
/Users/kirillzemlanskij/Workspace/PhysicsLM4/reports/main_midterm.tex
and 
/Users/kirillzemlanskij/Workspace/PhysicsLM4/reports/slides/slides.tex