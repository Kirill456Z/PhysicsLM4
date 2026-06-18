### A ui to compare configurations of experimental runs on wandb

#### Stack
React + Vite + TypeScript + Tailwind (frontend) · FastAPI + uvicorn (backend)

#### Setup & launch
In two terminals:
```bash
# Terminal 1 — backend
cd reports/helpers/run_comp_ui && ./start_backend.sh

# Terminal 2 — frontend
cd reports/helpers/run_comp_ui && ./start.sh
```
Open http://localhost:3000

#### Config source
Runs are loaded from `recipe_stashes/exps/`. Each sub-folder is an **exp name**; each `.yaml` file inside is a **run name**. Runs appear as `{exp_name}_{run_name}` in the UI.

The optional **minor** filter matches the semver minor in the run's `name` field (e.g. `mix_1.4.0` → minor 4). Leave blank to skip.



Implements two fetaures for finding runs on wandb and comparing their configurations

#### Feature 1. Parameter constrained search. Given a configuration yaml input, for example @/Users/kirillzemlanskij/Workspace/PhysicsLM4/lingua_modified/apps/main/configs/exps/all_tasks_comp/canon.yaml
Queries and finds all runs on wandb in specified project and workspace (default kirill456z / physics4llm ) that match provided yaml configuration (i.e. have the same values in keys that are provided in yaml). For example if yaml only contains
- canon_set : ABCD
Has to find all runs with canon_set equal to abcd. Only lists names of the runs

#### Feature 2. Config comparison
Given a list of run names (can be added from feature 1)
shows a list of features where the configurations of runs differ. For example if given runs

|-|x|y|z|t|
-------
|run1|1|2|3|4|
|run2|1|3|2|1|
|run3|1|2|2|4|

will show a list of parameters y, z, t
Allows to choose up to two parameters and display a table of bins with names of runs falling to each bin. 
For example if y and z are choosen should display

|y/z|2|3|
------
|2|run3|run1|
|3|run2|-|

The goal of the features is to find sets of runs where all options are ablated, for example
all runs of set model size with different training data. 