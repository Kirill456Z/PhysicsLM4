from fastapi import APIRouter, HTTPException

from models import (
    GenerateSampleRequest,
    GenerateBatchRequest,
    ValidateConfigRequest,
    ValidateConfigResponse,
    TasksResponse,
    DefaultConfigResponse,
    TaskTabsResponse,
    SaveTaskTabRequest,
    SaveTaskTabResponse,
    EvaluateRequest,
)
from config import (
    parse_config,
    DEFAULT_DEPO_CONFIG,
    load_tasks_config_yaml,
    build_single_task_config_yaml,
    save_task_tab_config,
)
from generation import generate_sample, generate_batch, evaluate_generation

from data_synthetic_pretrain.tasks import SYNTHETIC_TASKS

router = APIRouter()

_DEFAULT_CONFIGS: dict[str, str] = {
    "depo": DEFAULT_DEPO_CONFIG,
}


@router.get("/tasks", response_model=TasksResponse)
def list_tasks():
    return {"tasks": list(SYNTHETIC_TASKS.keys())}


@router.get("/task-tabs", response_model=TaskTabsResponse)
def get_task_tabs():
    full_yaml = load_tasks_config_yaml()
    # Build one tab per entry in synthetic_tasks list
    import yaml as _yaml

    data = _yaml.safe_load(full_yaml)
    tasks_raw = (
        (data or {}).get("synthetic_tasks_generation_args", {}).get("synthetic_tasks", []) or []
    )
    tabs = []
    for i in range(len(tasks_raw)):
        tab_name, task_name, config_yaml = build_single_task_config_yaml(full_yaml, i)
        tabs.append({"tab_name": tab_name, "task_name": task_name, "config_yaml": config_yaml})
    return {"full_config_yaml": full_yaml, "tabs": tabs}


@router.put("/task-tabs/{tab_name}", response_model=SaveTaskTabResponse)
def save_task_tab(tab_name: str, req: SaveTaskTabRequest):
    try:
        task_name, updated_full = save_task_tab_config(tab_name=tab_name, single_task_config_yaml=req.config_yaml)
        return {
            "tab_name": tab_name,
            "task_name": task_name,
            "updated_full_config_yaml": updated_full,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save config: {e}")


@router.get("/default-config/{task_name}", response_model=DefaultConfigResponse)
def get_default_config(task_name: str):
    # Backwards-compatible: default to repo tasks_config.yaml if available.
    try:
        config_yaml = load_tasks_config_yaml()
    except Exception:
        config_yaml = _DEFAULT_CONFIGS.get(task_name, DEFAULT_DEPO_CONFIG)
    return {"task_name": task_name, "config_yaml": config_yaml}


@router.post("/validate-config", response_model=ValidateConfigResponse)
def validate_config(req: ValidateConfigRequest):
    config, errors = parse_config(req.config_yaml)
    if errors:
        return {"valid": False, "errors": errors}
    task_names = [t.task_name for t in config.tasks]
    return {"valid": True, "errors": [], "parsed": {"tasks": task_names}}


@router.post("/generate-sample")
def generate(req: GenerateSampleRequest):
    config, errors = parse_config(req.config_yaml)
    if errors:
        raise HTTPException(status_code=400, detail={"errors": errors})
    try:
        return generate_sample(config, req.task_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")
   


@router.post("/generate-batch")
def batch(req: GenerateBatchRequest):
    config, errors = parse_config(req.config_yaml)
    if errors:
        raise HTTPException(status_code=400, detail={"errors": errors})
    try:
        capped = min(req.batch_size, 10)
        samples = generate_batch(config, req.task_name, capped)
        return {"samples": samples, "count": len(samples)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch generation failed: {e}")


@router.post("/evaluate")
def evaluate(req: EvaluateRequest):
    config, errors = parse_config(req.config_yaml)
    if errors:
        raise HTTPException(status_code=400, detail={"errors": errors})
    if req.task_name == "shortest_path" and (req.query_node is None or req.answer_nodes is None):
        raise HTTPException(
            status_code=400,
            detail="Missing shortest_path-specific fields: query_node, answer_nodes",
        )
    if req.task_name == "concomp_factor" and (req.answer_nodes is None or req.components is None):
        raise HTTPException(
            status_code=400,
            detail="Missing concomp_factor-specific fields: answer_nodes, components",
        )
    try:
        metrics = evaluate_generation(
            config=config,
            task_name=req.task_name,
            generation=req.generation,
            task_index=req.task_index,
            context=req.context,
            loss_mask=req.loss_mask,
            answer_start_index=req.answer_start_index,
            query_nodes=req.query_nodes,
            answer_nodes=req.answer_nodes,
            components=req.components,
            num_hops=req.num_hops,
            query_node=req.query_node,
            answer_sequence=req.answer_sequence,
            graph_nodes=req.graph_nodes,
            graph_edges=req.graph_edges,
        )
        return {"metrics": metrics}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {e}")
