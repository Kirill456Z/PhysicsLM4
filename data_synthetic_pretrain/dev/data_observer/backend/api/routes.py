from fastapi import APIRouter, HTTPException

from models import (
    GenerateSampleRequest,
    GenerateBatchRequest,
    ValidateConfigRequest,
    ValidateConfigResponse,
    TasksResponse,
    DefaultConfigResponse,
    EvaluateRequest,
)
from config import parse_config, DEFAULT_DEPO_CONFIG
from generation import generate_sample, generate_batch, evaluate_generation

from data_synthetic_pretrain.tasks import SYNTHETIC_TASKS

router = APIRouter()

_DEFAULT_CONFIGS: dict[str, str] = {
    "depo": DEFAULT_DEPO_CONFIG,
}


@router.get("/tasks", response_model=TasksResponse)
def list_tasks():
    return {"tasks": list(SYNTHETIC_TASKS.keys())}


@router.get("/default-config/{task_name}", response_model=DefaultConfigResponse)
def get_default_config(task_name: str):
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
            num_hops=req.num_hops,
            graph_nodes=req.graph_nodes,
            graph_edges=req.graph_edges,
        )
        return {"metrics": metrics}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {e}")
