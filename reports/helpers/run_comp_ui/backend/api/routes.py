import os
import re
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import logging

logger = logging.getLogger(__name__)

router = APIRouter()

_here = Path(__file__).resolve().parent.parent
_repo_root = _here.parent.parent.parent.parent
EXPS_DIR = Path(os.environ.get("EXPS_DIR", str(_repo_root / "recipe_stashes" / "exps")))

logger.warning("EXPS_DIR = %s  (exists: %s)", EXPS_DIR, EXPS_DIR.exists())

SEMVER_RE = re.compile(r"_(\d+)\.(\d+)\.(\d+)$")


# ── helpers ──────────────────────────────────────────────────────────────────

def _flatten(d: dict, prefix: str = "") -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = str(v)
    return out


def _get_nested(config: dict, key: str):
    node = config
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _load_all_runs() -> list[dict]:
    runs = []
    if not EXPS_DIR.exists():
        return runs
    for exp_dir in sorted(EXPS_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue
        exp_name = exp_dir.name
        for yaml_file in sorted(exp_dir.glob("*.yaml")):
            run_name = yaml_file.stem
            try:
                raw = yaml.safe_load(yaml_file.read_text())
            except Exception as e:
                logger.warning("YAML parse error in %s: %s", yaml_file, e)
                continue
            if not isinstance(raw, dict):
                logger.warning("Skipping %s: top-level is not a mapping", yaml_file)
                continue
            runs.append({
                "id": f"{exp_name}_{run_name}",
                "exp": exp_name,
                "run": run_name,
                "config": _flatten(raw),
            })
    return runs


# ── models ───────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    yaml_filter: str
    minor: int = -1  # -1 = no filter


# ── routes ───────────────────────────────────────────────────────────────────

@router.get("/runs")
def get_runs():
    runs = _load_all_runs()
    return {"runs": runs, "exps_dir": str(EXPS_DIR), "exps_dir_exists": EXPS_DIR.exists()}


@router.get("/sample")
def get_sample():
    """Return first run's flattened config — for debugging."""
    runs = _load_all_runs()
    if not runs:
        return {"run": None}
    return {"run": runs[0]}


@router.post("/search")
def search_runs(req: SearchRequest):
    try:
        raw_filter = yaml.safe_load(req.yaml_filter)
    except yaml.YAMLError as e:
        raise HTTPException(status_code=400, detail=f"YAML parse error: {e}")

    if not isinstance(raw_filter, dict):
        raise HTTPException(status_code=400, detail="Filter YAML must be a mapping")

    flat_filter = _flatten(raw_filter)
    all_runs = _load_all_runs()

    matched = []
    for run in all_runs:
        # Minor version filter: check the run's `name` field for semver pattern
        if req.minor >= 0:
            name_val = run["config"].get("name", "")
            m = SEMVER_RE.search(str(name_val))
            if m and int(m.group(2)) != req.minor:
                continue

        # Config key-value match
        if all(
            run["config"].get(k, "").lower() == str(v).lower()
            for k, v in flat_filter.items()
        ):
            matched.append(run)

    return {"runs": matched}
