import os
import re
import wandb

ENTITY = os.environ.get("WANDB_ENTITY", "kirill456z")
PROJECT = "physics4llm"


def _get_nested(config: dict, key: str):
    """Resolve a key that may be dotted (e.g. 'model.dim') against a nested dict."""
    parts = key.split(".")
    node = config
    for part in parts:
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _config_matches(run_config: dict, required: dict) -> bool:
    for key, expected in required.items():
        actual = _get_nested(run_config, key)
        if actual is None:
            return False
        # wandb stores numeric strings as numbers; normalise both sides to str for comparison
        if str(actual) != str(expected):
            return False
    return True


def find_runs_with_config(config: dict, minor: int = 4) -> list:
    """Return wandb runs from physics4llm whose name encodes the given minor version
    and whose config contains all key-value pairs in *config*.

    Run naming convention: {run_name}_{major}.{minor}.{patch}
    """
    api = wandb.Api(api_key=os.environ.get("WANDB_API_KEY"))
    runs = api.runs(f"{ENTITY}/{PROJECT}")

    semver_re = re.compile(r"_(\d+)\.(\d+)\.(\d+)$")

    matched = []
    for run in runs:
        m = semver_re.search(run.name)
        if m is None:
            continue
        run_minor = int(m.group(2))
        if run_minor != minor:
            continue
        if _config_matches(run.config, config):
            matched.append(run)

    return matched
