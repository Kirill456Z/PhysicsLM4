import logging
import os
import re
from pathlib import Path

import yaml
import wandb

from .models import RunConfiguration, WandbRunCollection
from .run_wrapper import WandbRun

logger = logging.getLogger(__name__)

_SRC_DIR = Path(__file__).resolve().parent
RECIPE_STASHES_DIR = _SRC_DIR.parent.parent.parent.parent / "recipe_stashes"

SEMVER_RE = re.compile(r"(\d+\.\d+\.\d+)$")

ENTITY = os.environ.get("WANDB_ENTITY", "kirill456z")
PROJECT = os.environ.get("WANDB_PROJECT", "physics4llm")


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def _config_matches(yaml_config: dict, filter_config: dict) -> bool:
    flat_filter = _flatten(filter_config)
    flat_config = _flatten(yaml_config)
    for k, v in flat_filter.items():
        config_val = flat_config.get(k)
        if config_val is None:
            return False
        if str(config_val).lower().strip() != str(v).lower().strip():
            return False
    return True


def _extract_semver(filename: str) -> str | None:
    m = SEMVER_RE.search(Path(filename).stem)
    return m.group(1) if m else None


def _wandb_run_name(yaml_path: Path, stashes_dir: Path) -> str:
    """
    Derive the wandb display name from the file path:
      - exps/{exp_name}/{run_stem}.yaml  →  "{exp_name}_{run_stem}"
      - {run_stem}.yaml                  →  "{run_stem}"
    """
    exps_dir = stashes_dir / "exps"
    if not exps_dir in yaml_path.parents:
        name = yaml_path.stem
        if "debug_" in name:
            name = name.replace("debug_", "")
        return name
    try:
        rel = yaml_path.relative_to(exps_dir)
        # rel = {exp_name}/{run_stem}.yaml
        exp_name = rel.parts[0]
        return f"{exp_name}_{yaml_path.stem}"
    except ValueError:
        return yaml_path.stem


class WandbRunExtractor:
    """
    Finds wandb runs by matching a RunConfiguration against local recipe stashes,
    then fetches their histories from the wandb API.

    Usage:
        extractor = WandbRunExtractor()
        runs = extractor.extract_runs(
            '''
            model:
                dim: 512
                n_layers: 8
            '''
        )
        import seaborn as sns
        sns.lineplot(x=runs[0].steps, y=runs[0].out.loss)
    """

    def __init__(
        self,
        entity: str = ENTITY,
        project: str = PROJECT,
        stashes_dir: str | Path | None = None,
    ):
        self.entity = entity
        self.project = project
        self.stashes_dir = Path(stashes_dir) if stashes_dir else RECIPE_STASHES_DIR
        self._api: wandb.Api | None = None

    @property
    def api(self) -> wandb.Api:
        if self._api is None:
            self._api = wandb.Api()
        return self._api

    def _find_matching_configs(self, parameters: RunConfiguration) -> list[tuple[str, Path]]:
        """
        Scan all YAML files under stashes_dir. For each file that matches
        `parameters`, return (wandb_run_name, yaml_path).
        """
        all_yamls = sorted(self.stashes_dir.rglob("*.yaml"))
        logger.info("Scanning %d YAML files in %s", len(all_yamls), self.stashes_dir)

        matches: list[tuple[str, Path]] = []
        for yaml_path in all_yamls:
            if _extract_semver(yaml_path.name) is None:
                continue
            try:
                config = yaml.safe_load(yaml_path.read_text())
            except Exception as e:
                logger.warning("Failed to parse %s: %s", yaml_path, e)
                continue
            if not isinstance(config, dict):
                continue
            if _config_matches(config, parameters):
                run_name = _wandb_run_name(yaml_path, self.stashes_dir)
                matches.append((run_name, yaml_path))

        logger.info("Matched %d local configs", len(matches))
        for name, path in matches:
            logger.debug("  %s  <-  %s", name, path.relative_to(self.stashes_dir))
        return matches

    def find_matching_run_names(self, parameters: RunConfiguration | str) -> list[str]:
        """Return just the resolved wandb run names without fetching histories."""
        if isinstance(parameters, str):
            parameters = RunConfiguration.from_yaml(parameters)
        return [name for name, _ in self._find_matching_configs(parameters)]

    def extract_runs(self, parameters: RunConfiguration | str) -> WandbRunCollection:
        """
        Find all runs in recipe_stashes matching `parameters` and return them
        as a WandbRunCollection with their histories available for plotting.

        Args:
            parameters: RunConfiguration or YAML string specifying the config
                        subset to match against. All specified keys must match.
        """
        if isinstance(parameters, str):
            parameters = RunConfiguration.from_yaml(parameters)

        matching = self._find_matching_configs(parameters)
        print(f"[extractor] local configs matched: {len(matching)}")
        if not matching:
            return WandbRunCollection([], parameters)

        run_names = [name for name, _ in matching]
        print(f"[extractor] querying wandb for {len(run_names)} runs: {run_names[:5]}{'...' if len(run_names) > 5 else ''}")

        filters = {"$or": [{"name": name} for name in run_names]}
        try:
            raw_runs = list(self.api.runs(f"{self.entity}/{self.project}", filters=filters))
        except Exception as e:
            raise RuntimeError(f"Failed to query wandb runs: {e}") from e

        print(f"[extractor] wandb returned {len(raw_runs)} runs")
        runs = [WandbRun(r) for r in raw_runs]
        return WandbRunCollection(runs, parameters)
