import copy
import numpy as np
import wandb

# Flat dot-separated keys stripped from config before it is exposed.
# These are run-specific identifiers that clutter differentiation output.
_STRIP_CONFIG_KEYS = {
    "name",
    "dump_dir",
    "logging.wandb.id",
    "logging.wandb.name",
    "checkpoint.path",
}


def _delete_nested(d: dict, flat_key: str) -> None:
    parts = flat_key.split(".")
    node = d
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            return
        node = node[part]
    node.pop(parts[-1], None)


def _merge_include_generators(cfg: dict) -> None:
    """
    Merge top-level include_generators and
    synthetic_tasks_formatting_args.include_generators into a single
    top-level include_generators list, deduplicating while preserving order.
    """
    top = cfg.get("include_generators") or []
    stfa = cfg.get("synthetic_tasks_formatting_args", {})
    nested = (stfa.get("include_generators") if isinstance(stfa, dict) else None) or []

    if not top and not nested:
        return

    seen: set = set()
    merged = []
    for item in list(top) + list(nested):
        if item not in seen:
            seen.add(item)
            merged.append(item)

    cfg["include_generators"] = merged
    if isinstance(stfa, dict):
        stfa.pop("include_generators", None)


class _MetricAccessor:
    """
    Proxy enabling chained dot-access for slash-separated wandb metric keys.
    e.g. run.out.loss resolves to the history column "out/loss".
    Behaves as a numpy array when iterated or passed to plotting functions.
    """

    def __init__(self, run_wrapper: "WandbRun", prefix: str):
        object.__setattr__(self, "_run", run_wrapper)
        object.__setattr__(self, "_prefix", prefix)

    def __getattr__(self, name: str) -> "_MetricAccessor":
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        prefix = object.__getattribute__(self, "_prefix")
        run = object.__getattribute__(self, "_run")
        return _MetricAccessor(run, f"{prefix}/{name}")

    def _resolve(self) -> np.ndarray:
        run = object.__getattribute__(self, "_run")
        prefix = object.__getattribute__(self, "_prefix")
        return run._get_column(prefix)

    def __array__(self, dtype=None) -> np.ndarray:
        arr = self._resolve()
        return arr.astype(dtype) if dtype is not None else arr

    def __len__(self) -> int:
        return len(self._resolve())

    def __iter__(self):
        return iter(self._resolve())

    def __repr__(self) -> str:
        prefix = object.__getattribute__(self, "_prefix")
        return f"MetricAccessor('{prefix}')"


class WandbRun:
    """
    Wraps a wandb Run object and exposes metric history via attribute access.

    Usage:
        run.steps          -> np.array of step indices
        run.loss           -> np.array of values for metric "loss"
        run.out.loss       -> np.array of values for metric "out/loss"
        run.train.out.loss -> np.array of values for metric "train/out/loss"
    """

    def __init__(self, run: wandb.sdk.wandb_run.Run):
        object.__setattr__(self, "_run", run)
        object.__setattr__(self, "_history_df", None)

    def _get_history(self):
        df = object.__getattribute__(self, "_history_df")
        if df is None:
            run = object.__getattribute__(self, "_run")
            df = run.history(samples=10000)
            object.__setattr__(self, "_history_df", df)
        return df

    def _get_column(self, key: str) -> np.ndarray:
        df = self._get_history()
        if key not in df.columns:
            available = [c for c in df.columns if not c.startswith("_")]
            raise AttributeError(
                f"Metric '{key}' not found in run history. "
                f"Available: {available[:20]}{'...' if len(available) > 20 else ''}"
            )
        return df[key].dropna().values

    @property
    def steps(self) -> np.ndarray:
        df = self._get_history()
        col = "_step" if "_step" in df.columns else "step"
        return df[col].values

    @property
    def name(self) -> str:
        return object.__getattribute__(self, "_run").name

    @property
    def id(self) -> str:
        return object.__getattribute__(self, "_run").id

    @property
    def config(self) -> dict:
        cfg = copy.deepcopy(dict(object.__getattribute__(self, "_run").config))
        _merge_include_generators(cfg)
        for key in _STRIP_CONFIG_KEYS:
            _delete_nested(cfg, key)
        return cfg

    def __getattr__(self, name: str) -> _MetricAccessor:
        return _MetricAccessor(self, name)

    def __dir__(self) -> list:
        try:
            df = self._get_history()
            return [c for c in df.columns if not c.startswith("_")]
        except Exception:
            return []

    def __repr__(self) -> str:
        run = object.__getattribute__(self, "_run")
        return f"WandbRun(name='{run.name}', id='{run.id}')"
