import logging
from collections import defaultdict
from typing import List, TYPE_CHECKING
import yaml

if TYPE_CHECKING:
    from .run_wrapper import WandbRun

logger = logging.getLogger(__name__)


# ── config helpers ────────────────────────────────────────────────────────────

def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def _unflatten(flat: dict) -> dict:
    result: dict = {}
    for key, value in flat.items():
        parts = key.split(".")
        node = result
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return result


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


# ── RunConfiguration ──────────────────────────────────────────────────────────

class RunConfiguration(dict):
    """A nested dictionary defining a subset of parameters to match against."""

    def to_yaml(self) -> str:
        return yaml.dump(dict(self))

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "RunConfiguration":
        data = yaml.safe_load(yaml_str)
        if not isinstance(data, dict):
            raise ValueError("RunConfiguration YAML must be a mapping")
        return cls(data)


# ── RunDifferentiation ────────────────────────────────────────────────────────

class RunDifferentiation(dict):
    """
    Maps each varying config property to a {value: run_count} breakdown.
    Only properties with more than one distinct value are included.

    Example:
        {
            "model.canon_set": {"": 10, "A": 20, "ABCD": 30},
            "optim.lr":        {0.0006: 30, 0.001: 30},
        }
    """

    def __repr__(self) -> str:
        if not self:
            return "RunDifferentiation (no varying properties)"

        total = sum(next(iter(v.values())) for v in self.values()) if self else 0
        # total = sum of counts for first property (all props have same total)
        first_counts = next(iter(self.values())) if self else {}
        total = sum(first_counts.values())

        lines = [f"RunDifferentiation  ({total} runs, {len(self)} varying properties)\n"]
        for prop, counts in self.items():
            lines.append(f"  {prop}")
            max_val_len = max(len(str(v)) for v in counts)
            for val, count in sorted(counts.items(), key=lambda x: -x[1]):
                bar = "█" * min(count, 40)
                lines.append(f"    {str(val):<{max_val_len}}  {count:>4} runs  {bar}")
            lines.append("")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        if not self:
            return "<em>RunDifferentiation (no varying properties)</em>"

        first_counts = next(iter(self.values())) if self else {}
        total = sum(first_counts.values())

        rows = [
            "<table style='border-collapse:collapse;font-family:monospace;font-size:13px'>",
            f"<caption style='text-align:left;font-weight:bold;padding-bottom:6px'>"
            f"RunDifferentiation — {total} runs, {len(self)} varying properties</caption>",
        ]
        for prop, counts in self.items():
            max_count = max(counts.values())
            rows.append(
                f"<tr><td colspan=3 style='padding-top:8px;font-weight:bold;"
                f"border-top:1px solid #ccc'>{prop}</td></tr>"
            )
            for val, count in sorted(counts.items(), key=lambda x: -x[1]):
                pct = int(count / max_count * 100)
                bar = (
                    f"<div style='background:#4c72b0;width:{pct}%;height:12px;"
                    f"min-width:2px;border-radius:2px'></div>"
                )
                rows.append(
                    f"<tr>"
                    f"<td style='padding:2px 12px 2px 16px;color:#555'>{val!r}</td>"
                    f"<td style='padding:2px 8px;text-align:right'>{count}</td>"
                    f"<td style='padding:2px 8px;width:150px'>{bar}</td>"
                    f"</tr>"
                )
        rows.append("</table>")
        return "\n".join(rows)


# ── WandbRunCollection ────────────────────────────────────────────────────────

class WandbRunCollection(list):
    """A collection of wandb runs satisfying a given set of parameters."""

    def __init__(self, runs: List["WandbRun"], parameters: RunConfiguration):
        super().__init__(runs)
        self.parameters = parameters

    def differentiate(self) -> RunDifferentiation:
        """
        Return a RunDifferentiation showing, for each config property that
        varies across runs, how many runs have each distinct value.
        """
        counts: dict[str, dict] = defaultdict(lambda: defaultdict(int))
        for run in self:
            for key, val in _flatten(run.config).items():
                if isinstance(val, list):
                    val = tuple(val)
                if isinstance(val, dict):
                    val = tuple(val.items())
                counts[key][val] += 1

        varying = {
            key: dict(val_counts)
            for key, val_counts in counts.items()
            if len(val_counts) > 1
        }
        return RunDifferentiation(varying)

    def constrain(self, conf_subset: dict) -> "WandbRunCollection":
        """
        Return a sub-collection of runs whose (flattened) config matches every
        key-value pair in `conf_subset`.

        `conf_subset` uses flat dot-separated keys, e.g.::
            {"model.canon_set": "A", "optim.lr": 6e-4}

        Raises ValueError if `conf_subset` conflicts with an already-set
        parameter in this collection's `.parameters`.
        """
        flat_existing = _flatten(self.parameters)
        for key, val in conf_subset.items():
            if key in flat_existing and str(flat_existing[key]) != str(val):
                raise ValueError(
                    f"Constraint conflict on '{key}': "
                    f"collection already has {flat_existing[key]!r}, "
                    f"constraint wants {val!r}"
                )

        selected = [
            run for run in self
            if all(
                str(_flatten(run.config).get(k, "")) == str(v)
                for k, v in conf_subset.items()
            )
        ]

        print(
            f"[constrain] {len(selected)} runs out of {len(self)} selected "
            f"on condition {conf_subset}"
        )

        new_params = RunConfiguration(_deep_merge(
            dict(self.parameters),
            _unflatten(conf_subset),
        ))
        return WandbRunCollection(selected, new_params)

    def __repr__(self) -> str:
        return f"WandbRunCollection({len(self)} runs, parameters={dict(self.parameters)})"
