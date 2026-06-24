from __future__ import annotations

from typing import Literal, Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .styles import PALETTE


def _get_flat_config_value(run, flat_key: str) -> str:
    parts = flat_key.split(".")
    d = run.config
    for part in parts:
        if not isinstance(d, dict) or part not in d:
            return "<unknown>"
        d = d[part]
    return str(d)


def _get_metric_series(run, metric_key: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (steps, values) arrays for `metric_key`, with NaNs dropped."""
    df = run._get_history()
    if metric_key not in df.columns:
        available = [c for c in df.columns if not c.startswith("_")]
        raise KeyError(
            f"Metric '{metric_key}' not in run '{run.name}'. "
            f"Available (first 20): {available[:20]}"
        )
    step_col = "_step" if "_step" in df.columns else "step"
    mask = df[metric_key].notna()
    return (
        df[step_col][mask].values.astype(float),
        df[metric_key][mask].values.astype(float),
    )


def _lighten(color, amount: float = 0.55) -> tuple:
    """Blend `color` toward white by `amount` (0 = unchanged, 1 = white)."""
    r, g, b = mcolors.to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def lineplot(
    collection,
    property: str,
    by: str,
    label: str = "",
    aggregate: Literal["class", "none"] = "class",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot a lineplot of `property` for every run in `collection`.

    Parameters
    ----------
    collection : WandbRunCollection
        Runs to visualise.
    property : str
        Slash-separated wandb metric key, e.g. ``"loss/out"`` or ``"train/acc"``.
    by : str
        Dot-separated config key used to colour-group runs,
        e.g. ``"model.canon_set"``.
    label : str
        Axes title.
    aggregate : {"class", "none"}
        ``"class"`` – show per-class mean (full colour, thick) and individual
        runs (light colour, thin).  ``"none"`` – plot every run without
        aggregation.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    # Group runs by their `by` config value.
    groups: dict[str, list] = {}
    for run in collection:
        key = _get_flat_config_value(run, by)
        groups.setdefault(key, []).append(run)

    n_classes = len(groups)
    palette = sns.color_palette(PALETTE[:n_classes] if n_classes <= len(PALETTE) else "tab10", n_colors=n_classes)
    color_map = {cls: palette[i] for i, cls in enumerate(sorted(groups))}

    for cls, runs in sorted(groups.items()):
        base_color = color_map[cls]
        light_color = _lighten(base_color, amount=0.55)

        if aggregate == "class":
            all_steps, all_values = [], []
            for run in runs:
                try:
                    steps, values = _get_metric_series(run, property)
                except KeyError as exc:
                    print(f"[lineplot] skipping run '{run.name}': {exc}")
                    continue
                all_steps.append(steps)
                all_values.append(values)
                ax.plot(steps, values, color=light_color, linewidth=0.8, alpha=0.5, zorder=1)

            if not all_steps:
                continue

            # Compute mean over the intersection of all step ranges.
            grid_min = max(s[0] for s in all_steps)
            grid_max = min(s[-1] for s in all_steps)
            if grid_min >= grid_max:
                # Fallback: use global range with NaN-safe interp.
                grid_min = min(s[0] for s in all_steps)
                grid_max = max(s[-1] for s in all_steps)

            grid = np.linspace(grid_min, grid_max, 500)
            interpolated = np.stack([np.interp(grid, s, v) for s, v in zip(all_steps, all_values)])
            mean_vals = interpolated.mean(axis=0)

            ax.plot(grid, mean_vals, color=base_color, linewidth=2.2, label=f"{by}={cls}", zorder=2)

        else:  # aggregate == "none"
            first = True
            for run in runs:
                try:
                    steps, values = _get_metric_series(run, property)
                except KeyError as exc:
                    print(f"[lineplot] skipping run '{run.name}': {exc}")
                    continue
                ax.plot(
                    steps, values,
                    color=base_color,
                    linewidth=1.2,
                    alpha=0.8,
                    label=f"{by}={cls}" if first else None,
                    zorder=2,
                )
                first = False

    # Deduplicate legend entries and draw.
    handles, labels = ax.get_legend_handles_labels()
    seen: dict[str, object] = {}
    for h, lbl in zip(handles, labels):
        seen.setdefault(lbl, h)
    if seen:
        ax.legend(seen.values(), seen.keys())

    if label:
        ax.set_title(label)
    ax.set_xlabel("Step")
    ax.set_ylabel(property)

    return ax
