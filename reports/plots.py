import os
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb
import wandb_workspaces.workspaces as ws
import wandb_workspaces.reports.v2 as wr # We use the Reports API for adding panels

def loss_lineplot(title="Text", y_max = 1, smoothing_factor=0.95, name = "out"):
    return wr.LinePlot(
        x="Step",
        y=[f"loss/{name}"],
        title=f"Loss : {title}",
        range_y=(None,y_max),
        log_y = True,
        layout=wr.Layout(x=0, y=0, w=20, h=8),
        smoothing_factor=smoothing_factor,
    )

def canon_weight_fixed_layer(canon_type = "A", layer = 0, max_shift = 6):
    varnames = [
        f"canon_weight/canon{canon_type}/layer_{layer}/shift_{shift}" for shift in range (0, max_shift)
    ]
    return wr.LinePlot(
        x="Step",
        y=varnames,
        title=f"Canon {canon_type} weights : Layer {layer}",
    )

def canon_rms_ratio(canon_type = "A", layers_set = None):
    if layers_set is None:
        layers_set = list(range(0,8))
    varnames = [
        f"canon_align/rms_ratio/canon{canon_type}/layer_{layer_idx}" for layer_idx in layers_set
    ]
    return wr.LinePlot(
        x = "Step",
        y = varnames,
        title = f"Canon {canon_type}: RMS ratio out/in"
    )

def realign_grid(plots, num_columns = None, num_rows = None, total_w = 25, plot_h = 10):
    if num_columns is None and num_rows is None:
        raise ValueError(f"Please specify at least one of num_rows, num_columns")
    if num_columns is None:
        num_columns = (len(plots) - 1) // num_rows + 1
    if num_rows is None:
        num_rows = (len(plots) - 1) // num_columns + 1
    plot_w = total_w // num_columns
    for idx, plot in enumerate(plots):
        x_id = idx % num_columns
        y_id = idx // num_columns
        plot.layout = wr.Layout(plot_w * x_id, plot_h * y_id, plot_w, plot_h)
    return plots

def residual_rms(location: str, layer_set = None):
    assert location in ["ffn", "attn"]
    #ln = f"Layer {layer_set[0]}" if len(layer_set) == 1 else ""
    if layer_set is None:
        layer_set = list(range(0, 8))
    varnames = [
        f"residual_rms/pre_{location}/layer_{layer_idx}" for layer_idx in layer_set
    ]
    return wr.LinePlot(
        x = "Step",
        y = varnames,
        title = f"RMS of residual stream before {location}" 
    )

def grad_contrib(layer):
    return wr.LinePlot(
        x = "Step",
        y = [f"grad_contrib/layer_{layer}"],
        title = f"Total gradient layer {layer}"
    )

def cos_sim(canon_type, layer_set = None):
    if layer_set is None:
        layer_set = list(range(0, 8))
    varnames = [
        f"canon_align/cos_sim/canon{canon_type}/layer_{layer_idx}" for layer_idx in layer_set
    ]
    return wr.LinePlot(
        x = "Step",
        y = varnames,
        title = f"Canon {canon_type}: Cos similarity out vs in"
    )

def outlier_features_kurtosis(layer_idx):
    return wr.LinePlot(
        x = "Step",
        y = [f"outlier_features/kurtosis/layer_{layer_idx}"],
        title = f"Feature RMS Kurtosis on layer {layer_idx}"
    )

def depo_eval_plot(title: str = "Depo eval"):
    return wr.LinePlot(
        x = "Step",
        y = ["eval/depo/avg_accuracy"],
        title=title,
        layout=wr.Layout(x=0, y=0, w=20, h=8)
    )

def speed_cur_iter_time():
    return wr.LinePlot(
        x = "Step",
        y = ["speed/curr_iter_time"],
        title = "Iteration Time"
    )


def _extract_run_names_from_filters(filters: str) -> list[str]:
    match = re.search(r'name in \[(.*)\]', filters)
    if not match:
        return []
    quoted = re.findall(r'"([^"]+)"', match.group(1))
    return quoted


def _run_match_score(run, token: str) -> int:
    token_l = token.lower()
    candidates = [
        str(getattr(run, "id", "")),
        str(getattr(run, "name", "")),
        str(getattr(run, "display_name", "")),
    ]
    path = getattr(run, "path", None)
    if path:
        joined = "/".join(path)
        candidates.append(joined)
        candidates.append(path[-1])

    for candidate in candidates:
        candidate_l = candidate.lower()
        if token_l == candidate_l:
            return 4
    for candidate in candidates:
        candidate_l = candidate.lower()
        if token_l in candidate_l:
            return 3
    for candidate in candidates:
        candidate_l = candidate.lower()
        if candidate_l and candidate_l in token_l:
            return 2
    # Useful for names that only differ by an inserted marker like "_big_vocab_".
    token_tail = token_l.split(".")[-1]
    for candidate in candidates:
        candidate_l = candidate.lower()
        if token_tail and candidate_l.endswith(token_tail):
            return 1
    return 0


def _resolve_runs_for_runset(api, runset):
    run_names = _extract_run_names_from_filters(getattr(runset, "filters", ""))
    if not run_names:
        return list(api.runs(f"{runset.entity}/{runset.project}")), {}

    resolved_runs = []
    color_token_by_run_id = {}
    seen_run_ids = set()

    # First try direct lookup as run ID/path key (fast and precise when tokens are run IDs).
    unresolved_tokens = []
    for token in run_names:
        try:
            direct_run = api.run(f"{runset.entity}/{runset.project}/{token}")
        except Exception:
            direct_run = None
        if direct_run is None:
            unresolved_tokens.append(token)
            continue
        if direct_run.id in seen_run_ids:
            continue
        seen_run_ids.add(direct_run.id)
        resolved_runs.append(direct_run)
        color_token_by_run_id[direct_run.id] = token

    # Then resolve remaining tokens by best matching over run metadata.
    if unresolved_tokens:
        all_runs = list(api.runs(f"{runset.entity}/{runset.project}"))
        for token in unresolved_tokens:
            best_run = None
            best_score = 0
            for run in all_runs:
                score = _run_match_score(run, token)
                if score > best_score:
                    best_score = score
                    best_run = run
            if best_run is None or best_score == 0 or best_run.id in seen_run_ids:
                continue
            seen_run_ids.add(best_run.id)
            resolved_runs.append(best_run)
            color_token_by_run_id[best_run.id] = token

    return resolved_runs, color_token_by_run_id


def _smooth_series(values: np.ndarray, smoothing_factor: float | None) -> np.ndarray:
    if smoothing_factor is None:
        return values
    if smoothing_factor <= 0:
        return values
    if smoothing_factor >= 1:
        return values
    smoothed = np.empty_like(values, dtype=np.float64)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = smoothing_factor * smoothed[i - 1] + (1 - smoothing_factor) * values[i]
    return smoothed


def _to_float_array(values) -> np.ndarray:
    out = []
    for value in values:
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            out.append(np.nan)
    return np.asarray(out, dtype=np.float64)


def _find_key_recursive(obj, target_key: str):
    if isinstance(obj, dict):
        if target_key in obj:
            return obj[target_key]
        for value in obj.values():
            found = _find_key_recursive(value, target_key)
            if found is not None:
                return found
    if isinstance(obj, list):
        for value in obj:
            found = _find_key_recursive(value, target_key)
            if found is not None:
                return found
    return None


def _extract_plot_spec(plot):
    x_key = getattr(plot, "x", None)
    y_keys = getattr(plot, "y", None)
    title = getattr(plot, "title", None)
    smoothing = getattr(plot, "smoothing_factor", None)
    log_y = getattr(plot, "log_y", None)
    range_y = getattr(plot, "range_y", None)

    if x_key is not None and y_keys is not None:
        return x_key, list(y_keys), title, smoothing, bool(log_y), range_y

    serialized = None
    for method_name in ("model_dump", "to_dict", "to_json"):
        method = getattr(plot, method_name, None)
        if callable(method):
            try:
                serialized = method()
                break
            except Exception:
                pass
    if serialized is None:
        serialized = getattr(plot, "__dict__", {})

    x_candidate = _find_key_recursive(serialized, "x")
    y_candidate = _find_key_recursive(serialized, "y")
    title_candidate = _find_key_recursive(serialized, "title")
    smoothing_candidate = _find_key_recursive(serialized, "smoothing_factor")
    log_y_candidate = _find_key_recursive(serialized, "log_y")
    range_y_candidate = _find_key_recursive(serialized, "range_y")

    final_x = x_key if x_key is not None else (x_candidate or "Step")
    final_y = y_keys if y_keys is not None else y_candidate
    if final_y is None:
        final_y = []
    if isinstance(final_y, str):
        final_y = [final_y]
    final_title = title if title is not None else (title_candidate or "Plot")
    final_smoothing = smoothing if smoothing is not None else smoothing_candidate
    final_log_y = bool(log_y) if log_y is not None else bool(log_y_candidate)
    final_range_y = range_y if range_y is not None else range_y_candidate
    return final_x, list(final_y), final_title, final_smoothing, final_log_y, final_range_y


def _history_rows(run, keys: list[str]) -> list[dict]:
    try:
        rows = run.history(keys=keys, pandas=False)
    except TypeError:
        rows = run.history(keys=keys)
    except Exception:
        rows = []

    if rows is None:
        return []
    if isinstance(rows, dict):
        return [rows]
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    return []


def _history_rows_sampled(run, keys: list[str], samples: int) -> list[dict]:
    try:
        rows = run.history(keys=keys, samples=samples, pandas=False)
    except TypeError:
        try:
            rows = run.history(keys=keys, samples=samples)
        except TypeError:
            rows = run.history(keys=keys)
    except Exception:
        rows = []

    if rows is None:
        return []
    if isinstance(rows, dict):
        return [rows]
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    return []


def _downsample_xy(x_values: np.ndarray, y_values: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or len(x_values) <= max_points:
        return x_values, y_values
    idx = np.linspace(0, len(x_values) - 1, num=max_points, dtype=int)
    return x_values[idx], y_values[idx]


def _compact_run_label(name: str, max_len: int = 26) -> str:
    compact = (name or "").replace("_big_vocab_", "_bv_")
    compact = compact.replace("text_ks_4_default_init_with_residual_trainable_", "canon_")
    compact = compact.replace("text_llama_", "llama_")
    if len(compact) <= max_len:
        return compact
    head = max_len // 2 - 2
    tail = max_len - head - 3
    return f"{compact[:head]}...{compact[-tail:]}"


def _short_metric_label(y_key: str) -> str:
    parts = y_key.split("/")
    if not parts:
        return y_key
    if parts[-1].startswith("shift_"):
        return parts[-1]
    if parts[-1].startswith("layer_"):
        return parts[-1]
    if len(parts) >= 2 and parts[-1] in {"out", "in"}:
        return "/".join(parts[-2:])
    return parts[-1]


def _normalize_output_name(output_name: str | None) -> str:
    if output_name is None:
        return ""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", output_name.strip())
    cleaned = cleaned.strip("._")
    if not cleaned:
        return ""
    if not cleaned.endswith(".png"):
        cleaned = f"{cleaned}.png"
    return cleaned


def to_matplotlib(
    plots: list[wr.LinePlot],
    runs,
    num_subplots_h: int,
    num_subplots_w: int,
    h: float,
    w: float,
    api_timeout: int = 60,
    max_points_per_series: int = 500,
    use_scan_fallback: bool = False,
    show_legend: bool = True,
    max_legend_items: int = 12,
    output_name: str | None = None,
):
    api = wandb.Api(timeout=api_timeout)
    fig, axes = plt.subplots(num_subplots_h, num_subplots_w, figsize=(w, h), squeeze=False)
    axes_flat = axes.flatten()

    runsets = runs if isinstance(runs, list) else [runs]
    run_infos: list[tuple] = []
    for runset in runsets:
        wb_runs, color_token_by_run_id = _resolve_runs_for_runset(api, runset)
        custom_colors = getattr(runset, "custom_run_colors", {}) or {}
        for run in wb_runs:
            token = color_token_by_run_id.get(run.id)
            run_color = (
                custom_colors.get(run.name)
                or custom_colors.get(getattr(run, "display_name", ""))
                or (custom_colors.get(token) if token else None)
            )
            run_infos.append((run, run_color))
    print(f"[to_matplotlib] Loaded {len(run_infos)} runs")

    for idx, plot in enumerate(plots):
        if idx >= len(axes_flat):
            break
        ax = axes_flat[idx]
        x_key, y_keys, title, smoothing, log_y, range_y = _extract_plot_spec(plot)
        if not title:
            title = f"Plot {idx + 1}"
        points_plotted = 0

        metric_palette = plt.cm.tab20(np.linspace(0, 1, max(2, len(y_keys))))
        run_palette = plt.cm.tab10(np.linspace(0, 1, max(2, len(run_infos))))
        line_styles = ["-", "--", "-.", ":"]

        for run_idx, (wb_run, run_color) in enumerate(run_infos):
            run_label = _compact_run_label(getattr(wb_run, "display_name", wb_run.name) or wb_run.name)
            for y_idx, y_key in enumerate(y_keys):
                history_keys = [y_key] if x_key == "Step" else [x_key, y_key]
                rows = _history_rows_sampled(wb_run, history_keys, max_points_per_series)
                if not rows and x_key != "Step":
                    rows = _history_rows_sampled(wb_run, [y_key], max_points_per_series)
                if not rows and use_scan_fallback:
                    rows = list(wb_run.scan_history(keys=history_keys))
                    if not rows and x_key != "Step":
                        rows = list(wb_run.scan_history(keys=[y_key]))
                    if not rows:
                        continue
                y_present = any(y_key in row for row in rows)
                if not y_present:
                    continue
                x_values = _to_float_array(
                    [row.get(x_key, row.get("_step", idx)) for idx, row in enumerate(rows)]
                )
                y_values = _to_float_array([row.get(y_key) for row in rows])
                mask = np.isfinite(x_values) & np.isfinite(y_values)
                if not np.any(mask):
                    continue
                x_valid = x_values[mask]
                y_valid = y_values[mask]
                x_valid, y_valid = _downsample_xy(x_valid, y_valid, max_points_per_series)
                if len(y_valid) > 1:
                    y_valid = _smooth_series(y_valid, smoothing)
                if len(y_keys) > 1:
                    color = metric_palette[y_idx % len(metric_palette)]
                    linestyle = line_styles[run_idx % len(line_styles)]
                    label = (
                        f"{run_label}:{_short_metric_label(y_key)}"
                        if len(run_infos) > 1
                        else _short_metric_label(y_key)
                    )
                    linewidth = 1.2
                else:
                    color = run_color or run_palette[run_idx % len(run_palette)]
                    linestyle = "-"
                    label = run_label
                    linewidth = 1.5
                ax.plot(x_valid, y_valid, label=label, color=color, linestyle=linestyle, linewidth=linewidth)
                points_plotted += len(x_valid)

        ax.set_title(title)
        ax.set_xlabel(x_key)
        ax.set_ylabel("value")
        if log_y:
            ax.set_yscale("log")
        if range_y is not None:
            ymin, ymax = range_y
            ax.set_ylim(
                bottom=float(ymin) if ymin is not None else None,
                top=float(ymax) if ymax is not None else None,
            )
        ax.grid(alpha=0.25, linestyle="--")
        if ax.lines and show_legend:
            handles, labels = ax.get_legend_handles_labels()
            if len(labels) > max_legend_items:
                handles = handles[:max_legend_items]
                labels = labels[:max_legend_items]
            legend_cols = 2 if len(labels) > 8 else 1
            ax.legend(
                handles,
                labels,
                fontsize=7,
                ncol=legend_cols,
                framealpha=0.8,
                loc="best",
            )
        else:
            metrics_text = ", ".join(y_keys) if y_keys else "none"
            ax.text(
                0.5,
                0.5,
                f"No data found\nmetrics: {metrics_text}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=9,
            )
            print(
                f"[to_matplotlib] Empty subplot '{title}' | "
                f"runs={len(run_infos)} | y_keys={y_keys} | points={points_plotted}"
            )

    for ax in axes_flat[len(plots):]:
        ax.axis("off")

    fig.tight_layout()

    output_dir = Path(__file__).parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_output_name = _normalize_output_name(output_name)
    if normalized_output_name:
        output_path = output_dir / normalized_output_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"wandb_plots_{timestamp}.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")

    try:
        from IPython.display import display
        display(fig)
    except Exception:
        pass

    return fig, str(output_path)

