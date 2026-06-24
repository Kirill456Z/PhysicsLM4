import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb


TASK_NAMES = [
    "concomp_factor_edges_list",
    "concomp_factor_adj_list",
    "depo_edges_list",
    "depo_adj_list",
    "shortest_path_edges_list",
    "shortest_path_adj_list",
    "bfs_edges_list",
    "bfs_adj_list",
]

# Primary "interesting" metric per task for the overview plot
TASK_PRIMARY_METRIC = {
    "depo_edges_list": "hop_4/accuracy",
    "depo_adj_list": "hop_4/accuracy",
    "concomp_factor_edges_list": "prefix_accuracy",
    "concomp_factor_adj_list": "prefix_accuracy",
    "shortest_path_edges_list": "prefix_accuracy",
    "shortest_path_adj_list": "prefix_accuracy",
    "bfs_edges_list": "prefix_accuracy",
    "bfs_adj_list": "prefix_accuracy",
}

# All per-task sub-metrics
TASK_METRICS = {
    "depo_edges_list": [f"hop_{h}/accuracy" for h in [1, 2, 4, 8]],
    "depo_adj_list": [f"hop_{h}/accuracy" for h in [1, 2, 4, 8]],
    "concomp_factor_edges_list": ["accuracy", "prefix_accuracy", "component_recall"],
    "concomp_factor_adj_list": ["accuracy", "prefix_accuracy", "component_recall"],
    "shortest_path_edges_list": ["set_accuracy", "prefix_accuracy", "is_correct_path"],
    "shortest_path_adj_list": ["set_accuracy", "prefix_accuracy", "is_correct_path"],
    "bfs_edges_list": ["set_recall", "prefix_accuracy"],
    "bfs_adj_list": ["set_recall", "prefix_accuracy"],
}


# ---------------------------------------------------------------------------
# Wandb key helpers
# ---------------------------------------------------------------------------

def wandb_eval_key(task_name: str, metric_name: str) -> str:
    return f"evals/synthetic/{task_name}/{metric_name}"


def get_wandb_metric_keys(task_names: list[str] | None = None) -> list[str]:
    """Return all wandb metric keys (loss + all eval metrics)."""
    if task_names is None:
        task_names = TASK_NAMES
    keys = ["loss/out"]
    for task in task_names:
        for metric in TASK_METRICS.get(task, []):
            keys.append(wandb_eval_key(task, metric))
    return keys


# ---------------------------------------------------------------------------
# Model size estimation from run config
# ---------------------------------------------------------------------------

def compute_model_params(config: dict) -> int:
    """
    Estimate total trainable parameters from a lingua run config dict.
    Handles the nested {"model": {...}} structure stored in wandb.
    """
    model_cfg = config.get("model", config)  # flat or nested
    dim = int(model_cfg.get("dim", 0))
    n_layers = int(model_cfg.get("n_layers", 0))
    vocab_size = int(model_cfg.get("vocab_size", 512))

    if dim == 0 or n_layers == 0:
        return 0

    # Embedding table (input + output weight tie)
    embed_params = vocab_size * dim
    # Attention: Q, K, V, O each dim×dim
    attn_params = 4 * dim * dim
    # FFN: two linear layers, intermediate = 4·dim  →  2 × dim × 4·dim
    ffn_params = 8 * dim * dim
    # LayerNorm (2 per layer): 2 × dim weights + biases ≈ 4·dim
    ln_params = 4 * dim
    # Canon convolution weights per layer
    canon_set = model_cfg.get("canon_set", None)
    canon_kernel = int(model_cfg.get("canon_kernel", 0))
    canon_multipliers = {"ABCD": 7, "AC": 2}
    canon_params = canon_multipliers.get(canon_set, 0) * dim * canon_kernel
    layer_params = attn_params + ffn_params + ln_params + canon_params
    return embed_params + n_layers * layer_params


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def _make_api(api_key: str) -> wandb.Api:
    return wandb.Api(api_key=api_key)


def _is_nan_scalar(v) -> bool:
    """True only for scalar NaN/None — safe to call on any value including dicts."""
    if isinstance(v, (dict, list)):
        return False
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _fetch_history(run, samples: int = 2000) -> pd.DataFrame:
    """Fetch run history sorted by _step. Returns empty DataFrame on failure."""
    hist = run.history(samples=samples)
    if not hist.empty and "_step" in hist.columns:
        hist = hist.sort_values("_step")
    return hist


def _fill_nan_from_history(summary: dict, hist: pd.DataFrame, run_name: str = "") -> None:
    """
    Fill missing or NaN metric values in summary from a pre-fetched history DF.

    Two cases are handled:
    - NaN scalar values already present as keys (partial summary)
    - Keys missing entirely (crashed run whose summary was never finalised)

    Modifies summary in-place.
    """
    nan_keys = {k for k, v in summary.items() if _is_nan_scalar(v)}
    summary_incomplete = "loss/out" not in summary

    if not nan_keys and not summary_incomplete:
        return

    if hist.empty:
        print(f"  [fallback] history is empty for {run_name}, cannot recover values")
        return

    tag = f"incomplete ({len(summary)} keys)" if summary_incomplete else f"{len(nan_keys)} NaN keys"
    print(f"  [fallback] summary {tag} — filling from history …")

    filled = 0
    for col in hist.columns:
        if col.startswith("_"):
            continue
        if col in nan_keys or col not in summary:
            non_null = hist[col].dropna()
            if not non_null.empty:
                last_step = hist.loc[non_null.index[-1], "_step"] if "_step" in hist.columns else "?"
                summary[col] = non_null.iloc[-1]
                filled += 1
                print(f"  [fallback] '{col}' = {summary[col]!r}  (step {last_step})")

    print(f"  [fallback] filled {filled} keys (last step in history: {hist['_step'].max() if '_step' in hist.columns else '?'})")


def debug_run_summary(run_id: str, api_key: str, entity: str, project: str) -> None:
    """
    Print a full diagnostic of the summary and history for a single run.
    Use this to investigate missing / NaN metric values.
    """
    api = _make_api(api_key)
    run = api.run(f"{entity}/{project}/{run_id}")
    summary = dict(run.summary)

    print(f"=== summary keys for {run.name} ===")
    for k, v in sorted(summary.items()):
        tag = "  NaN" if _is_nan_scalar(v) else ""
        print(f"  {k!r:60s} = {v!r}{tag}")

    nan_keys = [k for k, v in summary.items() if _is_nan_scalar(v)]
    print(f"\n{len(nan_keys)} NaN scalar keys: {nan_keys}\n")

    print("=== fetching history (samples=2000, no key filter) ===")
    hist = _fetch_history(run, samples=2000)
    print(f"  history shape: {hist.shape}")
    if not hist.empty:
        if "_step" in hist.columns:
            print(f"  step range: {hist['_step'].min()} – {hist['_step'].max()}")
        print(f"  history columns ({len(hist.columns)}): {sorted(hist.columns.tolist())}")
        print("\n  non-null counts for NaN summary keys:")
        for k in nan_keys:
            if k in hist.columns:
                nn = hist[k].dropna()
                last_val = nn.iloc[-1] if not nn.empty else "—"
                print(f"    {k!r}: {len(nn)} non-null rows, last value = {last_val!r}")
            else:
                print(f"    {k!r}: NOT IN HISTORY")


LOSS_AVG_STEPS = 100  # number of trailing steps used to smooth loss/out


def _fetch_run_summary(run_id: str, api: wandb.Api, entity: str, project: str) -> dict:
    run = api.run(f"{entity}/{project}/{run_id}")
    cfg = dict(run.config)
    summary = dict(run.summary)

    # Fetch history once — used for both crash recovery and loss smoothing.
    # We request enough samples to cover the last LOSS_AVG_STEPS loss points
    # plus eval checkpoints; 2000 is generous for a typical 80k-step run.
    hist = _fetch_history(run, samples=2000)

    # Fill missing / NaN summary values (handles crashed runs)
    _fill_nan_from_history(summary, hist, run_name=run.name)

    # Replace summary loss with a trailing average to reduce noise
    if not hist.empty and "loss/out" in hist.columns:
        loss_vals = hist["loss/out"].dropna()
        if not loss_vals.empty:
            n = min(LOSS_AVG_STEPS, len(loss_vals))
            summary["loss/out"] = float(loss_vals.iloc[-n:].mean())
            print(f"  loss/out = {summary['loss/out']:.4f}  (avg of last {n} steps)")

    return {
        "run_id": run_id,
        "name": run.name,
        "config": cfg,
        "summary": summary,
        "n_params": compute_model_params(cfg),
    }


def fetch_runs_results(
    run_ids: dict[str, list[str]] | list[str],
    api_key: str,
    entity: str,
    project: str,
) -> pd.DataFrame:
    """
    Fetch wandb summary data for multiple runs and return as a flat DataFrame.

    run_ids can be:
      - a dict  {label: [run_id, ...], ...}  — adds a "label" column used for
        per-architecture colouring in the plot helpers
      - a plain list [run_id, ...]  — no "label" column is added

    Each row is one run. Columns are run_id, run_name, n_params (+ label when
    a dict is given) plus every metric key from the run's wandb summary.
    Missing metrics for a run appear as NaN.
    """
    if isinstance(run_ids, list):
        run_ids = {"": run_ids}  # single group, no label column in output

    api = _make_api(api_key)
    rows = []
    for label, ids in run_ids.items():
        print(f"\n[{label}]")
        for run_id in ids:
            try:
                data = _fetch_run_summary(run_id, api, entity, project)
                row: dict = {
                    "run_id": data["run_id"],
                    "run_name": data["name"],
                    "label": label,
                    "n_params": data["n_params"],
                }
                row.update(data["summary"])
                rows.append(row)
                print(f"  {data['name']:40s}  n_params={data['n_params']:>12,}")
            except Exception as exc:
                print(f"[warn] run {run_id}: {exc}")

    df = pd.DataFrame(rows)
    # Drop the label column when there was only one (unlabelled) group
    if "label" in df.columns and df["label"].eq("").all():
        df = df.drop(columns=["label"])
    return df


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _setup_log_axes(ax: plt.Axes, log_x: bool, log_y: bool) -> None:
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("Model parameters")
    ax.grid(True, which="both", alpha=0.3, linestyle="--")


def plot_scaling_law(
    results_df: pd.DataFrame,
    metric_key: str,
    ax: plt.Axes | None = None,
    log_x: bool = True,
    log_y: bool = True,
    title: str | None = None,
    annotate: bool = True,
) -> plt.Axes:
    """
    Scatter + line plot of a metric value vs model size.

    When results_df contains a "label" column (produced by passing a dict to
    fetch_runs_results), each label gets its own colour and a legend is shown.

    Parameters
    ----------
    results_df : output of fetch_runs_results (one row per run)
    metric_key : column name, e.g. "loss/out" or
                 "evals/synthetic/depo_edges_list/hop_4/accuracy"
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    if metric_key not in results_df.columns:
        ax.set_title(f"{metric_key} — no data")
        return ax

    has_labels = "label" in results_df.columns

    if has_labels:
        colors = [p["color"] for p in plt.rcParams["axes.prop_cycle"]]
        any_data = False
        for i, (label, group) in enumerate(results_df.groupby("label", sort=False)):
            df = group.dropna(subset=[metric_key]).sort_values("n_params")
            if df.empty:
                continue
            any_data = True
            xs = df["n_params"].to_numpy(dtype=float)
            ys = df[metric_key].to_numpy(dtype=float)
            color = colors[i % len(colors)]
            ax.scatter(xs, ys, zorder=3, color=color, label=label)
            ax.plot(xs, ys, "--", alpha=0.45, color=color)
        if not any_data:
            ax.set_title(f"{metric_key} — no data")
            return ax
        ax.legend(fontsize=9)
    else:
        df = results_df.dropna(subset=[metric_key]).sort_values("n_params")
        if df.empty:
            ax.set_title(f"{metric_key} — no data")
            return ax
        xs = df["n_params"].to_numpy(dtype=float)
        ys = df[metric_key].to_numpy(dtype=float)
        ax.scatter(xs, ys, zorder=3)
        ax.plot(xs, ys, "--", alpha=0.45)
        if annotate:
            for x, y, lbl in zip(xs, ys, df["run_name"].tolist()):
                ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=8)

    _setup_log_axes(ax, log_x, log_y)
    ax.set_ylabel(metric_key)
    ax.set_title(title or metric_key)
    return ax


def plot_custom_metric(
    results_df: pd.DataFrame,
    metric_values: pd.Series,
    metric_name: str = "custom_metric",
    ax: plt.Axes | None = None,
    log_x: bool = True,
    log_y: bool = False,
    title: str | None = None,
    annotate: bool = True,
) -> plt.Axes:
    """
    Plot any precomputed metric Series against model size.

    metric_values must be aligned with results_df's index, e.g.:
        plot_custom_metric(results_df, results_df["a"] + results_df["b"], "a+b")

    When results_df contains a "label" column each label gets its own colour
    and a legend is shown — same behaviour as plot_scaling_law.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    meta_cols = ["n_params"] + (["label"] if "label" in results_df.columns else [])
    df = results_df[meta_cols].copy()
    df["_metric"] = metric_values
    df = df.dropna(subset=["_metric"])

    if df.empty:
        ax.set_title(f"{metric_name} — no data")
        return ax

    has_labels = "label" in df.columns
    if has_labels:
        colors = [p["color"] for p in plt.rcParams["axes.prop_cycle"]]
        for i, (label, group) in enumerate(df.groupby("label", sort=False)):
            g = group.sort_values("n_params")
            if g.empty:
                continue
            xs = g["n_params"].to_numpy(dtype=float)
            ys = g["_metric"].to_numpy(dtype=float)
            color = colors[i % len(colors)]
            ax.scatter(xs, ys, zorder=3, color=color, label=label)
            ax.plot(xs, ys, "--", alpha=0.45, color=color)
        ax.legend(fontsize=9)
    else:
        g = df.sort_values("n_params")
        xs = g["n_params"].to_numpy(dtype=float)
        ys = g["_metric"].to_numpy(dtype=float)
        ax.scatter(xs, ys, zorder=3)
        ax.plot(xs, ys, "--", alpha=0.45)
        if annotate and "run_name" in results_df.columns:
            for x, y, lbl in zip(xs, ys, results_df.loc[g.index, "run_name"].tolist()):
                ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=8)

    _setup_log_axes(ax, log_x, log_y)
    ax.set_ylabel(metric_name)
    ax.set_title(title or metric_name)
    return ax


def plot_task_metrics_grid(
    results_df: pd.DataFrame,
    task_name: str,
    log_x: bool = True,
    log_y: bool = True,
    figsize_per_plot: tuple[float, float] = (5.5, 4.0),
) -> plt.Figure:
    """Plot all sub-metrics for a single task in a grid."""
    metrics = TASK_METRICS.get(task_name, [])
    if not metrics:
        raise ValueError(f"Unknown task: {task_name}")

    ncols = min(3, len(metrics))
    nrows = (len(metrics) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows),
    )
    axes_flat = np.array(axes).flatten()

    for i, metric in enumerate(metrics):
        key = wandb_eval_key(task_name, metric)
        plot_scaling_law(results_df, key, ax=axes_flat[i], log_x=log_x, log_y=log_y)

    for ax in axes_flat[len(metrics):]:
        ax.set_visible(False)

    fig.suptitle(task_name, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def fetch_training_history(
    groups: dict[str, list[str]],
    metric_keys: list[str],
    api_key: str,
    entity: str,
    project: str,
    samples: int = 500,
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Fetch time-series training history for groups of runs.

    Parameters
    ----------
    groups : {group_name: [run_id, ...]}
    metric_keys : list of wandb metric names to fetch, e.g. ["loss/out"]
    samples : number of history points to request per run

    Returns
    -------
    {group_name: {run_id: DataFrame(columns=["_step"] + metric_keys)}}
    """
    api = _make_api(api_key)
    result: dict[str, dict[str, pd.DataFrame]] = {}
    for group, ids in groups.items():
        result[group] = {}
        print(f"\n[{group}]")
        for run_id in ids:
            try:
                run = api.run(f"{entity}/{project}/{run_id}")
                hist = run.history(samples=samples, keys=metric_keys)
                if "_step" not in hist.columns and not hist.empty:
                    hist = hist.reset_index().rename(columns={"index": "_step"})
                hist = hist.sort_values("_step").reset_index(drop=True)
                result[group][run_id] = hist
                print(f"  {run.name:55s}  rows={len(hist)}")
            except Exception as exc:
                print(f"  [warn] {run_id}: {exc}")
    return result


def find_runs_by_name_pattern(
    pattern: str,
    api_key: str,
    entity: str,
    project: str,
    state: str | None = "finished",
) -> list[str]:
    """
    Return wandb run IDs (short 8-char IDs) whose displayName contains `pattern`.
    Useful for discovering run IDs when you know the experiment folder/config name.
    """
    api = _make_api(api_key)
    filters = {"displayName": {"$regex": pattern}}
    if state:
        filters["state"] = state
    runs = api.runs(f"{entity}/{project}", filters=filters)
    ids = []
    for r in runs:
        print(f"  {r.name:60s}  id={r.id}  state={r.state}")
        ids.append(r.id)
    return ids


def plot_variance_curves(
    histories: dict[str, dict[str, pd.DataFrame]],
    metric_key: str,
    ax: plt.Axes | None = None,
    title: str | None = None,
    smoothing: int = 1,
    alpha_individual: float = 0.25,
    show_individual: bool = True,
    palette: list[str] | None = None,
) -> plt.Axes:
    """
    Plot training curves with mean ± std band per group.

    Parameters
    ----------
    histories : output of fetch_training_history
    metric_key : column name in the history DataFrames, e.g. "loss/out"
    smoothing : rolling-average window for smoothing (1 = no smoothing)
    alpha_individual : transparency for individual run lines
    show_individual : whether to draw individual run lines underneath the band
    palette : list of hex colours, one per group

    Each group gets one colour: thin semi-transparent lines for individual seeds
    and a solid mean line with ±1 std shaded band.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    default_palette = [p["color"] for p in plt.rcParams["axes.prop_cycle"]]
    colors = palette or default_palette

    for gi, (group_name, runs) in enumerate(histories.items()):
        color = colors[gi % len(colors)]
        dfs = []
        for run_id, hist in runs.items():
            if hist.empty or metric_key not in hist.columns:
                continue
            s = hist[["_step", metric_key]].dropna()
            if smoothing > 1:
                s = s.copy()
                s[metric_key] = s[metric_key].rolling(smoothing, min_periods=1).mean()
            if show_individual:
                ax.plot(s["_step"], s[metric_key], color=color, alpha=alpha_individual, linewidth=0.8)
            dfs.append(s.set_index("_step")[metric_key])

        if not dfs:
            continue

        combined = pd.concat(dfs, axis=1).sort_index().interpolate(method="index")
        mean = combined.mean(axis=1)
        std = combined.std(axis=1)
        ax.plot(mean.index, mean.values, color=color, linewidth=2.0, label=f"{group_name} (n={len(dfs)})")
        ax.fill_between(mean.index, mean - std, mean + std, color=color, alpha=0.15)

    ax.set_xlabel("Step")
    ax.set_ylabel(metric_key)
    ax.set_title(title or metric_key)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=9)
    return ax


def plot_final_metric_bars(
    histories: dict[str, dict[str, pd.DataFrame]],
    metric_key: str,
    ax: plt.Axes | None = None,
    title: str | None = None,
    trailing_steps: int = 50,
    palette: list[str] | None = None,
) -> plt.Axes:
    """
    Bar chart of final (trailing-average) metric values per group, with error bars.

    Each bar = group mean; error bar = ±1 std across seeds.
    Individual seed values plotted as scatter points on top.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    default_palette = [p["color"] for p in plt.rcParams["axes.prop_cycle"]]
    colors = palette or default_palette

    group_means, group_stds, group_seeds = [], [], []
    group_names = list(histories.keys())

    for group_name, runs in histories.items():
        vals = []
        for run_id, hist in runs.items():
            if hist.empty or metric_key not in hist.columns:
                continue
            series = hist[metric_key].dropna()
            if series.empty:
                continue
            n = min(trailing_steps, len(series))
            vals.append(float(series.iloc[-n:].mean()))
        group_means.append(np.mean(vals) if vals else np.nan)
        group_stds.append(np.std(vals) if len(vals) > 1 else 0.0)
        group_seeds.append(vals)

    xs = np.arange(len(group_names))
    for i, (name, mean, std, seeds) in enumerate(zip(group_names, group_means, group_stds, group_seeds)):
        color = colors[i % len(colors)]
        ax.bar(i, mean, yerr=std, color=color, alpha=0.75, capsize=5, label=name)
        jitter = np.random.default_rng(i).uniform(-0.12, 0.12, len(seeds))
        ax.scatter(i + jitter, seeds, color=color, zorder=3, s=30, edgecolors="white", linewidths=0.5)

    ax.set_xticks(xs)
    ax.set_xticklabels(group_names, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel(metric_key)
    ax.set_title(title or metric_key)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    return ax


def plot_all_tasks_primary_metrics(
    results_df: pd.DataFrame,
    task_names: list[str] | None = None,
    log_x: bool = True,
    log_y: bool = True,
    figsize_per_plot: tuple[float, float] = (5.5, 4.0),
) -> plt.Figure:
    """
    One subplot per task showing its primary metric vs model size.
    Useful as a quick overview across all tasks.
    """
    if task_names is None:
        task_names = TASK_NAMES

    ncols = min(4, len(task_names))
    nrows = (len(task_names) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows),
    )
    axes_flat = np.array(axes).flatten()

    for i, task in enumerate(task_names):
        metric = TASK_PRIMARY_METRIC.get(task, "accuracy")
        key = wandb_eval_key(task, metric)
        plot_scaling_law(results_df, key, ax=axes_flat[i], log_x=log_x, log_y=log_y, title=task)

    for ax in axes_flat[len(task_names):]:
        ax.set_visible(False)

    fig.suptitle("primary metric per task", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig
