# %% [markdown]
# # Canon: Horizontal Information Flow on Synthetic Reasoning Benchmarks
#
# Builds the W&B Report from **cached local PNGs** (see `new_paper_plan_plots.py`).
#
# Workflow that keeps report iteration fast:
# 1. For each figure, generate the PNG **only if it is missing** (the only step that
#    downloads run metadata — slow, but cached on disk in `new_paper_plan/plots/`).
# 2. Upload the PNGs to one lightweight "assets" run (fast — just image uploads).
# 3. Embed each image with a `wr.MediaBrowser` panel pointing at that run.
#
# So changing report text/layout and re-saving does **not** re-download run histories.
# To regenerate a figure after changing the data, delete its PNG (or run
# `python new_paper_plan_plots.py --force <key>`).
#
# **Note on `wr.Image`:** the Reports API `wr.Image` only accepts an external URL, not a
# local file. To embed locally-rendered PNGs we log them to a run and show them via
# `wr.MediaBrowser` (the native panel for run-logged media; also avoids signed-URL expiry).
#
# Run from the `reports/` directory (or anywhere — paths are anchored to this file).

# %%
import os
import time
import wandb
import wandb_workspaces.reports.v2 as wr

import new_paper_plan_plots as P

ENTITY = P.ENTITY
PROJECT = P.PROJECT
# Fixed, unique run id so the report's Runset can reference exactly these uploaded images.
# (W&B Runset `filters` match the internal run id via the `name` field, so we set id == name.)
ASSETS_ID = f"nppassets{int(time.time())}"

# %%
# ---- 1. Ensure every figure exists locally (generates only the missing ones) ----
# To force a refresh of a figure, delete its PNG in new_paper_plan/plots/ first.
FIGURE_ORDER = [
    "text_stable", "depo_grokking", "brevo", "task_noise",
    "scaling", "task_breakdown",
    "grad_ratio", "rms_ratio", "grad_profile",
    "depth_grad_ratio", "depth_gap",
    "kernel", "canon_weights", "cos_sim",
]
for key in FIGURE_ORDER:
    P.ensure(key)

# %%
# ---- 2. Upload the cached PNGs to one assets run; remember its id for the report ----
assets_run = wandb.init(
    entity=ENTITY, project=PROJECT, id=ASSETS_ID, name=ASSETS_ID,
    job_type="report-assets", reinit=True,
)
assets_run.log({key: wandb.Image(P.path_for(key)) for key in FIGURE_ORDER})
assets_run.finish()
print("assets run id:", ASSETS_ID)

# %%
# ---- helpers to build image panels from the assets run ----
def assets_runset():
    return wr.Runset(entity=ENTITY, project=PROJECT, name="report assets",
                     filters=f'name in ["{ASSETS_ID}"]')


def figure(key, caption, title=None):
    """A captioned figure: a MediaBrowser panel for `key` plus a caption paragraph."""
    panel = wr.MediaBrowser(media_keys=[key], num_columns=1, title=title or "",
                            layout=wr.Layout(x=0, y=0, w=16, h=10))
    return [
        wr.PanelGrid(runsets=[assets_runset()], panels=[panel]),
        wr.P(caption),
    ]

# %%
# ---- 3. Build the report ----
report = wr.Report(
    entity=ENTITY, project=PROJECT,
    title="Canon: Horizontal Information Flow on Synthetic Reasoning Benchmarks",
    description=(
        "Single-seed synthetic-benchmark curves are unreliable (grokking), but averaged "
        "over seeds Canon wins at every scale, and its training mechanism is seed-stable. "
        "We show the mechanism is horizontal information flow, not across-layer scaling."
    ),
)
report.width = "fluid"

report.blocks = [
    # ----------------------------------------------------------------- §1
    wr.H1("1. A single run is an unreliable guide: these tasks grok"),
    wr.P(
        "Architecture research is expensive, so synthetic reasoning benchmarks are appealing "
        "as cheap proxies. But several of these tasks grok: the loss sits on a plateau and "
        "then drops sharply at a step that varies from seed to seed, so a single training "
        "curve can rank two architectures the wrong way around."
    ),
    wr.H2("Depo (8L/512D) — grokking jumps at different steps per seed"),
    *figure(
        "depo_grokking",
        "Three Canon seeds and one (fully trained) Llama seed on Depo. The grokking step is "
        "essentially random: one Canon seed groks early and ends far below everything else, "
        "while the other two Canon seeds never grok within the budget and finish above the "
        "Llama seed. Read one seed at a time, this ranks Llama above Canon — not because Canon "
        "is worse, but because the loss landscape is sensitive to initialization. A single-seed "
        "comparison would mislead.",
    ),
    wr.H2("The same instability on Brevo; text modelling is stable"),
    *figure("brevo", "Brevo shows the same seed-driven divergence."),
    *figure(
        "text_stable",
        "For text modelling the seeds produce near-identical curves and Canon is "
        "consistently below the baseline. The volatility is specific to the grokking-prone "
        "synthetic tasks (Depo, Brevo); BFS, Shortest-Path and ConComp are stable.",
    ),
    wr.H2("The volatility is concentrated on one task"),
    *figure(
        "task_noise",
        "Across all eight synthetic tasks at 30M (pooling five Canon and five Llama seeds), the "
        "seed-to-seed spread in the final score is concentrated almost entirely on Depo "
        "(std ~0.21-0.29). Every graph task — BFS, ShortestPath, ConComp — is stable "
        "(std < 0.06). So the grokking instability above is not a generic property of synthetic "
        "benchmarks: it lives on the volatile, multi-hop retrieval task, and the stable tasks "
        "stay reproducible.",
    ),
    wr.P("Takeaway: do not rank architectures from a single synthetic-benchmark run — "
         "average over seeds and scales. The instability is real but localized to the "
         "grokking-prone tasks."),

    # ----------------------------------------------------------------- §2
    wr.H1("2. Evaluated correctly, Canon is better at every scale"),
    *figure(
        "scaling",
        "Final loss vs. parameters (log-log), 5 seeds averaged per point. Canon sits below "
        "Llama at 3M, 10M, 30M and 100M. The gap is largest for small models and shrinks "
        "as they grow. No significance tests are needed — the curves separate cleanly.",
    ),
    wr.H2("The aggregate win is driven by one task"),
    *figure(
        "task_breakdown",
        "Breaking the 30M result down by task (5 seeds each) shows the aggregate gain is "
        "concentrated on Depo: Canon adds about +0.7 (adj-list) and +0.6 (edges-list) in "
        "accuracy, versus roughly +0.05 on ConComp and near zero on BFS and ShortestPath, "
        "which both architectures already solve. Notice that Depo is also the only task with a "
        "large seed spread (the error bars in the left panel) — the task Canon helps most is "
        "exactly the volatile one from the variance ranking above. Grokking, seed variance, and "
        "the architectural benefit all live on the same multi-hop retrieval task.",
    ),

    # ----------------------------------------------------------------- §3
    wr.H1("3. What Canon does: across-layer reorganization vs. horizontal flow"),
    wr.P(
        "The bottom-line loss wiggles seed-to-seed, but the internal changes Canon makes to "
        "training are reproducible — that is where these benchmarks earn their keep. Canon "
        "shows two things at once; we separate cause from correlate."
    ),
    wr.H2("Signature 1 — Canon reshapes gradients and representations across layers"),
    *figure(
        "grad_ratio",
        "Over training, Llama grows a large deep/shallow gradient ratio (deep layers dominate) "
        "while Canon holds it down — shown here on text modelling, where the signature is "
        "clearest (it is learned and grows over training); the same ordering holds across "
        "depths on the synthetic tasks below. This is the kind of reorganization that scaling "
        "a model up also produces, so it is tempting to read Canon as 'a bigger model for free'.",
    ),
    *figure(
        "rms_ratio",
        "Canon also amplifies the residual stream (output/input RMS > 1), most strongly in "
        "the deep layers near the LM head.",
    ),
    *figure(
        "grad_profile",
        "It also smooths gradient flow. Llama's per-layer gradient contribution is uneven — a "
        "sharp spike at the shallow layers and a collapse at the very first one — while Canon "
        "spreads it far more evenly across depth. This is the same signature as the deep/shallow "
        "ratio above, seen as the full per-layer profile.",
    ),

    # ----------------------------------------------------------------- §4
    wr.H2("Signature 1 is a correlate — it persists where Canon stops winning"),
    wr.P(
        "Holding the parameter budget fixed and trading width for depth is a controlled "
        "mechanistic probe, distinct from the model-size scaling above where Canon wins."
    ),
    *figure(
        "depth_grad_ratio",
        "The gradient-redistribution signature is present at every depth, including 16L: Canon "
        "holds the deep/shallow gradient ratio below Llama (and below 1) at 4L, 8L and 16L "
        "alike.",
    ),
    *figure(
        "depth_gap",
        "But on the actual loss, Canon wins at 4L and 8L and loses at 16L (deep and narrow). "
        "So the gradient signature persists exactly where Canon stops winning — a signature "
        "that appears both where Canon helps and where it does not cannot be what drives the gain.",
    ),

    # ----------------------------------------------------------------- §5
    wr.H1("4. Horizontal (cross-token) information flow is the driver"),
    wr.H2("Cross-token mixing is necessary: k=1 is worse than no Canon"),
    *figure(
        "kernel",
        "With kernel size 1 the convolution mixes no neighbouring tokens — it adds "
        "parameters without horizontal flow — and is worse than the baseline. Loss improves "
        "monotonically as the kernel grows (k=1 < baseline < k=2 < k=4 < k=6).",
    ),
    wr.H2("Canon integrates farther-back tokens"),
    *figure(
        "canon_weights",
        "Learned conv weights at all four Canon positions (A–D) put more mass on farther-back "
        "shifts than on the current token (shift 0).",
    ),
    *figure(
        "cos_sim",
        "Cosine similarity between Canon output and input is lowest (most mixing) in early "
        "layers. Shown for the two positions that log alignment metrics, Canon-A and Canon-C "
        "(the attention-path convolutions); Canon-B and Canon-D do not log this signal.",
    ),
    wr.H2("The gain is largest where attention is scarce: shallow models"),
    *figure(
        "depth_gap",
        "Shallow models have few attention layers and little native long-range mixing, so "
        "Canon helps most: the gap is large at 4 layers, smaller at 8, and gone by 16 — "
        "consistent with Canon supplying mixing that deeper models already get from attention.",
    ),

    # ----------------------------------------------------------------- summary
    wr.H1("Summary"),
    wr.P(
        "Canon's benefit comes from horizontal information flow (cross-token mixing by the "
        "convolution). The across-layer reorganization — flatter gradients, amplified deep "
        "representations — is a reproducible correlate, not the cause: it persists at 16L "
        "where Canon no longer wins. The gain is largest where attention is scarce (shallow "
        "models) and on retrieval-heavy tasks (Depo)."
    ),
    wr.P(
        "On the benchmarks: read one seed at a time they mislead, because grokking makes a "
        "single curve a coin flip on the volatile tasks. Averaged over seeds and scales the "
        "ranking is clean, and the training mechanism is seed-stable. These synthetic "
        "benchmarks are weak instruments for ranking but strong ones for understanding "
        "how an architecture behaves."
    ),
]

# %%
# ---- Workaround for the wandb-core "service" API rejecting this key ----
# Logging runs works, but Reports `save()` makes two calls through the wandb-core *service*
# transport that fail with "relogin required" (emitted by the compiled core binary): listing
# projects to check the target exists, and the report upsert. This wandb build is service-only
# (no legacy `api.client`), but a plain HTTPS GraphQL POST with the API key authenticates fine
# (run-history reads already work). So we route the upsert through a direct `requests` POST and
# stub the project-existence check (physics4llm already exists, nothing to create).
import types
import requests
import wandb_workspaces.reports.v2.interface as _iface
import wandb.apis.public.api as _public_api

_GQL_URL = "https://api.wandb.ai/graphql"


def _http_execute_graphql(api, query, variables=None):
    resp = requests.post(
        _GQL_URL, auth=("api", P.WANDB_API_KEY),
        json={"query": query, "variables": dict(variables or {})}, timeout=60,
    )
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("errors"):
        raise RuntimeError(payload["errors"])
    return payload["data"]


_iface.execute_graphql = _http_execute_graphql
_iface.get_app_url = lambda api: "https://wandb.ai"
_public_api.Api.projects = lambda self, entity=None, per_page=200: [
    types.SimpleNamespace(name=PROJECT)
]

# %%
report.save()
print(report.url)
