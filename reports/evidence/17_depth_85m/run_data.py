"""
Depth sweep at fixed ~85M params — Canon substitutes for depth.

Param-matched: width shrinks as layers grow (10L/832D, 12L/768D, 14L/704D),
holding total params ~85M. Single seed per config, 80k steps, 8-task evals.
Runs (llama_diff_all_tasks_*):
  Llama 10/12/14L : 0.4.{141,142,143}
  Canon 10/12L    : 0.4.{163,164}   (Canon 14L 0.4.165 failed to train)

Story: Llama needs depth to solve multi-hop retrieval (Depo); a Canon model
reaches the same capability several layers shallower. Canon-10L matches Llama-14L.

Outputs depth_85m.png into the paper's plots/ dir.
"""
import os, numpy as np, matplotlib.pyplot as plt, wandb

WANDB_API_KEY = "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"
ENTITY, PROJECT = "kirill456z", "physics4llm"
PAPER_PLOTS = os.path.normpath(os.path.join(os.path.dirname(__file__),
                               "..", "..", "new_paper_plan", "plots"))
C_CANON, C_LLAMA = "#d6604d", "#2166ac"

RUNS = {
    ("Llama", 10): "llama_diff_all_tasks_llama_10_layers_0.4.141",
    ("Llama", 12): "llama_diff_all_tasks_llama_12_layers_0.4.142",
    ("Llama", 14): "llama_diff_all_tasks_llama_14_layers_0.4.143",
    ("Canon", 10): "llama_diff_all_tasks_canon_llama_10_layers_0.4.163",
    ("Canon", 12): "llama_diff_all_tasks_canon_llama_12_layers_0.4.164",
}
DEPO = ["evals/synthetic/depo_edges_list/hop_4/accuracy",
        "evals/synthetic/depo_adj_list/hop_4/accuracy"]

def fetch(api, rid):
    r = api.run(f"{ENTITY}/{PROJECT}/{rid}")
    h = r.history(samples=600, keys=["loss/out"])
    loss = float(h["loss/out"].dropna().iloc[-100:].mean())
    s = r.summary._json_dict
    depo = float(np.nanmean([s.get(k, np.nan) for k in DEPO]))
    return loss, depo

def main():
    api = wandb.Api(api_key=WANDB_API_KEY)
    data = {k: fetch(api, v) for k, v in RUNS.items()}
    for (arch, nl), (loss, depo) in data.items():
        print(f"  {arch:6s} {nl}L  loss={loss:.4f}  depo_h4={depo:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for arch, color in [("Llama", C_LLAMA), ("Canon", C_CANON)]:
        depths = [d for (a, d) in data if a == arch]
        ax = axes[0]
        ax.plot(depths, [data[(arch, d)][0] for d in depths], "-o" if arch=="Llama" else "--s",
                color=color, lw=2, ms=8, label=arch)
        ax = axes[1]
        ax.plot(depths, [data[(arch, d)][1] for d in depths], "-o" if arch=="Llama" else "--s",
                color=color, lw=2, ms=8, label=arch)

    # annotate Canon-10L ~ Llama-14L on the loss panel
    axes[0].annotate("Canon 10L $\\approx$ Llama 14L",
                     xy=(10, data[("Canon",10)][0]), xytext=(10.3, 0.16),
                     fontsize=8.5, color="gray",
                     arrowprops=dict(arrowstyle="->", color="gray", lw=1))
    axes[0].set_xlabel("Number of layers"); axes[0].set_ylabel("Loss/out (trailing-100 mean)")
    axes[0].set_title("Aggregate loss")
    axes[1].set_xlabel("Number of layers"); axes[1].set_ylabel("Depo hop-4 accuracy (edges+adj mean)")
    axes[1].set_title("Multi-hop retrieval (Depo)")
    for ax in axes:
        ax.set_xticks([10, 12, 14]); ax.grid(alpha=0.3, ls="--"); ax.legend(fontsize=9)
    fig.suptitle("Depth sweep at fixed ~85M params: Canon substitutes for depth",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    os.makedirs(PAPER_PLOTS, exist_ok=True)
    path = os.path.join(PAPER_PLOTS, "depth_85m.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"Saved: {path}")

if __name__ == "__main__":
    main()
