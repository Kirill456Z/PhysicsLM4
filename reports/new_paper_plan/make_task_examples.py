"""Generate illustrative example graphs for each benchmark task.

One small graph per task, highlighting the query node(s) and answer node(s),
with the expected model output shown underneath. These are hand-laid
illustrations (not sampled from the generator) chosen to be readable on a slide.

Saves to reports/new_paper_plan/plots/:
  task_example_depo.png
  task_example_concomp.png
  task_example_shortpath.png
  task_example_bfs.png
"""
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

OUT = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------------ palette
C_QUERY = "#d62728"   # query node  (red)
C_ANS = "#2ca02c"     # answer node (green)
C_PLAIN = "#cfd8e3"   # other nodes (light blue-grey)
C_EDGE = "#5a6a85"
C_PATH = "#2ca02c"
NODE_SIZE = 1400
FONT = 13


def _draw(G, pos, ax, query, answers, *, directed=False,
          path_edges=None, ans_is_path=False):
    node_colors = []
    for n in G.nodes():
        if n in query:
            node_colors.append(C_QUERY)
        elif n in answers:
            node_colors.append(C_ANS)
        else:
            node_colors.append(C_PLAIN)

    nx.draw_networkx_edges(
        G, pos, ax=ax, edge_color=C_EDGE, width=1.6,
        arrows=directed, arrowsize=18,
        connectionstyle="arc3,rad=0.0", node_size=NODE_SIZE,
    )
    if path_edges:
        nx.draw_networkx_edges(
            G, pos, ax=ax, edgelist=path_edges, edge_color=C_PATH, width=3.6,
            arrows=directed, arrowsize=20, node_size=NODE_SIZE,
        )
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=node_colors, node_size=NODE_SIZE,
        edgecolors="#33405a", linewidths=1.5,
    )
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=FONT, font_weight="bold")
    ax.set_axis_off()
    # Asymmetric padding: extra headroom on top so the title/legend never
    # overlap the graph, and a tighter bottom to absorb empty space.
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    xpad = 0.12 * (max(xs) - min(xs) + 1e-9)
    yspan = max(ys) - min(ys) + 1e-9
    ax.set_xlim(min(xs) - xpad, max(xs) + xpad)
    ax.set_ylim(min(ys) - 0.10 * yspan, max(ys) + 0.55 * yspan)


def _legend(ax, items):
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                   markeredgecolor="#33405a", markersize=12, label=l)
        for l, c in items
    ]
    ax.legend(handles=handles, loc="upper center", ncol=len(items),
              frameon=False, fontsize=11, bbox_to_anchor=(0.5, 1.02),
              handletextpad=0.2, columnspacing=1.0)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ============================================================== Depo
# Directed permutation graph: every node has exactly one successor (cycles).
# query = q, num_hops = 2 -> follow 2 outgoing edges.
def depo():
    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    # one 6-cycle:  A->B->C->D->E->F->A
    seq = ["A", "B", "C", "D", "E", "F"]
    G = nx.DiGraph()
    for i in range(len(seq)):
        G.add_edge(seq[i], seq[(i + 1) % len(seq)])
    pos = nx.circular_layout(G)
    query = ["A"]
    # 2 hops from A: A->B->C
    answers = ["C"]
    path_edges = [("A", "B"), ("B", "C")]
    _draw(G, pos, ax, query, answers, directed=True, path_edges=path_edges)
    _legend(ax, [("query  q", C_QUERY), ("answer  a", C_ANS)])
    ax.set_title(r"Depo:  $\langle$query$_2\rangle$  A  $\rightarrow$  C",
                 fontsize=14, pad=14)
    save(fig, "task_example_depo.png")


# ============================================================== ConComp
# Undirected random graph, several components. Answer = lexicographically
# smallest node per component, in traversal order.
def concomp():
    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    G = nx.Graph()
    comp1 = [("B", "E"), ("E", "G"), ("B", "G")]
    comp2 = [("A", "D"), ("D", "F")]
    comp3 = [("C", "H")]
    for e in comp1 + comp2 + comp3:
        G.add_edge(*e)
    pos = {
        "B": (-1.0, 1.0), "E": (-1.7, 0.2), "G": (-0.6, 0.2),
        "A": (0.6, 1.0), "D": (1.2, 0.3), "F": (0.4, 0.1),
        "C": (1.6, -0.9), "H": (0.7, -0.9),
    }
    query = []
    answers = ["B", "A", "C"]  # min per component
    _draw(G, pos, ax, query, answers, directed=False)
    _legend(ax, [("component min (answer)", C_ANS)])
    ax.set_title(r"ConComp:  $\rightarrow$  A  B  C",
                 fontsize=14, pad=14)
    save(fig, "task_example_concomp.png")


# ============================================================== ShortPath
# Undirected random graph. Output intermediate nodes on shortest s->t path.
def shortpath():
    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    G = nx.Graph()
    # Unique shortest path S-A-C-T (len 3); alternative S-B-D-C-T is longer (len 4).
    edges = [
        ("S", "A"), ("S", "B"), ("A", "C"),
        ("C", "T"), ("B", "D"), ("D", "C"), ("A", "E"),
    ]
    G.add_edges_from(edges)
    pos = {
        "S": (-2.0, 0.0), "A": (-1.0, 0.9), "B": (-1.0, -0.9),
        "C": (0.2, 0.6), "D": (0.2, -0.9), "E": (-1.4, 1.9),
        "T": (1.4, 0.0),
    }
    query = ["S", "T"]
    # shortest path S-A-C-T  (length 3): intermediate = A, C
    answers = ["A", "C"]
    path_edges = [("S", "A"), ("A", "C"), ("C", "T")]
    _draw(G, pos, ax, query, answers, directed=False, path_edges=path_edges)
    _legend(ax, [("start / target", C_QUERY), ("path node (answer)", C_ANS)])
    ax.set_title(r"ShortPath:  S  T  $\rightarrow$  A  C",
                 fontsize=14, pad=14)
    save(fig, "task_example_shortpath.png")


# ============================================================== BFS
# Undirected random graph. Enumerate nodes reachable from q in BFS order.
def bfs():
    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    G = nx.Graph()
    edges = [
        ("Q", "A"), ("Q", "B"), ("A", "C"), ("B", "C"),
        ("C", "D"), ("X", "Y"),  # separate component, not reachable
    ]
    G.add_edges_from(edges)
    pos = {
        "Q": (-1.6, 0.0), "A": (-0.6, 0.9), "B": (-0.6, -0.9),
        "C": (0.5, 0.0), "D": (1.6, 0.0),
        "X": (0.2, 1.9), "Y": (1.4, 1.7),
    }
    query = ["Q"]
    answers = ["A", "B", "C", "D"]  # BFS order from Q
    _draw(G, pos, ax, query, answers, directed=False)
    _legend(ax, [("query  q", C_QUERY), ("reachable (answer)", C_ANS)])
    ax.set_title(r"BFS:  $\langle$query$\rangle$  Q  $\rightarrow$  A  B  C  D",
                 fontsize=14, pad=14)
    save(fig, "task_example_bfs.png")


if __name__ == "__main__":
    depo()
    concomp()
    shortpath()
    bfs()
