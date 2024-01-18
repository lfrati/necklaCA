from functools import partial

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from ca import RULES, multi_step, step_implicit, to_int
from data import make_group, archive

# import numba
# @numba.njit("u8[:,:](u4[:], u8, u1[:])", parallel=True)
# def make_edges(necklaces, N, rule):
#     M = necklaces.shape[0]
#     edges = np.empty((M, 2), dtype=np.uint64)
#     edges[:, 0] = necklaces
#     for i in numba.prange(M):
#         edges[i, 1] = step_implicit(edges[i, 0], N, rule)
#     return edges

plt.rcParams.update({"font.size": 16})

np.random.seed(4)

layouts = {
    "circular": nx.circular_layout,
    "kamada_kawai": nx.kamada_kawai_layout,
    "planar": nx.planar_layout,
    "random": nx.random_layout,
    "spectral": nx.spectral_layout,
    "spring": nx.spring_layout,
    "shell": nx.shell_layout,
}

from tqdm import tqdm


def make_edges_tuples(necklaces, N, rule, group):
    f = partial(step_implicit, N=N, rule=rule)
    return [(n, group[f(n)]) for n in necklaces]


def plot_grid(N, layout="kamada_kawai", numbers=True):
    group = make_group(N)
    necklaces = np.unique(group)

    _, axs = plt.subplots(nrows=16, ncols=16, figsize=(20, 20))
    for i, ax in tqdm(enumerate(axs.flatten())):

        edges = make_edges_tuples(necklaces, N, RULES[i], group)

        # no digraph because arrowheads are obnoxious
        G = nx.Graph()
        # G = nx.DiGraph()
        G.add_edges_from(edges)

        # show largest connected component only
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        # G = G.subgraph(max(nx.strongly_connected_components(G), key=len)).copy()
        # G = G.subgraph(max(nx.weakly_connected_components(G), key=len)).copy()

        pos = layouts[layout](G)

        options = {
            "node_color": "black",
            "node_size": 1,
            "width": 1,
            "alpha": 0.7,
            "arrowsize": 6,
        }
        nx.draw_networkx(G, pos=pos, with_labels=False, ax=ax, **options)

        ax.axis("off")
        if numbers:
            ax.text(
                0.95,
                0.95,
                f"{i}",
                fontsize=10,
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )
    plt.tight_layout()
    archive(f"figures/{N}-{layout}-all.pdf")


#%%

# import pickle
# with open("necklaces.pkl", "rb") as f:
#     data = pickle.load(f)
# print(data)
# N = data["N"]
# group = data["group"]

for N in [9, 10, 11]:
    plot_grid(N)

#%%


def show_sequence_and_graph(rule, rule_num, layout):

    group = make_group(N)
    necklaces = np.unique(group)

    edges = make_edges_tuples(necklaces, N, rule, group)

    # no digraph because arrowheads are obnoxious
    # G = nx.Graph()
    G = nx.DiGraph()
    G.add_edges_from(edges)

    # show largest connected component only
    # G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    # G = G.subgraph(max(nx.strongly_connected_components(G), key=len)).copy()
    # G = G.subgraph(max(nx.weakly_connected_components(G), key=len)).copy()

    pos = layouts[layout](G)

    options = {
        "width": 1,
        # "alpha": 0.7,
        "arrowsize": 4,
    }

    _, axs = plt.subplots(ncols=2, figsize=(16, 10))

    state = np.random.randint(2, size=N, dtype=np.uint8)
    history = multi_step(state, rule, N)

    seq = [group[to_int(state)] for state in history]
    nodes = set(seq)
    node_color = ["red" if node in nodes else "black" for node in G.nodes]
    node_size = [40 if node in nodes else 10 for node in G.nodes]

    axs[0].pcolormesh(history, cmap="binary", edgecolors="gray")
    axs[0].set_title(f"Rule={rule_num} N={N}")
    axs[0].axis("off")

    # axs[0].set_xticks(seq)

    nx.draw_networkx(
        G,
        pos=pos,
        with_labels=False,
        ax=axs[1],
        node_color=node_color,
        node_size=node_size,
        **options,
    )
    axs[1].set_title(f"Necklaces:\n{seq}")

    plt.tight_layout()
    name = f"figures/{rule_num=}-{layout}-{N}.pdf"
    print(name)
    archive(name)


N = 11
layout = "kamada_kawai"
for rule_num in [30, 90, 110]:
    rule = RULES[rule_num]
    show_sequence_and_graph(rule, rule_num, layout)

#%%

from matplotlib.patches import Rectangle, Circle
from matplotlib.collections import PatchCollection

from math import cos, sin, pi
from data import arr2dict, int2bin


def state2coll(pos, state):
    x, y = pos
    rects = []
    colors = []
    for i, s in enumerate(state):
        colors.append("black" if s == "0" else "white")
        rects.append(Rectangle((x + i, y), 1, 1))
    return PatchCollection(rects, color=colors, edgecolor="black")


def line(x0, y0, x1, y1, ax=None, **kwargs):
    if ax is None:
        plt.plot([x0, x1], [y0, y1], **kwargs)
    else:
        ax.plot([x0, x1], [y0, y1], **kwargs)


def show_rule_and_necklace(rule, rule_num, N):

    group = make_group(N)
    # necklaces = np.unique(group)
    D = arr2dict(group)

    _, ax = plt.subplots(figsize=(10, 10))
    r = 30
    M = len(D)
    state2coords = {}
    for i, states in enumerate(D.values()):
        progress = i / M
        x, y = (r * sin(progress * pi * 2), r * cos(progress * pi * 2))
        for j, state in enumerate(states):
            s = int2bin(state, N)
            s_x = x
            s_y = y - j * 1.3
            state2coords[state] = (s_x + N / 2, s_y)
            ax.add_collection(state2coll((s_x, s_y), s))

        ax.add_patch(
            Circle(
                (x + N / 2, y - len(states) / 2),
                int(r / 5.2),
                facecolor="none",
                edgecolor="black",
            )
        )

    f = partial(step_implicit, N=N, rule=rule)
    edges = [(v, f(v)) for v in range(len(group))]

    for edge in edges:
        _from, _to = edge
        line(*state2coords[_from], *state2coords[_to], ax=ax, color="red", lw=0.5)

    plt.xlim(-1.2 * r, 1.3 * r)
    plt.ylim(-1.3 * r, 1.2 * r)
    plt.axis("off")
    plt.title(f"Rule: {rule_num}")
    plt.tight_layout()
    name = f"figures/{rule_num=}-rule_to_necklace.pdf"
    archive(name)


N = 6
for rule_num in [0, 2, 90, 110]:
    rule = RULES[rule_num]
    show_rule_and_necklace(rule, rule_num, N)

#%%

from tqdm import trange


graphs = []
for N in trange(16):
    N = N + 1
    group = make_group(N)
    necklaces = np.unique(group)
    edges = make_edges_tuples(necklaces, N, RULES[90], group)
    # no digraph because arrowheads are obnoxious
    G = nx.Graph()
    # G = nx.DiGraph()
    G.add_edges_from(edges)
    # show largest connected component only
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    # G = G.subgraph(max(nx.strongly_connected_components(G), key=len)).copy()
    # G = G.subgraph(max(nx.weakly_connected_components(G), key=len)).copy()
    graphs.append(G)

#%%

# layout = "kamada_kawai"
layout = "planar"
_, axs = plt.subplots(nrows=4, ncols=4, figsize=(10, 10))

for N, ax in tqdm(enumerate(axs.flatten())):
    G = graphs[N]
    pos = layouts[layout](G)

    options = {
        "node_color": "black",
        "node_size": 1,
        "width": 1,
        "alpha": 0.7,
        "arrowsize": 6,
    }
    nx.draw_networkx(G, pos=pos, with_labels=False, ax=ax, **options)
    ax.text(
        0.95,
        0.95,
        f"{N+1}",
        fontsize=10,
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
    )
    ax.axis("off")
plt.tight_layout()
# archive(f"figures/rule_90_progression.pdf")

plt.show()

#%%

opts = ["circular", "planar", "random", "spectral", "spring", "shell"]

for layout in opts:
    _, ax = plt.subplots(figsize=(10, 10))

    N = 15
    G = graphs[N]
    pos = layouts[layout](G)

    options = {
        "node_color": "black",
        "node_size": 1,
        "width": 1,
        "alpha": 0.7,
        "arrowsize": 6,
    }
    nx.draw_networkx(G, pos=pos, with_labels=False, ax=ax, **options)
    ax.text(
        0.95,
        0.95,
        f"{N+1}",
        fontsize=10,
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
    )
    ax.axis("off")
    plt.title(layout)
    plt.tight_layout()
    # archive(f"figures/rule_90_progression.pdf")

    plt.show()
