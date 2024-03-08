from functools import partial
from math import cos, pi, sin

from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Rectangle
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import numpy as np
from tqdm import trange, tqdm

from ca import RULES, multi_step, step_implicit, to_int
from data import archive, make_group, arr2dict, int2bin

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
    "sfdp": lambda G: graphviz_layout(G, prog="sfdp"),
}


def make_edges_tuples(necklaces, N, rule, group):
    f = partial(step_implicit, N=N, rule=rule)
    return [(n, group[f(n)]) for n in necklaces]


def plot_grid(N, layout="kamada_kawai", numbers=True):
    group = make_group(N)
    necklaces = np.unique(group)

    _, axs = plt.subplots(nrows=16, ncols=16, figsize=(20, 20))
    for i, ax in tqdm(enumerate(axs.flatten()), total=256):

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
            "node_color": "gray",
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
    node_color = ["red" if node in nodes else "gray" for node in G.nodes]
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
    axs[1].set_title(f"{seq}")

    plt.tight_layout()
    name = f"figures/{rule_num=}-{layout}-{N}.pdf"
    print(name)
    archive(name)


np.random.seed(1338)

N = 11
# layout = "kamada_kawai"
layout = "sfdp"
for rule_num in [30, 90, 110]:
    rule = RULES[rule_num]
    show_sequence_and_graph(rule, rule_num, layout)

#%%

import matplotlib.patches as patches


def show_sequence_and_graph_loop(rule, rule_num, layout):

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

    print(history.shape)

    axs[0].pcolormesh(np.flipud(history), cmap="binary", edgecolors="gray")
    axs[0].set_title(f"Rule={rule_num} N={N}")
    axs[0].axis("off")

    from collections import Counter

    duplicates = [key for key, val in Counter(seq).items() if val > 1]
    if len(duplicates) > 0:
        duplicate = duplicates[0]
        (positions,) = np.where(seq == duplicate)
        for x in positions:
            border = 0.01
            # Create a Rectangle patch
            rect = patches.Rectangle(
                (border, N - x + border),
                N - border * 2,
                1,
                linewidth=2,
                edgecolor="r",
                facecolor=(1.0, 0.0, 0.0, 0.1),
            )
            # Add the patch to the Axes
            axs[0].add_patch(rect)

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
    axs[1].set_title(f"{seq}")

    plt.tight_layout()
    name = f"figures/{rule_num=}-{layout}-{N}_loop.pdf"
    print(name)
    archive(name)


np.random.seed(6)
N = 11
layout = "sfdp"
# rule_num = np.random.randint(256)
rule_num = 111
rule = RULES[rule_num]
show_sequence_and_graph_loop(rule, rule_num, layout)

#%%


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


def draw_round_necklace(N, rule, ax):
    group = make_group(N)
    # necklaces = np.unique(group)
    D = arr2dict(group)

    r = 10
    M = len(D)
    state2coords = {}
    circle_coords = []
    for i, states in enumerate(D.values()):
        progress = i / M
        x, y = (r * sin(progress * pi * 2), r * cos(progress * pi * 2))
        circle_coords.append((x, y))
        for state in states:
            state2coords[state] = (x, y)

    f = partial(step_implicit, N=N, rule=rule)
    edges = [(v, f(v)) for v in range(len(group))]

    for edge in edges:
        _from, _to = edge
        line(
            *state2coords[_from],
            *state2coords[_to],
            ax=ax,
            color="red",
            lw=0.5,
        )

    for (x, y) in circle_coords:
        ax.add_patch(
            Circle(
                (x, y),
                1,
                facecolor="white",
                edgecolor="black",
                lw=0.5,
                zorder=1000,
            )
        )

    ax.set_xlim(-1.15 * r, 1.15 * r)
    ax.set_ylim(-1.15 * r, 1.15 * r)
    ax.set_axis_off()

    # ## show the box but no ticks
    # for tick in ax.xaxis.get_major_ticks():
    #     tick.tick1line.set_visible(False)
    #     tick.tick2line.set_visible(False)
    #     tick.label1.set_visible(False)
    #     tick.label2.set_visible(False)
    # for tick in ax.yaxis.get_major_ticks():
    #     tick.tick1line.set_visible(False)
    #     tick.tick2line.set_visible(False)
    #     tick.label1.set_visible(False)
    #     tick.label2.set_visible(False)


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
        line(
            *state2coords[_from],
            *state2coords[_to],
            ax=ax,
            color="red",
            alpha=0.5,
            # lw=0.5,
        )

    ax.set_xlim(-1.2 * r, 1.3 * r)
    ax.set_ylim(-1.3 * r, 1.2 * r)
    ax.set_axis_off()

    axins = ax.inset_axes(
        [0.85, 0.85, 0.15, 0.15],
        # xlim=(0, subset),
        # ylim=(0, subset),
        # xticklabels=[],
        # yticklabels=[],
    )

    draw_round_necklace(N, rule, axins)

    # plt.title(f"Rule: {rule_num}")
    plt.tight_layout()
    name = f"figures/{rule_num=}-rule_to_necklace.pdf"
    archive(name)
    # plt.savefig(name)
    # plt.show()


N = 6
for rule_num in [0, 2, 90, 110]:
    rule = RULES[rule_num]
    show_rule_and_necklace(rule, rule_num, N)


#%%


graphs = []
xs = np.arange(7, 17)
for N in tqdm(xs):
    group = make_group(N)
    necklaces = np.unique(group)
    edges = make_edges_tuples(necklaces, N, RULES[90], group)
    # no digraph because arrowheads are obnoxious
    # G = nx.Graph()
    G = nx.DiGraph()
    G.add_edges_from(edges)
    # show largest connected component only
    # G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    # G = G.subgraph(max(nx.strongly_connected_components(G), key=len)).copy()
    G = G.subgraph(max(nx.weakly_connected_components(G), key=len)).copy()
    graphs.append(G)

#%%


import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec


def format_axes(fig):
    for _, ax in enumerate(fig.axes):
        # ax.text(0.5, 0.5, "ax%d" % (i + 1), va="center", ha="center")
        ax.tick_params(labelbottom=False, labelleft=False)
        ax.set_aspect("equal")


fig = plt.figure(figsize=(10, 10), layout="constrained")

gs = GridSpec(5, 5, figure=fig)
# identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))
axs = []
for i in range(5):
    axs.append(fig.add_subplot(gs[0, i]))
for i in range(1, 5):
    axs.append(fig.add_subplot(gs[i, 0]))
axs.append(fig.add_subplot(gs[1:, 1:]))


for i, N in tqdm(enumerate(xs)):
    ax = fig.axes[i]
    G = graphs[i]
    # pos = layouts[layout](G)
    # pos = graphviz_layout(G, prog="sfdp", args='-Gbeautify=true')
    pos = graphviz_layout(G, prog="sfdp")

    options = {
        "node_color": "gray",
        "node_size": 5,
        "width": 1,
        "alpha": 0.7,
        "arrowsize": 4,
    }
    nx.draw_networkx(G, pos=pos, with_labels=False, ax=ax, **options)
    ax.text(
        0.95,
        0.95,
        f"{N}",
        fontsize=10,
        horizontalalignment="center",
        verticalalignment="center",
        color="red" if N in [8, 16] else "black",
        weight="bold" if N in [8, 16] else "normal",
        transform=ax.transAxes,
    )
    ax.axis("off")

# archive(f"figures/rule_90_progression.pdf")
format_axes(fig)
plt.tight_layout()
# archive(f"figures/rule_90_progression.pdf")
plt.savefig(f"figures/rule_90_progression.png")
# plt.show()

#%%

from numpy.lib.stride_tricks import sliding_window_view
from ca import RULES, BASE


def step(state, rule, mode="wrap"):
    s = np.pad(state, (1, 1), mode)
    s = sliding_window_view(s, 3)
    s = (s * BASE).sum(axis=1)
    return rule[s]


def multi_step(state, rule, L, mode="wrap"):
    history = np.empty((L + 1, state.shape[0]), dtype=np.uint8)
    history[0] = state

    for i in range(1, L + 1):
        state = step(state, rule, mode)
        history[i] = state

    return history


L = 64
rule_num = 30
fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
state = np.random.randint(2, size=L)
hist = multi_step(state, RULES[rule_num], L, "constant")
axs[0].imshow(hist, interpolation="nearest", cmap="gray")
axs[0].set_title("constant")
axs[0].set_axis_off()
hist = multi_step(state, RULES[rule_num], L, "wrap")
axs[1].imshow(hist, interpolation="nearest", cmap="gray")
axs[1].set_title("wrap")
axs[1].set_axis_off()
plt.tight_layout()
archive("figures/constant_vs_wrap.png")
# plt.savefig("figures/constant_vs_wrap.png")
# plt.show()
