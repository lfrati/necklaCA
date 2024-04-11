from functools import partial

import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import numpy as np
from tqdm import tqdm

from ca import step_implicit
from ca import RULES, step_implicit
from data import make_group

plt.rcParams.update({"font.size": 16})

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


def necklaces_edges(N, rule):
    group = make_group(N)
    necklaces = np.unique(group)
    f = partial(step_implicit, N=N, rule=RULES[rule])
    return [(n, group[f(n)]) for n in necklaces]


def all_edges(N, rule):
    f = partial(step_implicit, N=N, rule=RULES[rule])
    return [(n, f(n)) for n in range(2**N)]


def compare(N, rule, layout="sfdp"):

    _, axs = plt.subplots(nrows=2, ncols=1, figsize=(4, 8))

    ax = axs[0]

    for ax, edges, title in zip(
        axs,
        [all_edges(N, rule), necklaces_edges(N, rule)],
        ["all nodes", "necklaces only"],
    ):

        # no digraph because arrowheads are obnoxious
        G = nx.Graph()
        # G = nx.DiGraph()
        G.add_edges_from(edges)

        # show largest connected component only
        # G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        # G = G.subgraph(max(nx.strongly_connected_components(G), key=len)).copy()
        # G = G.subgraph(max(nx.weakly_connected_components(G), key=len)).copy()

        pos = layouts[layout](G)

        options = {
            "node_color": "gray",
            "node_size": 0,
            "width": 1,
            "alpha": 0.7,
            "arrowsize": 6,
        }
        nx.draw_networkx(G, pos=pos, with_labels=False, ax=ax, **options)
        # ax.axis("off")
        ax.text(
            0.95,
            0.95,
            title,
            transform=ax.transAxes,
            horizontalalignment="right",
            verticalalignment="center",
            fontsize=12,
            color="black",
        )
        # ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(f"figures/all_vs_necks_rule{rule}_N{N}.pdf")
    # plt.show()
    plt.close()


#%%

N = 15
for rule in tqdm([179, 238]):
    # rule = np.random.randint(256)
    compare(N, rule)
