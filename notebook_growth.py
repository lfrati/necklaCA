from functools import partial
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import numpy as np
from tqdm import trange, tqdm

from ca import RULES, multi_step, step_implicit, to_int
from data import archive, make_group, arr2dict, int2bin

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


#%%


def grid_progression(rule_num, start=7):
    end = start + 9
    rule = RULES[rule_num]
    graphs = []
    xs = np.arange(start, end)
    for N in tqdm(xs):
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
        graphs.append(G)

    _, axs = plt.subplots(ncols=3, nrows=3, figsize=(10, 10))
    axes = axs.flatten()
    for i, N in tqdm(enumerate(xs)):
        ax = axes[i]
        G = nx.Graph(graphs[i])
        # pos = layouts[layout](G)
        # pos = graphviz_layout(G, prog="sfdp", args='-Gbeautify=true')
        pos = graphviz_layout(G, prog="sfdp")

        options = {
            "node_color": "gray",
            "node_size": 1,
            "width": 1,
            "alpha": 0.7,
        }
        nx.draw_networkx(G, pos=pos, with_labels=False, ax=ax, **options)
        ax.text(
            0.95,
            0.95,
            f"{N}",
            fontsize=10,
            horizontalalignment="center",
            verticalalignment="center",
            color="black",
            weight="normal",
            transform=ax.transAxes,
        )
        ax.axis("off")

    plt.tight_layout()
    archive(f"figures/rule_{rule_num}_grid_progression.pdf")


grid_progression(45)
grid_progression(110)

#%%

"""
How representative is the LWC component?
- std of indegree
- what percentage of nodes are in it -> meh
- cycle basis Note: there cannot be multiple cycles in a single LWC since out_degree = 1
"""


def indegree_progression(rule_num, start=7):
    end = start + 9
    rule = RULES[rule_num]
    graphs = []
    xs = np.arange(start, end)
    # for N in tqdm(xs):
    for N in xs:
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
        graphs.append(G)
    return graphs


graphs = indegree_progression(45)
for graph in graphs:
    std = np.std([graph.in_degree(node) for node in graph.nodes])
    cycles = set([len(cycle) for cycle in nx.simple_cycles(graph)])
    print(f"{std} {cycles}")


graphs = indegree_progression(45)
for graph in graphs:
    std = np.std([graph.in_degree(node) for node in graph.nodes])
    cycles = set([len(cycle) for cycle in nx.simple_cycles(graph)])
    print(f"{std} {cycles}")

# Do rules's indegree variability stabilizes?
indeg_scaling = []
for rule_num in trange(0, 256):
    graphs = indegree_progression(rule_num)
    stds = [np.std([graph.in_degree(node) for node in graph.nodes]) for graph in graphs]
    indeg_scaling.append((rule_num, stds))

for rule_num, stds in indeg_scaling[1:-1]:
    if np.max(stds) < 6:
        plt.plot(np.arange(7, 16), stds, ".-")
plt.grid()
plt.show()

print("Growers:\n RULE_NUM, max(STD)")
for rule_num, stds in indeg_scaling:
    M = np.max(stds)
    if M >= 6:
        print(rule_num, M)

print("Shrinkers:\n RULE_NUM, max(STD)")
for rule_num, stds in indeg_scaling:
    M = np.max(stds)
    if M <= 1:
        print(rule_num, M)


#%%


def indegree_distribution(N):
    cycles = []
    for rule_num in trange(256):
        rule = RULES[rule_num]
        group = make_group(N)
        necklaces = np.unique(group)
        edges = make_edges_tuples(necklaces, N, rule, group)
        G = nx.DiGraph()
        G.add_edges_from(edges)
        cycles.append(len(set([len(cycle) for cycle in nx.simple_cycles(G)])))
    return cycles


cycles_12 = indegree_distribution(N=12)
cycles_13 = indegree_distribution(N=13)

#%%

print(np.where(np.array(cycles_12) >= np.max(cycles_12)))
print(np.where(np.array(cycles_13) >= np.max(cycles_13)))

plt.plot(sorted(cycles_12), label="N=12")
plt.plot(sorted(cycles_13), label="N=13")
plt.legend()
plt.show()
