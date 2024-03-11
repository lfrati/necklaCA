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
    plt.savefig(f"figures/rule_{rule_num}_grid_progression.pdf")
    plt.show()


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

#%%

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

#%%

from collections import defaultdict


def columnify(items, ncols):
    """
    Create a ncols matrix from a list of strings
    E.g.
        columnify([a,b,c,d,e,f,g,h], ncols=3):

        returns

        a,b,c,
        d,e,f,
        g,h
    """
    items = sorted(items, key=int)
    nchunks = len(items) // ncols
    reminder = len(items) % ncols
    chunks = [items[ncols * i : ncols * (i + 1)] for i in range(nchunks)]
    if reminder > 0:
        chunks.append(items[-reminder:])
    rows = [", ".join(chunk) for chunk in chunks]
    return ",\n".join(rows)


def grouper(d, threshold):
    """
    Assume d is a dictionary of (float|int, list)
    Return a new dictionary that groups together (concat keys) all entries
    that have absolute distance less than threshold
    """
    grouped = {}
    for y, nums in d.items():
        if len(grouped) == 0:
            grouped[y] = nums
        else:
            existing = np.array(list(grouped.keys()))
            dist = np.abs(existing - y)
            if np.min(dist) < threshold:
                values = existing[np.argmin(dist)]
                new_group = np.max([values, y])
                new_values = grouped[values] + nums
                del grouped[values]
                grouped[new_group] = new_values
            else:
                grouped[y] = nums
    return grouped


#######################################
############## GROWERS ################
#######################################

growers = set()
print("Growers:\n RULE_NUM, max(STD)")
for rule_num, stds in indeg_scaling:
    M = np.mean(stds)
    if M >= 3:
        growers.add(rule_num)
        print(rule_num, M)
print(growers)
lbls = defaultdict(list)
for rule_num, stds in indeg_scaling:
    if rule_num in growers:
        print(rule_num)
        plt.plot(np.arange(7, 16), stds, ".-")
        final = stds[-1]
        lbls[final].append(str(rule_num))
for i, (y, nums) in enumerate(grouper(lbls, threshold=3).items()):
    plt.text(
        15 + 0.2,
        y,
        columnify(nums, ncols=5),
        fontsize=8,
        va="top",
        bbox=dict(facecolor="white", alpha=0.8),
    )
plt.grid()
plt.xlabel("Sequence length")
plt.ylabel("SD of indegree")
plt.tight_layout()
plt.savefig("figures/growers.pdf")
plt.show()


#######################################
############## SHRINKERS ##############
#######################################

shrinkers = set()
for rule_num, stds in indeg_scaling:
    M = np.mean(stds)
    if M <= 1.0:
        shrinkers.add(rule_num)
print("Shrinkers", shrinkers)


lbls = defaultdict(list)
for rule_num, stds in indeg_scaling:
    if rule_num in shrinkers:
        print(rule_num)
        plt.plot(np.arange(7, 16), stds, ".-")
        final = stds[-1]
        lbls[final].append(str(rule_num))
for i, (y, nums) in enumerate(lbls.items()):
    plt.text(
        15 + 0.3,
        y,
        columnify(nums, ncols=3),
        fontsize=8,
        va="top",
        bbox=dict(facecolor="white", alpha=0.8),
    )
plt.xlabel("Sequence length")
plt.ylabel("SD of indegree")
plt.grid()
plt.tight_layout()
plt.savefig("figures/shrinkers.pdf")
plt.show()

#######################################
############## MIDDLERS ###############
#######################################

lbls = defaultdict(list)
# plt.figure(figsize=(4, 8))
for rule_num, stds in indeg_scaling:
    if rule_num not in shrinkers and rule_num not in growers:
        plt.plot(np.arange(7, 16), stds, ".-")
        final = stds[-1]
        lbls[final].append(str(rule_num))
for i, (y, nums) in enumerate(grouper(lbls, threshold=0.7).items()):
    plt.text(
        15 + 0.4,
        y,
        columnify(nums, ncols=8),
        fontsize=8,
        va="top",
        bbox=dict(facecolor="white", alpha=0.7),
    )
plt.xlabel("Sequence length")
plt.ylabel("SD of indegree")
plt.grid()
plt.tight_layout()
plt.savefig("figures/middlers.pdf")
plt.show()


#%%


def cycles_distribution(N):
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


cycles_12 = cycles_distribution(N=12)
cycles_13 = cycles_distribution(N=13)
cycles_14 = cycles_distribution(N=14)
cycles_15 = cycles_distribution(N=15)

#%%

print(np.where(np.array(cycles_12) >= np.max(cycles_12)))
print(np.where(np.array(cycles_13) >= np.max(cycles_13)))
print(np.where(np.array(cycles_14) >= np.max(cycles_14)))
print(np.where(np.array(cycles_15) >= np.max(cycles_15)))

# plt.plot(sorted(cycles_12), label="N=12")
# plt.plot(sorted(cycles_13), label="N=13")
# plt.plot(sorted(cycles_14), label="N=14")
plt.plot(cycles_15, ".", label="N=15")

lbls = defaultdict(list)
for rule, y in enumerate(cycles_15):
    if y > 3:
        lbls[y].append(str(rule))
for i, (y, nums) in enumerate(grouper(lbls, threshold=2).items()):
    plt.text(
        260,
        y,
        columnify(nums, ncols=5),
        fontsize=8,
        va="top",
        bbox=dict(facecolor="white", alpha=0.8),
    )
# plt.legend()
plt.grid()
plt.xlabel("Rule")
plt.ylabel("Unique simple cycles, N=15")
plt.tight_layout()
plt.show()
