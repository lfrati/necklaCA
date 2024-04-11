from functools import partial
from pathlib import Path
import pickle

import networkx as nx
import numpy as np
from tqdm import trange

from ca import RULES, step_implicit
from data import make_group
import matplotlib.pyplot as plt
from collections import defaultdict


def make_edges_tuples(necklaces, N, rule, group):
    f = partial(step_implicit, N=N, rule=rule)
    return [(n, group[f(n)]) for n in necklaces]


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


#%%

"""
How representative is the LWC component?
- std of indegree
- what percentage of nodes are in it -> meh
- cycle basis Note: there cannot be multiple cycles in a single LWC since out_degree = 1
"""


def indegree_progression_grid(rule_num, start=7):
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


graphs = indegree_progression_grid(45)
for graph in graphs:
    std = np.std([graph.in_degree(node) for node in graph.nodes])
    cycles = set([len(cycle) for cycle in nx.simple_cycles(graph)])
    print(f"{std} {cycles}")


graphs = indegree_progression_grid(110)
for graph in graphs:
    std = np.std([graph.in_degree(node) for node in graph.nodes])
    cycles = set([len(cycle) for cycle in nx.simple_cycles(graph)])
    print(f"{std} {cycles}")


#%%


def indegree_progression(rule_num, start, end):
    rule = RULES[rule_num]
    graphs = []
    xs = np.arange(start, end)
    for N in xs:
        group = make_group(N)
        necklaces = np.unique(group)
        edges = make_edges_tuples(necklaces, N, rule, group)
        G = nx.DiGraph()
        G.add_edges_from(edges)
        # show largest connected component only
        # G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        # G = G.subgraph(max(nx.strongly_connected_components(G), key=len)).copy()
        # G = G.subgraph(max(nx.weakly_connected_components(G), key=len)).copy()
        graphs.append(G)
    return graphs


fname = "indeg_scaling_std.pkl"

if Path(fname).exists():
    indeg_scaling = pickle.load(open(fname, "rb"))
else:
    # Do rules's indegree variability stabilizes?
    indeg_scaling = []
    for rule_num in trange(0, 256):
        graphs = indegree_progression(rule_num, start=10, end=21)
        stds = [
            np.std([graph.in_degree(node) for node in graph.nodes]) for graph in graphs
        ]
        indeg_scaling.append((rule_num, stds))
    pickle.dump(indeg_scaling, open(fname, "wb"))


#%%

#######################################
############## GROWERS ################
#######################################

growers = set()
print("Growers:\n RULE_NUM, max(STD)")
for rule_num, stds in indeg_scaling:
    if rule_num < 128:
        M = np.mean(stds)
        if M >= 8:
            growers.add(rule_num)
            print(rule_num, M)
print(growers)
lbls = defaultdict(list)
for rule_num, stds in indeg_scaling:
    if rule_num in growers:
        print(rule_num)
        plt.plot(np.arange(10, 10 + len(stds)), stds, ".-")
        final = stds[-1]
        lbls[final].append(str(rule_num))
for i, (y, nums) in enumerate(grouper(lbls, threshold=20).items()):
    plt.text(
        21 + 0.2,
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

#%%

#######################################
############## SHRINKERS ##############
#######################################

shrinkers = set()
for rule_num, stds in indeg_scaling:
    if rule_num < 128:
        M = np.mean(stds)
        if M <= 1.5:
            shrinkers.add(rule_num)
print("Shrinkers", shrinkers)


lbls = defaultdict(list)
for rule_num, stds in indeg_scaling:
    if rule_num in shrinkers:
        print(rule_num)
        plt.plot(np.arange(10, 10 + len(stds)), stds, ".-")
        final = stds[-1]
        lbls[final].append(str(rule_num))
for i, (y, nums) in enumerate(grouper(lbls, threshold=0.2).items()):
    plt.text(
        21 + 0.3,
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

#%%

#######################################
############## MIDDLERS ###############
#######################################

lbls = defaultdict(list)
# plt.figure(figsize=(4, 8))
for rule_num, stds in indeg_scaling:
    if rule_num < 128:
        if rule_num not in shrinkers and rule_num not in growers and np.max(stds) >= 5:
            plt.plot(np.arange(10, 10 + len(stds)), stds, ".-")
            final = stds[-1]
            lbls[final].append(str(rule_num))
for i, (y, nums) in enumerate(
    grouper(
        lbls,
        threshold=1,
    ).items()
):
    plt.text(
        20 + 0.4,
        y,
        columnify(nums, ncols=5),
        fontsize=8,
        va="top",
        bbox=dict(facecolor="white", alpha=0.7),
    )
plt.xlabel("Sequence length")
plt.ylabel("SD of indegree")
plt.grid()
plt.tight_layout()
plt.savefig("figures/middlers_top.pdf")
plt.show()

lbls = defaultdict(list)
# plt.figure(figsize=(4, 8))
for rule_num, stds in indeg_scaling:
    if rule_num < 128:
        if rule_num not in shrinkers and rule_num not in growers and np.max(stds) < 5:
            plt.plot(np.arange(10, 10 + len(stds)), stds, ".-")
            final = stds[-1]
            lbls[final].append(str(rule_num))
for i, (y, nums) in enumerate(grouper(lbls, threshold=0.5).items()):
    plt.text(
        20 + 0.4,
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
plt.savefig("figures/middlers_bot.pdf")
plt.show()


##############################


fname = "indeg_scaling_std.pkl"

if Path(fname).exists():
    indeg_scaling = pickle.load(open(fname, "rb"))
else:
    # Do rules's indegree variability stabilizes?
    indeg_scaling = []
    for rule_num in trange(0, 256):
        graphs = indegree_progression(rule_num, start=10, end=21)
        stds = [
            np.std([graph.in_degree(node) for node in graph.nodes]) for graph in graphs
        ]
        indeg_scaling.append((rule_num, stds))
    pickle.dump(indeg_scaling, open(fname, "wb"))


def mean_abs(xs):
    xs = np.array(xs)
    return (xs - 1.0).sum()


for N in range(11, 12, 2):
    ys = []
    for rule in range(128):
        graphs = indegree_progression(rule, start=N, end=N + 1)
        stds = [[graph.in_degree(node) for node in graph.nodes] for graph in graphs]
        xs = stds[0]
        a = ((np.array(xs) - 1) > 0).sum()
        b = ((np.array(xs) - 1) < 0).sum()
        if b > 0:
            print(rule)
            print(a / b)
            print()
            ys.append(a / b)
        else:
            ys.append(0)

    plt.plot(ys, ".")
    plt.xticks(np.linspace(0, len(ys), 6))
    plt.grid()
    plt.xlabel("Rule")
    # plt.yscale("log")
    # plt.ylabel("Log Attractor Ratio")
    plt.ylabel("Attractor Ratio")
    plt.tight_layout()
    plt.savefig(f"figures/attractor_ratio_N={N}.pdf")
    plt.show()
