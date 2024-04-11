from collections import defaultdict
from functools import partial
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import numpy as np
from tqdm import tqdm, trange

from ca import RULES, step_implicit
from data import make_group

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


def grid_progression(rule_num, start=7, largest=False):
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
        if largest:
            G = G.subgraph(max(nx.weakly_connected_components(G), key=len)).copy()
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
            "node_size": 0.5,
            "width": 1,
            "alpha": 0.7,
        }
        nx.draw_networkx(G, pos=pos, with_labels=False, ax=ax, **options)
        ax.text(
            0.95,
            0.95,
            f"N={N}",
            fontsize=14,
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


# grid_progression(45)

grid_progression(110, start=8, largest=True)

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
        cycles.append([len(cycle) for cycle in nx.simple_cycles(G)])
    return cycles


fname = "cycles_distribution.pkl"

if Path(fname).exists():
    cycles = pickle.load(open(fname, "rb"))
else:
    cycles = {N: cycles_distribution(N) for N in range(12, 22)}
    pickle.dump(cycles, open(fname, "wb"))


#%%

uniquify = lambda x: [len(set(cycles)) for cycles in x]


def simple_cycles_distr(cycles, N):
    plt.plot(cycles, ".")
    lbls = defaultdict(list)
    for rule, y in enumerate(cycles):
        if y >= 7:
            lbls[y].append(str(rule))
    for y, nums in grouper(lbls, threshold=1.5).items():
        plt.text(
            235,
            y,
            columnify(nums, ncols=4),
            fontsize=8,
            va="top",
            bbox=dict(facecolor="white", alpha=0.8),
        )
    plt.grid(linestyle="--")
    plt.xlabel("Rule")
    plt.ylabel(f"Unique simple cycles, N={N}")
    plt.tight_layout()
    plt.savefig(f"figures/simple_cycles_N{N}.pdf")
    plt.show()


N = 21
simple_cycles_distr(uniquify(cycles[N]), N)

#%%

print("N   M  top_dogs")
print("------------------------------")
for N in sorted(cycles.keys()):
    unique_cycles = uniquify(cycles[N])
    M = np.max(unique_cycles)
    top_dogs = [i for i, c in enumerate(unique_cycles) if c == M]
    print(f"{N} {M:2d}  {top_dogs}")

#%%

unique_cycles = [uniquify(cycles[N]) for N in sorted(cycles.keys())]

all_ranks = []
for row in unique_cycles:
    cycles2ranks = {
        cycles: rank + 1
        for rank, cycles in enumerate(sorted(list(set(row)), reverse=True))
    }
    all_ranks.append([cycles2ranks[cycles] for cycles in row])

ranks = np.array(all_ranks).sum(axis=0)

rules_by_cycles = np.argsort(ranks)
print(rules_by_cycles)
print(ranks[rules_by_cycles] / len(unique_cycles))

# (ranks[rules_by_cycles] / len(unique_cycles))[:6]

from matplotlib.ticker import FuncFormatter

# Function to convert number to ordinal string with superscript
def to_ordinal(n):
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"${n}^{{\\mathrm{{{suffix}}}}}$"  # LaTeX formatting for superscript


# Formatter function for the y-axis
def ordinal_formatter(x, _):
    return to_ordinal(int(x))


fig, ax = plt.subplots(figsize=(6, 4))
values = ranks[rules_by_cycles] / len(unique_cycles)
ax.plot(values, ".-", lw=1, alpha=0.4)
ax.grid()
ax.set_xlabel("Rules (sorted by Avg. Rank)")
ax.set_ylabel("Avg. Rank")
txt = ", ".join([str(rule) for rule in sorted(rules_by_cycles[:6])])
plt.annotate(
    txt,
    xy=(8.5, 1.45),
    xytext=(50, 2),
    arrowprops=dict(facecolor="black", alpha=0.8, width=0.1, headwidth=6, headlength=6),
    bbox=dict(facecolor="white", alpha=0.8),
    ha="left",
)
plt.gca().yaxis.set_major_formatter(FuncFormatter(ordinal_formatter))
plt.gca().set_xticklabels([])
plt.tight_layout()
plt.savefig("figures/cycle_ranks.pdf")
plt.show()

#%%

maxify = lambda x: [max(s) for s in x]
Ns = sorted(cycles.keys())
max_cycles = np.array([max(maxify(cycles[N])) for N in Ns])
max_lens = np.array([len(np.unique(make_group(N))) for N in Ns])

#%%

from matplotlib.ticker import FuncFormatter

# Formatter function for the y-axis
def percentage_formatter(x, _):
    return f"{x:.0f}%"


rows = ["N      Rules w/ max loop"]
for N, m in zip(Ns, max_cycles):
    top_dogs = [rule for rule, mcycles in enumerate(maxify(cycles[N])) if mcycles == m]
    formatted = ", ".join([f"{r:>3d}" for r in top_dogs])
    row = f"{N} | {formatted}"
    print(row)
    rows.append(row)

plt.plot(Ns, (max_cycles / max_lens) * 100, ".-")
plt.text(
    0.6,
    0.95,
    "\n".join(rows),
    fontsize=9,
    horizontalalignment="left",
    verticalalignment="top",
    color="black",
    weight="normal",
    fontname="monospace",
    bbox=dict(facecolor="white", alpha=0.8),
    transform=plt.gca().transAxes,
)
plt.grid()
plt.gca().yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
plt.ylim(0, 100)
plt.xlabel("N")
plt.ylabel("Nodes in the longest loop")
plt.tight_layout()
plt.savefig("figures/longest_loops.pdf")
plt.show()


#%%


import matplotlib.patheffects as patheffects
from mpl_toolkits.axes_grid1 import make_axes_locatable


def LWC_coverage(N):
    group = make_group(N)
    necklaces = np.unique(group)
    ratios = []
    for i in range(256):
        rule = RULES[i]
        edges = make_edges_tuples(necklaces, N, rule, group)
        # no digraph because arrowheads are obnoxious
        # G = nx.Graph()
        G = nx.DiGraph()
        G.add_edges_from(edges)
        # show largest connected component only
        # G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        # G = G.subgraph(max(nx.strongly_connected_components(G), key=len)).copy()
        G = G.subgraph(max(nx.weakly_connected_components(G), key=len)).copy()
        ratio = len(G) / len(necklaces)
        print(f"{i:>3d} {ratio*100:.2f}%")
        assert ratio <= 1.0
        ratios.append(ratio)
    return ratios


N = 21
ratios = LWC_coverage(N=N)

fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(np.array(ratios).reshape(16, 16), vmin=0.0, vmax=1.0)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=12)
ax.axis("off")
for i in range(16):
    for j in range(16):
        ax.text(
            j,
            i,
            f"{i*16+j}",
            ha="center",
            va="center",
            color="white",
            fontsize=8,
            path_effects=[patheffects.withStroke(linewidth=1.0, foreground="black")],
        )
plt.tight_layout()
plt.savefig(f"figures/LWC_coverage_N{N}.pdf")
plt.show()


#%%

import seaborn as sns

sns.histplot(ratios, kde=False, stat="proportion", zorder=3)
plt.grid(zorder=0)
plt.xlabel("Fraction of nodes in the LWC component")
plt.tight_layout()
plt.savefig(f"figures/LWC_distribution_N{N}.pdf")
plt.show()

#%%


def get_num_components(N):
    group = make_group(N)
    necklaces = np.unique(group)
    components = []
    for i in trange(256):
        rule = RULES[i]
        edges = make_edges_tuples(necklaces, N, rule, group)
        G = nx.DiGraph()
        G.add_edges_from(edges)
        components.append(len(list(nx.weakly_connected_components(G))))
    return components


N = 21
num_components = get_num_components(N)

num_cycles = [max(c) for c in cycles[N]]
# num_cycles = [len(set(c)) for c in cycles[N]]

#%%

import seaborn as sns

trianglex = [1, 93, np.max(num_components)]
triangley = [1, 39_000, 1]

top_dogs = []
sns.set_theme()
sns.kdeplot(x=num_components, y=num_cycles, log_scale=True, fill=True)
plt.loglog(num_components, num_cycles, ".")
for i, (rule,) in enumerate(np.argwhere(num_cycles == np.max(num_cycles))):
    top_dogs.append(str(rule))
    plt.loglog(num_components[rule], num_cycles[rule], color="red", marker=".")
plt.text(
    150,
    39_000,
    ", ".join(top_dogs),
    fontsize=10,
    va="bottom",
    bbox=dict(facecolor="white", alpha=0.8),
)
plt.fill(trianglex, triangley, alpha=0.1, color="red")
plt.xlabel("# Components")
plt.ylabel("Max Cycle Length")
plt.tight_layout()
plt.savefig("figures/maxlen_components_N21.pdf")
plt.show()
sns.reset_defaults()
