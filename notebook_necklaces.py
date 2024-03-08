from time import monotonic

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from data import make_group, arr2dict, make_group, archive


plt.rcParams.update({"font.size": 16})

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

#%%


# def show(necklace, N):
#     D = {}
#     for i, v in enumerate(necklace):
#         if v not in D:
#             D[v] = []
#         D[v].append(int2bin(i, N))
#
#     for k, vals in D.items():
#         print(k)
#         for v in vals:
#             print(f"  {v}")
#         print()

"""
What does the distribution of the necklaces look like for each N?
Since the space is growing exponentially (N -> 2**N) I plot things in loglog 
space to visually compare (otherwise larger N overshadows everything).
"""

plt.figure(figsize=(10, 8))

for N in range(12, 22):
    st = monotonic()
    necklaces = make_group(N)
    et = monotonic()
    print(f"Elapsed: {et - st:.6f}")

    data = sorted(set(necklaces))
    plt.loglog(data, label=f"{N=}")
    # plt.plot(data, ".",label=f"{N=}")
    # plt.plot(data,label=f"{N=}")

plt.legend()
plt.grid()
plt.xlabel("Necklace rank (for each N : [0, #necklaces])")
plt.ylabel("Necklace value (e.g. [0011, ..., 1100]  = 3)")
plt.tight_layout()
archive("figures/distribution_of_necklaces.pdf")

"""
What a nice regular shape, no wonder people propose recursive algorithms
"""

#%%

"""
Let's look at the actual necklaces
"""

# for N in [8, 10, 12, 14, 16]:
#     necklaces = make_group(N)
#     plt.figure(figsize=(10, 10))
#     plt.plot(necklaces, ".")
#     plt.title(f"{N=}")
#     plt.tight_layout()
#     plt.show()

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

N = 18
subset = 50_000
npoints = 2**N
xs = np.arange(npoints)
necklaces = make_group(N)
extent = 0, npoints, 0, necklaces.max()
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(xs, necklaces, ",")

axins = ax.inset_axes(
    [0.05, 0.55, 0.4, 0.4],
    xlim=(0, subset),
    ylim=(0, subset),
    xticklabels=[],
    yticklabels=[],
)

# ax_inset = inset_axes(ax, width="30%", height="30%", loc="upper center")
axins.plot(xs[:subset], necklaces[:subset], ",", label="Inset Plot")
rect, connectors = ax.indicate_inset_zoom(axins, edgecolor="black")

# print(dir(connectors[0]))
connectors[0].set_visible(False)
connectors[1].set_visible(False)
connectors[2].set_visible(False)
connectors[3].set_visible(False)

point1 = [subset / 2, subset]
point2 = [subset, subset * 2.7]
# Draw an arrow between the two points
ax.arrow(
    point1[0],
    point1[1],
    point2[0] - point1[0],
    point2[1] - point1[1],
    head_width=5_000.0,
    head_length=5_000,
    fc="gray",
    ec="gray",
)

ax.set_xlabel("x")
ax.set_ylabel("necklace(x)")
plt.tight_layout()
archive("figures/necklaces_foreach_x.png", dpi=300)


#%%


def line(a, b, N, ax, **kwargs):
    xs = np.linspace(0, 1, N)
    ys = a * xs + b
    pos = np.where((ys >= 0) & (ys <= 1))
    ax.plot(xs[pos], ys[pos], **kwargs)


N = 18
npoints = 2**N
necklaces = make_group(N)
fig, axs = plt.subplots(ncols=2, figsize=(16, 8.5))
axs[0].plot(np.arange(npoints), necklaces, ",")
axs[1].plot(np.linspace(0, 1, npoints), necklaces / 2**N, ",")


for i in range(0, 6):
    if i == 0:
        line(
            a=1 / 2**i,
            b=0,
            N=npoints,
            ax=axs[1],
            color="red",
            lw=0.5,
            label="$f(x) = \\dfrac{x}{2^i},\\; i=[0,1,...]$",
        )
    else:
        line(a=1 / 2**i, b=0, N=npoints, ax=axs[1], color="red", lw=0.5)


def get_b_from_ax(a, x):
    return -a * x


for i in range(1, 6):
    a = 2**i
    b = -(2**i) + 1
    if i == 1:
        line(
            a=a,
            b=b,
            N=npoints,
            ax=axs[1],
            color="green",
            lw=0.5,
            label="$f(x) = 2^i(x-1)+1$",
        )
    else:
        line(a=a, b=b, N=npoints, ax=axs[1], color="green", lw=0.5)

# x = 1 / 2
# a = x * 2**2
# line(a=a, b=get_b_from_ax(a, x), N=npoints, ax=axs[1], color="red", lw=1)
#
# x = 1 / 4
# a = x * 2**4
# line(a=a, b=get_b_from_ax(a, x), N=npoints, ax=axs[1], color="red", lw=1)
#
# x = 1 / 8
# a = x * 2**6
# line(a=a, b=get_b_from_ax(a, x), N=npoints, ax=axs[1], color="red", lw=1)
#
# x = 1 / 16
# a = x * 2**8
# line(a=a, b=get_b_from_ax(a, x), N=npoints, ax=axs[1], color="red", lw=1)
#
# x = 1 / 32
# a = x * 2**10
# line(a=a, b=get_b_from_ax(a, x), N=npoints, ax=axs[1], color="red", lw=1)


axs[0].set_aspect("equal")
axs[1].set_aspect("equal")
axs[0].set_xlabel("x")
axs[0].set_ylabel("necklace(x)")

plt.grid()
plt.legend()
plt.tight_layout()
archive("figures/lines_of_necklaces.png", dpi=300)


#%%

"""
These necklaces distributions for growing values of N feels like they
have a "self-similar" structure.
"""

plt.figure(figsize=(10, 10))
for N in range(14, 8, -1):
    necklaces = make_group(N)
    plt.plot(necklaces, ".", label=f"N={N}")
plt.tight_layout()
plt.legend()
plt.xlabel("x")
plt.xlabel("Necklace(x)")
archive("figures/consecutive_necklaces.pdf")

"""
But do them? If I plot multiple runs in the same range (o.w. it grows exponentially)
it doesn't look points overlap.
Definitely have the same structure, but I don't think later versions contain
"exactly" previous iterations 
"""

plt.figure(figsize=(10, 10))
for N in range(13, 10, -1):
    necklaces = make_group(N)
    xs = (np.linspace(0, 1, 2**N) + 1) / 2
    ys = necklaces / 2**N  # normalize to 0-1
    plt.plot(xs, ys, ".", label=f"{N=}")
plt.legend()
plt.tight_layout()
# plt.grid()
plt.xlabel("x")
plt.xlabel("Necklace(x)")
archive("figures/consecutive_necklaces_same_range.pdf")

# plt.figure(figsize=(10, 10))
# for N in range(20, 10, -1):
#     necklaces = make_group(N)
#     xs = (np.linspace(0, 1, 2**N) + 1) / 2
#     ys = necklaces / 2**N  # normalize to 0-1
#     plt.plot(xs, ys, ",")
# plt.tight_layout()
# # plt.grid()
# plt.show()

#%%

"""
I can quantify how much neckalces N are contained in necklaces N + 1 by checking
among the first 2**N values of necklaces N+1 how many match exactly
"""

print("    N |    Size | Tot. overlap")
print("---------------------------------------------")
for N in range(4, 20):

    neck_N = make_group(N)
    neck_Nplus = make_group(N + 1)

    # range of values that N and N + 1 share
    assert neck_N.size == 2**N, f"{neck_N.size} != {2**N}"
    assert neck_Nplus.size == 2 ** (N + 1), f"{neck_Nplus.size} != {2**N}"
    shared_range = neck_Nplus[: neck_N.size]

    matching = neck_N == shared_range
    print(f"{N:>5} | {neck_N.size:>7} | {matching.sum()/neck_N.size * 100:<.2f}%")


#%%

"""
Necklaces of len N and N+1 overlap on the first  2 ** int(np.floor(N / 2) + 1)) elements

This is influenced by the number of zeros left and right to the left-most 1
"""


def int_to_binary(n, digits):
    # Convert integer to binary string, removing the '0b' prefix
    binary_str = bin(n)[2:]

    # Pad with leading zeros if necessary to achieve the desired length
    if len(binary_str) < digits:
        binary_str = binary_str.rjust(digits, "0")
    elif len(binary_str) > digits:
        raise ValueError("The number exceeds the length provided by digits")

    return binary_str


N = 20
for N in range(3, 16):
    neck_N = make_group(N)
    neck_Nplus = make_group(N + 1)
    shared_range = neck_Nplus[: neck_N.size]
    # matching = neck_N == shared_range
    # print(np.all(neck_N <= shared_range))
    # different = neck_N != shared_range
    different = np.where(neck_N != shared_range)
    divergence = np.min(different)
    m = 0
    print(f"N:{N}, divergence point={divergence}", 2 ** int(np.floor(N / 2) + 1))
    print(
        f"{int_to_binary(divergence-m, N):>20}",
        f"{int_to_binary(neck_N[divergence-m], N):>20}",
        neck_N[divergence - m],
    )
    print(
        f"{int_to_binary(divergence-m, N+1):>20}",
        f"{int_to_binary(neck_Nplus[divergence-m], N+1):>20}",
        neck_Nplus[divergence - m],
    )


#%%


x = np.arange(4, 30)
y = np.array(
    [
        81.25,
        65.62,
        64.06,
        52.34,
        50.00,
        44.53,
        41.80,
        38.04,
        36.40,
        33.73,
        32.17,
        30.45,
        29.10,
        27.73,
        26.64,
        25.55,
        24.61,
        23.72,
        22.91,
        22.15,
        21.46,
        20.81,
        20.21,
        19.64,
        19.10,
        18.60,
    ]
)


# Plotting
plt.plot(x, y / 100, ".-", label="Overlap between necklaces of length N and N+1")
plt.xlabel("N")
plt.ylabel("Overlap")
plt.legend()
plt.grid()
plt.tight_layout()
archive("figures/supplementary_scaling_overlap.pdf")
plt.show()

#%%

N = 10
neck_N = make_group(N).astype(np.int64)
neck_Nplus = make_group(N + 1).astype(np.int64)
shared_range = neck_Nplus[: neck_N.size]
matching = neck_N == shared_range
divergence_point = np.argmin(matching)

plt.figure(figsize=(10, 10))
plt.plot(neck_Nplus, ".", label=f"N={N+1}")
plt.plot(neck_N, ".", label=f"N={N}")
plt.plot(
    np.arange(shared_range.size)[matching],
    shared_range[matching],
    ".",
    label="Overlap",
    color="red",
)
plt.legend()
plt.xlabel("x")
plt.xlabel("Necklace(x)")
plt.tight_layout()
archive("figures/overlapping_necklaces.pdf")

#%%

xs = []
ys = []
for N in trange(4, 33):
    necks = np.load(f"data/necks-{N}.npy", mmap_mode="r")
    nnecks = necks.size
    xs.append(N)
    ys.append(nnecks)

print(ys)

xs = np.array(xs)
ys = np.array(ys)

plt.plot(xs, (ys / 2**xs), ".-", label="$\\dfrac{Necklaces(N)}{2^N}$")
plt.plot(xs, (1 / xs), "--", alpha=0.7, label="1/N", color="red")
plt.xlabel("N")
plt.grid()
plt.legend()
plt.tight_layout()
archive("figures/necklaces_scaling.pdf")

# fig, ax = plt.subplots()
# ax.set_yscale("log")
# ax.plot(xs, ys, ".-", alpha=0.5, label="necklaces")
# ax.plot(xs, 2**xs / xs, "--", label="$2^N/N$")
# plt.xlabel("Length")
# plt.ylabel("% Necklaces")
# plt.grid()
# plt.legend()
# plt.tight_layout()
# plt.show()

#%%


ys = []
xs = np.arange(4, 24)
for N in xs:
    group = make_group(N)
    group = arr2dict(group)
    sizes = np.array([len(val) for val in group.values()])
    ratio = np.sum(sizes >= N) / len(sizes)
    print(N, ratio)
    ys.append(ratio)

# plt.figure(figsize=(8, 4))
plt.plot(xs, ys, ".-", label="fraction of aperiodic\nnecklaces of length N")
plt.grid()
plt.legend(loc="center right")
plt.text(xs[-1] - 0.5, ys[-1] - 0.05, f"{ys[-1]:.5f}", fontsize=12)
plt.xlabel("N")
plt.tight_layout()
archive("figures/aperiodic_necklaces.pdf")
