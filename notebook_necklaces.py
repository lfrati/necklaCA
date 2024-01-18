from time import monotonic

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from data import make_group, arr2dict, make_group, archive


plt.rcParams.update({"font.size": 16})

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

print("    N |    Size | Head overlap | Tot. overlap")
print("---------------------------------------------")
for N in range(3, 17):

    neck_N = make_group(N)
    neck_Nplus = make_group(N + 1)

    # range of values that N and N + 1 share
    assert neck_N.size == 2**N, f"{neck_N.size} != {2**N}"
    shared_range = neck_Nplus[: neck_N.size]

    matching = neck_N == shared_range

    divergence_point = np.argmin(matching)
    assert all(matching[:divergence_point])

    print(
        f"{N:>5} | {neck_N.size:>7} | {divergence_point/neck_N.size * 100:11.2f}% | {matching.sum()/neck_N.size * 100:<.2f}%"
    )


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
plt.show()

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
