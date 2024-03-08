import inspect
from itertools import product
import os
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numba
import numpy as np


def getenv(key: str, default=0):
    return type(default)(os.getenv(key, default))


FORCE = getenv("FORCE", 0)


def get_head_name():
    head = Path(".git/HEAD").read_text().strip()
    match = re.match("ref:\s+refs/heads/(\w+)", head)
    if match is not None:
        return match.groups()[0]
    raise RuntimeError(f"Couldn't parse .git/HEAD: {head}")


def is_interactive():
    import __main__ as main

    return not hasattr(main, "__file__")


def archive(path, tag=False, force=False, **kwargs):
    """
    Get a tag for the current line with format:
        {short_git_hash}-{__file__}-{lineno}
    """
    if is_interactive():
        print(f"Running in interactive mode. Showing plot instead of saving to: {path}")
        plt.show()
        return

    filename = path

    if tag:
        head = get_head_name()
        commit = Path(f".git/refs/heads/{head}").read_text().strip()[:8]
        caller = inspect.currentframe().f_back
        caller_line = caller.f_lineno  # f_back = the caller, not us
        caller_file = Path(inspect.getfile(caller)).name

        # if there are multiple folders insert the tag right before
        # the filename e.g. ./figures/plot.pdf -> ./figures/TAG_plot.pdf
        path = Path(path)
        parent = path.parent
        name = path.name
        filename = f"{parent}/{commit}:{caller_file}:{caller_line}-{name}"


    if force or FORCE or not Path(filename).exists():
        plt.savefig(filename, **kwargs)
        print(f"Saved {filename}")
    else:
        print(f"{filename} exists. Skipping...")
    plt.close()


"""
How do we represent all the neckaces? In the following we use 2 approaches arrays or dictionaries

We call "necklace" the smallest element in each equivalence class.

In the array version: 
- arr[n] = the decimal representation of the necklace representing the equivalence class that n belongs to
- easy to get the necklace given a number, hard to get all the numbers given a necklace

In the  dict version: 
- dict[i] = the members of the group represented by the necklace i
- dict.keys() = all necklaces, hard to get the necklace that a numbers belong to

"""


def bin2int(seq: tuple | str) -> int:
    return int("".join(str(v) for v in seq), 2)


def int2bin(x: int, N: int) -> str:
    return f"{x:0{N}b}"


"""
V1: rotating tuples
"""


def rotate(seq: tuple) -> tuple:
    return tuple([seq[-1], *seq[:-1]])


def all_rots(seq: tuple):
    for _ in range(len(seq)):
        seq = rotate(seq)
        yield seq


def make_group_tuple(N):
    seqs = list(product([0, 1], repeat=N))
    groups = {}
    for seq in seqs:
        found = False
        group = seq
        for rot in all_rots(seq):
            if rot in groups:
                group = rot
                found = True
                break
        if found:
            groups[group].append(seq)
        else:
            groups[seq] = [seq]
    return groups


# %%

"""
V2 : rotating ints

Implement rotations as bit-shifts on integers + some masks (no more silly keeping tracks of highs and lows while iterating)
"""


def make_rot(N):

    # length mask, i.e. 111...111 of len N
    l_mask = 2**N - 1

    @numba.njit("u4(u4)")
    def rot(x):
        # left rotations
        carry = x >> (N - 1)
        return (x * 2 + carry) & l_mask

    vrot = numba.vectorize("u4(u4)", target="parallel")(rot)
    ## Equivalent to
    # @numba.njit("u4[:](u4[:])", parallel=True)
    # def prot(x):
    #     y = np.empty_like(x)
    #     for i in numba.prange(len(x)):
    #         y[i] = rot(x[i])
    #     return y

    return rot, vrot


def vmap(D, f):
    return {k: sorted([f(v) for v in vals]) for k, vals in D.items()}


def arr2dict(arr):
    """ """
    D = {}
    for i, v in enumerate(arr):
        if v not in D:
            D[v] = set()
        D[v].add(i)
    return D


def dict2arr(D):
    """ """
    N = max(D.keys()) + 1
    arr = np.empty(N, dtype=np.uint32)
    for k, vals in D.items():
        for val in vals:
            arr[val] = k
    return arr


def _make_group(vrot, N):
    # Idea: "fast > smart"
    #       1. Generate the 2**N numbers of N bits
    #       2. Rotate each one N times (one full rotation)
    #       3. Record the minimum of every rotation
    #
    # Takes N * 2**N ops, but ops are fast and parallel
    # so I can go up to N=32 easily, beyond there there are
    # just too many necklaces anyways

    x = np.arange(2**N, dtype=np.uint32)
    necklaces = np.copy(x)
    for _ in range(N + 1):
        x = vrot(x)
        necklaces = np.minimum(necklaces, x)

    return necklaces


def make_group(N):
    """
    Wrapper that combines make_rot and _make_necklaces into a complete pass.
    NOTE:
        Jitting code takes time, for multiple runs of small values of N is
        better to just call make_rot once and pass it to _make_necklaces
    """
    _, vrot = make_rot(N)
    return _make_group(vrot, N)


def get_neck(N):
    """
    Precomputing everything can take a lot of space.
    `get_neck` is a fast way to get the necklace a specific number belongs to.
    All it does is rotate it N times and return the min.
    """
    rot, _ = make_rot(N)

    @numba.njit("u4(u4)")
    def min(x):
        m = x
        for _ in range(N):
            x = rot(x)
            if x < m:
                m = x
        return m

    return min


#%%

if __name__ == "__main__":
    pass

    from time import monotonic

    for N in range(4, 33):
        st = monotonic()
        group = make_group(N)
        # np.save(f"data/group-{N}.npy", group)
        necks = np.unique(group)
        np.save(f"data/necks-{N}.npy", necks)
        et = monotonic()
        print(f"{N=:<2d}:{et - st:11.3f}s")

    # import argparse
    #
    # parser = argparse.ArgumentParser(description="Process some integers.")
    # parser.add_argument(
    #     "N",
    #     type=int,
    #     help="length of the binary necklaces",
    # )
    # parser.add_argument(
    #     "--out",
    #     type=str,
    #     default="necklaces.pkl",
    #     help="the file where to save the neckalces",
    # )
    #
    # args = parser.parse_args()
    # print(args)
    # with open(args.out, "wb") as f:
    #     pickle.dump({"N": args.N, "group": make_group(args.N)}, f)
    # print(f"Saved to {args.out}")
