from itertools import product

import numba
import numpy as np

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

# %%

"""
V1: rotating tuples
"""


def rotate(seq: tuple) -> tuple:
    return tuple([seq[-1], *seq[:-1]])


def all_rots(seq: tuple):
    for _ in range(len(seq)):
        seq = rotate(seq)
        yield seq


def make_necklaces_tuple(N):
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


def _make_necklaces(vrot, N):
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


def make_necklaces(N):
    """
    Wrapper that combines make_rot and _make_necklaces into a complete pass.
    NOTE:
        Jitting code takes time, for multiple runs of small values of N is
        better to just call make_rot once and pass it to _make_necklaces
    """
    _, vrot = make_rot(N)
    return _make_necklaces(vrot, N)


if __name__ == "__main__":

    import numpy as np

    N = 8
    necklaces = make_necklaces(N)

    assert np.all(dict2arr(arr2dict(necklaces)) == necklaces)
