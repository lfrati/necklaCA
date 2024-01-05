import numpy as np
import numba
from itertools import product
from numpy.lib.stride_tricks import sliding_window_view


def int2rule(N):
    return np.array([int(i) for i in f"{N:08b}"], dtype=np.uint8)


RULES = np.fliplr(np.array(list(product([0, 1], repeat=8)), dtype=np.uint8))
BASE = 2 ** np.arange(3)[::-1]


def step(state, rule):
    s = np.pad(state, (1, 1), "wrap")
    s = sliding_window_view(s, 3)
    s = (s * BASE).sum(axis=1)
    return rule[s]


def multi_step(state, rule, L):
    history = np.empty((L + 1, state.shape[0]), dtype=np.uint8)
    history[0] = state

    for i in range(1, L + 1):
        state = step(state, rule)
        history[i] = state

    return history


@numba.njit
def to_int(bits: np.ndarray) -> int:
    """
    Convert array of bits (np.uint8) to integer:
    e.g.
        to_int(np.array([0,0,0,0,1,1,0,1],dtype=np.uint8)) -> 13

    Note: numpy magic makes it faster than my other numba implementation.
    """
    BASE = 2 ** np.arange(len(bits) - 1, -1, -1)
    return (bits * BASE).sum()


@numba.njit("u1[:](u8, u8)")
def to_bits(n: int, N: int) -> np.ndarray:
    """
    Convert integer n to binary array of length N
    e.g.
        to_bits(13,8) -> array([0, 0, 0, 0, 1, 1, 0, 1], dtype=uint8)

    Note: much faster than the weird python string manipulation version.
    """
    bits = np.zeros(N, dtype=np.uint8)
    for i in range(N):
        v = n % 2
        bits[i] = v
        n = n // 2
    return bits[::-1]


@numba.njit("u8(u8, u8, u1[:])")
def step_implicit(num, N, rule):
    """
    Necklaces are uints, so to apply a CA rule I should do:
        uint -> array -> array -> uint
    that's so sad...
    Let's do bit stuff to go uint -> uint :) Lots of shifting
    """
    # c starts with the leftmost bit to loop
    a, b, c = (num & 2) >> 1, num & 1, (num >> (N - 1)) & 1
    num += b << N  # lpad to loop
    num = num >> 1
    res = 0
    m = 1
    for _ in range(0, N):
        res += m * rule[a * 4 + b * 2 + c]
        num = num >> 1
        a, b, c = num & 1, a, b
        m = m << 1
    return res


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    N = 100
    state = np.zeros(N, dtype=np.uint8)
    state[N // 2] = 1
    rule = RULES[110]
    history = multi_step(state, rule, N)
    print(history)

    plt.figure(figsize=(10, 10))
    plt.gca().invert_yaxis()
    plt.pcolormesh(history, edgecolors="gray", linewidth=1, cmap="binary")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

