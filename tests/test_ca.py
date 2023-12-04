import numpy as np


def test_implicit():
    from ca import RULES, step_implicit, step, to_int, to_bits

    N = 5
    rule = RULES[90]
    for i in range(2**N):
        implicit = step_implicit(i, N, rule)
        explicit = to_int(step(to_bits(i, N), rule))
        assert implicit == explicit, f"ERROR: {explicit=}!={implicit=}"


def test_explicit():
    from ca import RULES, multi_step

    rule = RULES[110]
    seed = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    expected = np.array([1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1])
    history = multi_step(seed, rule, 19)

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10, 10))
    # plt.gca().invert_yaxis()
    # plt.pcolormesh(history, edgecolors='gray', linewidth=1, cmap="binary")
    # plt.axis("off")
    # plt.tight_layout()
    # plt.show()

    assert np.all(history[-1] == expected)
