def test_int_bin():
    from utils import int2bin, bin2int

    N = 8
    v = 137
    assert bin2int(int2bin(v, N)) == v


def test_bin2int():
    from itertools import product
    from utils import bin2int

    N = 5
    seqs = list(product([0, 1], repeat=N))
    for i, seq in enumerate(seqs):
        assert i == bin2int(seq)
