from itertools import product

import numpy as np
from tqdm import trange

from data import (
    _make_group,
    arr2dict,
    bin2int,
    dict2arr,
    get_neck,
    int2bin,
    make_group,
    make_group_tuple,
    make_rot,
)


def test_int_bin():

    N = 8
    v = 137
    assert bin2int(int2bin(v, N)) == v


def test_bin2int():

    N = 5
    seqs = list(product([0, 1], repeat=N))
    for i, seq in enumerate(seqs):
        assert i == bin2int(seq)


def get_oeis(seq):
    # TODO: is there a better api? this feel very brittle
    import requests
    import json

    url = f"https://oeis.org/search?fmt=json&q={seq}&start=0"
    r = requests.get(url)
    sequence = []
    if r.status_code == 200:
        response = json.loads(r.text)
        raw = response["results"][0]["data"]
        sequence = list(map(int, raw.split(",")))
    return sequence


def contains(container, contained):
    return (set(container) & set(contained)) == set(contained)


def test_oeis():

    sizes = []
    for i in range(4, 18):
        groups = make_group_tuple(i)
        sizes.append(len(groups))

    oeis = get_oeis("A000031")

    assert contains(oeis, sizes)

    sizes = []
    for i in range(4, 22):
        arr = make_group(i)
        groups = arr2dict(arr)
        sizes.append(len(groups))

    assert contains(oeis, sizes)


def test_arr_dict():

    N = 8
    group = make_group(N)
    assert np.all(dict2arr(arr2dict(group)) == group)


def test_rots():
    # one full turn puts you back at the start

    N = 8
    rot, vrot = make_rot(N)

    print()
    start = 24
    x = start
    print(int2bin(x, N), x)
    for _ in range(N):
        x = rot(x)
        print(int2bin(x, N), x)

    assert x == start

    start = np.arange(2**N, dtype=np.uint32)
    x = np.copy(start)
    for _ in range(N):
        x = vrot(x)

    assert np.all(x == start)


def test_get_neck():

    N = 14
    n = get_neck(N)
    group = make_group(N)

    for i, v in enumerate(group):
        assert n(i) == v


def test_make_group():
    def validate(arr, rot, N):

        d = arr2dict(arr)
        for k, vals in d.items():
            assert isinstance(vals, set)
            assert k in vals
            # how do I check I got all of them?
            # pick one and rotate N times
            # every rotation has to be in the group
            # Note: this might do too many checks
            candidate = next(iter(vals))
            for _ in range(N):
                candidate = rot(candidate)
                assert candidate in vals, f"Error: {candidate} not in {vals}"

    for N in trange(4, 10):
        rot, vrot = make_rot(N)
        necklaces = _make_group(vrot, N)
        validate(necklaces, rot, N)
