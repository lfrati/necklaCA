import numpy as np


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
    from data import make_necklaces_tuple, make_necklaces, arr2dict

    sizes = []
    for i in range(4, 18):
        groups = make_necklaces_tuple(i)
        sizes.append(len(groups))

    oeis = get_oeis("A000031")

    assert contains(oeis, sizes)

    sizes = []
    for i in range(4, 22):
        arr = make_necklaces(i)
        groups = arr2dict(arr)
        sizes.append(len(groups))

    assert contains(oeis, sizes)


def test_arr_dict():
    from data import make_necklaces, dict2arr, arr2dict

    N = 8
    necklaces = make_necklaces(N)
    assert np.all(dict2arr(arr2dict(necklaces)) == necklaces)


def test_rots():
    # one full turn puts you back at the start

    from data import make_rot
    from utils import int2bin

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
