def bin2int(seq: tuple | str) -> int:
    return int("".join(str(v) for v in seq), 2)


def int2bin(x: int, N: int) -> str:
    return f"{x:0{N}b}"
