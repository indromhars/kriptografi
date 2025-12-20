import numpy as np
from core.gf256 import gf_inverse

def build_sbox(affine_matrix, constant):
    sbox = []

    for x in range(256):
        inv = gf_inverse(x)

        # inverse ke vektor bit (LSB di bawah)
        bits = np.array(list(map(int, f"{inv:08b}")))[::-1]

        # affine transform
        out = (affine_matrix @ bits) % 2

        # XOR dengan konstanta
        out ^= constant

        # balik ke integer
        sbox.append(int("".join(map(str, out[::-1])), 2))

    return sbox


def is_bijective(sbox):
    return len(set(sbox)) == 256
