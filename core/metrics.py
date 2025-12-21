import numpy as np
import math
from itertools import product

# =========================
# Helper
# =========================
def bool_func(sbox, bit):
    return [(sbox[x] >> bit) & 1 for x in range(256)]

def walsh_hadamard(f):
    W = []
    for a in range(256):
        s = 0
        for x in range(256):
            ax = bin(a & x).count("1") % 2
            s += (-1) ** (f[x] ^ ax)
        W.append(s)
    return W


# =========================
# NONLINEARITY
# =========================
def nonlinearity(sbox):
    nls = []
    for bit in range(8):
        f = bool_func(sbox, bit)
        W = walsh_hadamard(f)
        max_w = max(abs(w) for w in W)
        nls.append(128 - max_w // 2)
    return sum(nls) / len(nls)


# =========================
# STRICT AVALANCHE CRITERION
# =========================
def sac(sbox):
    total = 0
    count = 0
    for x in range(256):
        for i in range(8):
            x2 = x ^ (1 << i)
            diff = sbox[x] ^ sbox[x2]
            total += bin(diff).count("1") / 8
            count += 1
    return total / count


# =========================
# BIT INDEPENDENCE CRITERION
# =========================
def bic(sbox):
    corrs = []
    for i in range(8):
        for j in range(i + 1, 8):
            c = 0
            for x in range(256):
                bi = (sbox[x] >> i) & 1
                bj = (sbox[x] >> j) & 1
                c += bi ^ bj
            corrs.append(abs(c / 256 - 0.5))
    return sum(corrs) / len(corrs)


# =========================
# DIFFERENTIAL APPROXIMATION PROBABILITY
# =========================
def dap(sbox):
    max_dp = 0
    for dx in range(1, 256):
        table = {}
        for x in range(256):
            dy = sbox[x] ^ sbox[x ^ dx]
            table[dy] = table.get(dy, 0) + 1
        max_dp = max(max_dp, max(table.values()))
    return max_dp / 256


# =========================
# LINEAR APPROXIMATION PROBABILITY
# =========================
def lap(sbox):
    max_lp = 0
    for a in range(1, 256):
        for b in range(1, 256):
            c = 0
            for x in range(256):
                ax = bin(a & x).count("1") % 2
                bx = bin(b & sbox[x]).count("1") % 2
                c += (-1) ** (ax ^ bx)
            max_lp = max(max_lp, abs(c))
    return max_lp / 256

def calc_correlation(arr1, arr2):
    """Menghitung koefisien korelasi (Ideal: 1.0 untuk hasil dekripsi)"""
    return np.corrcoef(arr1.flatten(), arr2.flatten())[0, 1]

def calc_psnr(arr1, arr2):
    """Menghitung PSNR (Ideal: Infinity atau sangat tinggi)"""
    mse = np.mean((arr1.astype(float) - arr2.astype(float)) ** 2)
    if mse == 0: return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))
