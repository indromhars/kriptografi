# aes_metrics.py
# Academic S-box evaluation: Balance/Bijective, NL, SAC, BIC (summary), LAP, DAP
# Pure python (no numpy) for portability.

from typing import List, Tuple, Dict
from collections import defaultdict

def byte_to_bits(x: int) -> List[int]:
    return [(x >> i) & 1 for i in range(8)]

def hamming_weight(x: int) -> int:
    return bin(x).count("1")

# ---------- 3.3 balance & bijective ----------
def is_bijective(sbox: List[int]) -> bool:
    return len(set(sbox)) == 256 and all(0 <= v < 256 for v in sbox)

def is_balanced(sbox: List[int]) -> bool:
    # For permutation S-box, balanced in sense of equal counts of output bits 0/1 per bit
    for bit in range(8):
        zeros = sum(1 for v in sbox if ((v >> bit) & 1) == 0)
        ones = 256 - zeros
        if zeros != ones:
            return False
    return True

# ---------- 4.1 Nonlinearity ----------
def walsh_transform_boolean(f: List[int]) -> List[int]:
    # f: length-256 boolean function (0/1). Return Walsh spectrum W(a)
    W = [0]*256
    for a in range(256):
        s = 0
        for x in range(256):
            fx = f[x]
            dot = hamming_weight(a & x) % 2
            s += 1 if (fx ^ dot) == 0 else -1
        W[a] = s
    return W

def nonlinearity_per_bit(sbox: List[int]) -> List[int]:
    nls = []
    for bit in range(8):
        f = [ (sbox[x] >> bit) & 1 for x in range(256) ]
        W = walsh_transform_boolean(f)
        maxW = max(abs(v) for v in W)
        nl = 128 - (maxW // 2)
        nls.append(nl)
    return nls

# ---------- 4.2 Strict Avalanche Criterion (SAC) ----------
def sac_matrix(sbox: List[int]) -> List[List[float]]:
    # sac_matrix[i][j] = probability output bit j flips when input bit i flips
    mat = [[0.0]*8 for _ in range(8)]
    for i in range(8):
        count = [0]*8
        for x in range(256):
            y0 = sbox[x]
            y1 = sbox[x ^ (1<<i)]
            for j in range(8):
                if ((y0 >> j) & 1) != ((y1 >> j) & 1):
                    count[j] += 1
        for j in range(8):
            mat[i][j] = count[j] / 256.0
    return mat

# ---------- 4.3 Bit Independence Criterion (BIC) ----------
def bic_average_from_sac(sbox: List[int]) -> float:
    # A simple BIC proxy: average of SAC matrix off-diagonal behavior
    mat = sac_matrix(sbox)
    total = 0.0
    cnt = 0
    for i in range(8):
        for j in range(8):
            total += mat[i][j]
            cnt += 1
    return total / cnt if cnt else 0.0

# ---------- 4.4 Linear Approximation Probability (LAP) ----------
def lap_max(sbox: List[int]) -> float:
    # Compute maximum bias probability: abs(count-128)/256 for all non-zero masks a,b
    max_bias = 0.0
    for a in range(1,256):
        for b in range(1,256):
            count = 0
            for x in range(256):
                ax = hamming_weight(a & x) % 2
                bx = hamming_weight(b & sbox[x]) % 2
                if ax == bx:
                    count += 1
            bias = abs(count - 128) / 256.0
            if bias > max_bias:
                max_bias = bias
    return max_bias

# ---------- 4.6 Differential Approximation Probability (DAP) ----------
def dap_max(sbox: List[int]) -> float:
    max_prob = 0.0
    for dx in range(1,256):
        counter = defaultdict(int)
        for x in range(256):
            dy = sbox[x] ^ sbox[x ^ dx]
            counter[dy] += 1
        local_max = max(counter.values())
        prob = local_max / 256.0
        if prob > max_prob:
            max_prob = prob
    return max_prob

# ---------- Convenience runner ----------
def full_evaluation(sbox: List[int]) -> Dict:
    return {
        'bijective': is_bijective(sbox),
        'balanced': is_balanced(sbox),
        'nonlinearity_per_bit': nonlinearity_per_bit(sbox),
        'nonlinearity_avg': sum(nonlinearity_per_bit(sbox))/8.0,
        'sac_matrix': sac_matrix(sbox),
        'bic_average': bic_average_from_sac(sbox),
        'lap_max': lap_max(sbox),
        'dap_max': dap_max(sbox),
    }
