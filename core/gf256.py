AES_POLY = 0x11B

def gf_mul(a, b):
    res = 0
    for _ in range(8):
        if b & 1:
            res ^= a
        carry = a & 0x80
        a <<= 1
        if carry:
            a ^= AES_POLY
        a &= 0xFF
        b >>= 1
    return res

def gf_inverse(a):
    if a == 0:
        return 0
    for i in range(1, 256):
        if gf_mul(a, i) == 1:
            return i
