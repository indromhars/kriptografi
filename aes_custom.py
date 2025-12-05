# aes_custom.py
# AES-128 (pure python) with customizable S-box (standard AES or S-box44)
# ECB mode for demonstration. PKCS7 padding provided.
# Author: revised for student's project

from typing import List

# ---------- Standard AES S-box ----------
AES_SBOX = [
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
]

# ---------- S-box44 from the paper (Table 5) ----------
SBOX44 = [
  99,205,85,71,25,127,113,219,63,244,109,159,11,228,94,214,
  77,177,201,78,5,48,29,30,87,96,193,80,156,200,216,86,
  116,143,10,14,54,169,148,68,49,75,171,157,92,114,188,194,
  121,220,131,210,83,135,250,149,253,72,182,33,190,141,249,82,
  232,50,21,84,215,242,180,198,168,167,103,122,152,162,145,184,
  43,237,119,183,7,12,125,55,252,206,235,160,140,133,179,192,
  110,176,221,134,19,6,187,59,26,129,112,73,175,45,24,218,
  44,66,151,32,137,31,35,147,236,247,117,132,79,136,154,105,
  199,101,203,52,57,4,153,197,88,76,202,174,233,62,208,91,
  231,53,1,124,0,28,142,170,158,51,226,65,123,186,239,246,
  38,56,36,108,8,126,9,189,81,234,212,224,13,3,40,64,
  172,74,181,118,39,227,130,89,245,166,16,61,106,196,211,107,
  229,195,138,18,93,207,240,95,58,255,209,217,15,111,46,173,
  223,42,115,238,139,243,23,98,100,178,37,97,191,213,222,155,
  165,2,146,204,120,241,163,128,22,90,60,185,67,34,27,248,
  164,69,41,230,104,47,144,251,20,17,150,225,254,161,102,70
]

SBOXES = {'aes': AES_SBOX, 'sbox44': SBOX44}

def inverse_sbox(sbox: List[int]) -> List[int]:
    inv = [0]*256
    for i,v in enumerate(sbox):
        inv[v] = i
    return inv

INV_SBOXES = {name: inverse_sbox(arr) for name, arr in SBOXES.items()}

# GF(2^8) multiply (simple implementation)
def mul(a: int, b: int) -> int:
    res = 0
    for _ in range(8):
        if b & 1:
            res ^= a
        high = a & 0x80
        a = (a << 1) & 0xFF
        if high:
            a ^= 0x1B
        b >>= 1
    return res

# AES parameters
Nb = 4
Nk = 4
Nr = 10

RCON = [
    0x00000000,
    0x01000000,
    0x02000000,
    0x04000000,
    0x08000000,
    0x10000000,
    0x20000000,
    0x40000000,
    0x80000000,
    0x1b000000,
    0x36000000
]

def sub_word(word: int, sbox: List[int]) -> int:
    return ((sbox[(word >> 24) & 0xFF] << 24) |
            (sbox[(word >> 16) & 0xFF] << 16) |
            (sbox[(word >> 8) & 0xFF] << 8) |
            (sbox[word & 0xFF]))

def rot_word(word: int) -> int:
    return ((word << 8) & 0xFFFFFFFF) | ((word >> 24) & 0xFF)

def key_expansion(key: bytes, sbox: List[int]) -> List[int]:
    key_words = []
    for i in range(Nk):
        w = (key[4*i] << 24) | (key[4*i+1] << 16) | (key[4*i+2] << 8) | key[4*i+3]
        key_words.append(w)
    i = Nk
    while i < Nb * (Nr + 1):
        temp = key_words[i-1]
        if i % Nk == 0:
            temp = sub_word(rot_word(temp), sbox) ^ RCON[i//Nk]
        key_words.append(key_words[i-Nk] ^ temp)
        i += 1
    return key_words

def add_round_key(state, round_key_words):
    for c in range(4):
        word = round_key_words[c]
        state[0][c] ^= (word >> 24) & 0xFF
        state[1][c] ^= (word >> 16) & 0xFF
        state[2][c] ^= (word >> 8) & 0xFF
        state[3][c] ^= word & 0xFF

def sub_bytes(state, sbox):
    for r in range(4):
        for c in range(4):
            state[r][c] = sbox[state[r][c]]

def inv_sub_bytes(state, inv_sbox):
    for r in range(4):
        for c in range(4):
            state[r][c] = inv_sbox[state[r][c]]

def shift_rows(state):
    state[1] = state[1][1:] + state[1][:1]
    state[2] = state[2][2:] + state[2][:2]
    state[3] = state[3][3:] + state[3][:3]

def inv_shift_rows(state):
    state[1] = state[1][-1:] + state[1][:-1]
    state[2] = state[2][-2:] + state[2][:-2]
    state[3] = state[3][-3:] + state[3][:-3]

def mix_single_column(col):
    a = col.copy()
    col[0] = mul(a[0],2) ^ mul(a[1],3) ^ a[2] ^ a[3]
    col[1] = a[0] ^ mul(a[1],2) ^ mul(a[2],3) ^ a[3]
    col[2] = a[0] ^ a[1] ^ mul(a[2],2) ^ mul(a[3],3)
    col[3] = mul(a[0],3) ^ a[1] ^ a[2] ^ mul(a[3],2)

def mix_columns(state):
    for c in range(4):
        col = [state[r][c] for r in range(4)]
        mix_single_column(col)
        for r in range(4):
            state[r][c] = col[r]

def inv_mix_single_column(col):
    a = col.copy()
    col[0] = mul(a[0],0x0e) ^ mul(a[1],0x0b) ^ mul(a[2],0x0d) ^ mul(a[3],0x09)
    col[1] = mul(a[0],0x09) ^ mul(a[1],0x0e) ^ mul(a[2],0x0b) ^ mul(a[3],0x0d)
    col[2] = mul(a[0],0x0d) ^ mul(a[1],0x09) ^ mul(a[2],0x0e) ^ mul(a[3],0x0b)
    col[3] = mul(a[0],0x0b) ^ mul(a[1],0x0d) ^ mul(a[2],0x09) ^ mul(a[3],0x0e)

def inv_mix_columns(state):
    for c in range(4):
        col = [state[r][c] for r in range(4)]
        inv_mix_single_column(col)
        for r in range(4):
            state[r][c] = col[r]

def bytes2state(block: bytes):
    assert len(block) == 16
    state = [[0]*4 for _ in range(4)]
    for r in range(4):
        for c in range(4):
            state[r][c] = block[c*4 + r]
    return state

def state2bytes(state):
    out = bytearray(16)
    for c in range(4):
        for r in range(4):
            out[c*4 + r] = state[r][c]
    return bytes(out)

def encrypt_block(block: bytes, round_keys: List[int], sbox: List[int]) -> bytes:
    state = bytes2state(block)
    add_round_key(state, round_keys[0:4])
    for rnd in range(1, Nr):
        sub_bytes(state, sbox)
        shift_rows(state)
        mix_columns(state)
        add_round_key(state, round_keys[rnd*4:(rnd+1)*4])
    sub_bytes(state, sbox)
    shift_rows(state)
    add_round_key(state, round_keys[Nr*4:(Nr+1)*4])
    return state2bytes(state)

def decrypt_block(block: bytes, round_keys: List[int], inv_sbox: List[int]) -> bytes:
    state = bytes2state(block)
    add_round_key(state, round_keys[Nr*4:(Nr+1)*4])
    for rnd in range(Nr-1, 0, -1):
        inv_shift_rows(state)
        inv_sub_bytes(state, inv_sbox)
        add_round_key(state, round_keys[rnd*4:(rnd+1)*4])
        inv_mix_columns(state)
    inv_shift_rows(state)
    inv_sub_bytes(state, inv_sbox)
    add_round_key(state, round_keys[0:4])
    return state2bytes(state)

def pkcs7_pad(data: bytes, block_size: int = 16) -> bytes:
    pad_len = block_size - (len(data) % block_size)
    return data + bytes([pad_len])*pad_len

def pkcs7_unpad(data: bytes) -> bytes:
    if len(data) == 0:
        raise ValueError("Invalid PKCS7 padding (empty data)")
    pad_len = data[-1]
    if pad_len < 1 or pad_len > 16:
        raise ValueError("Invalid PKCS7 padding value")
    if data[-pad_len:] != bytes([pad_len])*pad_len:
        raise ValueError("Invalid PKCS7 padding bytes")
    return data[:-pad_len]

def _validate_and_prepare_key(key: bytes) -> bytes:
    if len(key) not in (16, 24, 32):
        if len(key) < 16:
            key = key.ljust(16, b'\0')
        else:
            key = key[:16]
    return key[:16]

def encrypt_bytes(plaintext: bytes, key: bytes, sbox_name: str = 'aes') -> bytes:
    key = _validate_and_prepare_key(key)
    sbox = SBOXES.get(sbox_name.lower())
    if sbox is None:
        raise ValueError("Unknown sbox_name. Choose 'aes' or 'sbox44'")
    round_keys = key_expansion(key, sbox)
    data = pkcs7_pad(plaintext, 16)
    out = bytearray()
    for i in range(0, len(data), 16):
        out.extend(encrypt_block(data[i:i+16], round_keys, sbox))
    return bytes(out)

def decrypt_bytes(ciphertext: bytes, key: bytes, sbox_name: str = 'aes') -> bytes:
    key = _validate_and_prepare_key(key)
    sbox = SBOXES.get(sbox_name.lower())
    if sbox is None:
        raise ValueError("Unknown sbox_name. Choose 'aes' or 'sbox44'")
    inv_sbox = INV_SBOXES[sbox_name.lower()]
    round_keys = key_expansion(key, sbox)
    if len(ciphertext) % 16 != 0:
        raise ValueError("Ciphertext length must be multiple of 16")
    out = bytearray()
    for i in range(0, len(ciphertext), 16):
        out.extend(decrypt_block(ciphertext[i:i+16], round_keys, inv_sbox))
    return pkcs7_unpad(bytes(out))
