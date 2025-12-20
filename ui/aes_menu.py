import streamlit as st
import numpy as np
from core.sbox import build_sbox
from core.affine import generate_affine_matrix, gf2_rank


def sub_bytes(state, sbox):
    return [sbox[b] for b in state]


def aes_menu():
    st.header("🔐 AES Encrypt / Decrypt (Demo)")

    st.warning("""
    Implementasi ini bersifat edukatif:
    - Fokus pada SubBytes menggunakan custom S-box
    - BUKAN AES full standard (MixColumns, ShiftRows disederhanakan)
    """)

    col1, col2 = st.columns(2)

    with col1:
        plaintext = st.text_input("Plaintext (hex, 16 bytes)", "00112233445566778899aabbccddeeff")
    with col2:
        key = st.text_input("Key (hex, 16 bytes)", "000102030405060708090a0b0c0d0e0f")

    dec = st.number_input("Affine First Row", 0, 255, 7)
    const = st.number_input("Affine Constant", 0, 255, 99)

    first_row = np.array(list(map(int, f"{dec:08b}")))
    constant = np.array(list(map(int, f"{const:08b}")))[::-1]
    affine = generate_affine_matrix(first_row)

    if gf2_rank(affine) != 8:
        st.error("Affine matrix tidak valid")
        return

    sbox = build_sbox(affine, constant)

    if st.button("🔐 Encrypt (SubBytes Only)"):
        state = [int(plaintext[i:i+2], 16) for i in range(0, 32, 2)]
        out = sub_bytes(state, sbox)
        st.success("Ciphertext (hex)")
        st.code("".join(f"{b:02x}" for b in out))
