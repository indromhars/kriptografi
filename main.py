# app.py
import streamlit as st
import binascii
from aes_custom import encrypt_bytes, decrypt_bytes, SBOXES
from aes_custom import AES_SBOX, SBOX44
import aes_metrics
import time
from textwrap import shorten

st.set_page_config(page_title="AES custom S-box (S-box44) — Demo + Metrics", layout="centered")

st.title("AES: Standard S-box vs S-box44 (paper)")

st.markdown("""
Masukkan plaintext dan key. Pilih S-box untuk enkripsi/dekripsi.
Gunakan panel Evaluation untuk menjalankan perhitungan akademik (3.1 — 4.6).
""")

# --- Input area ---
col1, col2 = st.columns(2)
with col1:
    key_input = st.text_input("Key (text)", value="my secret key 123")
    key_hex = st.text_input("Or Key (hex, optional)", value="")
with col2:
    sbox_choice = st.radio("S-box selection (affects both encryption and key schedule)", ("aes", "sbox44"))
    st.info("Mode: ECB (for demo). For production use authenticated modes (GCM) and proper key management.")

plaintext = st.text_area("Plaintext (text)", value="Hello world! This is a test.")
plaintext_hex = st.text_input("Or Plaintext (hex, optional)", value="")

st.write(" ")

col_a, col_b = st.columns(2)
with col_a:
    if st.button("Encrypt"):
        try:
            key_bytes = binascii.unhexlify(key_hex.strip()) if key_hex.strip() else key_input.encode('utf-8')
            pt_bytes = binascii.unhexlify(plaintext_hex.strip()) if plaintext_hex.strip() else plaintext.encode('utf-8')
            ct = encrypt_bytes(pt_bytes, key_bytes, sbox_name=sbox_choice)
            st.success("Encryption OK")
            st.code(f"Ciphertext (hex): {binascii.hexlify(ct).decode()}")
            other = "sbox44" if sbox_choice == "aes" else "aes"
            ct_other = encrypt_bytes(pt_bytes, key_bytes, sbox_name=other)
            st.write("Comparison (other S-box):")
            st.code(f"{other} ciphertext (hex): {binascii.hexlify(ct_other).decode()}")
        except Exception as e:
            st.error(f"Encryption error: {e}")

with col_b:
    if st.button("Decrypt"):
        try:
            key_bytes = binascii.unhexlify(key_hex.strip()) if key_hex.strip() else key_input.encode('utf-8')
            if plaintext_hex.strip():
                ct_bytes = binascii.unhexlify(plaintext_hex.strip())
            else:
                try:
                    ct_bytes = binascii.unhexlify(plaintext.strip())
                except Exception:
                    st.warning("To decrypt, paste ciphertext as hex into 'Plaintext (text)' or 'Plaintext (hex, optional)'.")
                    ct_bytes = b""
            if not ct_bytes:
                st.error("No ciphertext provided to decrypt.")
            else:
                pt = decrypt_bytes(ct_bytes, key_bytes, sbox_name=sbox_choice)
                st.success("Decryption OK")
                try:
                    st.code(f"Plaintext (utf-8): {pt.decode('utf-8')}")
                except UnicodeDecodeError:
                    st.code(f"Plaintext (hex): {binascii.hexlify(pt).decode()}")
        except Exception as e:
            st.error(f"Decryption error: {e}")

st.markdown("---")
st.markdown("Reference: S-box44 table taken from the uploaded paper (Jurnal Pak Alam). :contentReference[oaicite:0]{index=0}")

# --- Evaluation panel ---
st.header("Academic Evaluation (sections 3.1 — 4.6)")

eval_box = st.expander("Run full evaluation (this may take time)", expanded=False)
with eval_box:
    target = st.selectbox("Choose S-box to evaluate", ("aes", "sbox44"))
    if st.button("Run Evaluation"):
        sbox = AES_SBOX if target == "aes" else SBOX44
        t0 = time.time()
        with st.spinner("Running evaluation (NL, SAC, LAP, DAP)..."):
            # 1. balance & bijective
            bij = aes_metrics.is_bijective(sbox)
            bal = aes_metrics.is_balanced(sbox)
            st.subheader("3.3 Balance & Bijective")
            st.write("Bijective:", bij)
            st.write("Balanced (per-output-bit equal 0/1 counts):", bal)

            # 2. Nonlinearity
            st.subheader("4.1 Nonlinearity (NL)")
            nl_bits = aes_metrics.nonlinearity_per_bit(sbox)
            st.write("NL per output bit (bit0..bit7):")
            st.write(nl_bits)
            avg_nl = sum(nl_bits) / len(nl_bits)
            st.write("Average NL:", avg_nl)

            # 3. SAC
            st.subheader("4.2 Strict Avalanche Criterion (SAC)")
            sac_mat = aes_metrics.sac_matrix(sbox)
            st.write("SAC matrix (rows: input bit flipped, cols: output bits). Values close to 0.5 are ideal.")
            # pretty print matrix
            for i, row in enumerate(sac_mat):
                st.write(f"Input bit {i}:", [round(v,5) for v in row])

            # 4. BIC (simple avg)
            st.subheader("4.3 Bit Independence Criterion (BIC) — summary")
            bic_avg = aes_metrics.bic_average_from_sac(sbox)
            st.write("BIC (average SAC):", round(bic_avg, 6))

            # 5. LAP
            st.subheader("4.4 Linear Approximation Probability (LAP)")
            st.write("Computing max LAP (this is the heaviest step)...")
            lap_val = aes_metrics.lap_max(sbox)
            st.write("Max LAP (bias):", lap_val)

            # 6. DAP
            st.subheader("4.6 Differential Approximation Probability (DAP)")
            dap_val = aes_metrics.dap_max(sbox)
            st.write("Max DAP (over all non-zero input differences):", dap_val)

        t1 = time.time()
        st.success(f"Evaluation finished in {round(t1-t0,2)} seconds.")

st.markdown("---")
st.caption("If some computations take long on your machine, consider running evaluation only for the specific S-box (sbox44) or improving performance by using numpy/numba.")
