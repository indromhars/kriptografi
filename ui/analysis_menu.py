import streamlit as st
import numpy as np
import pandas as pd

from core.affine import generate_affine_matrix, gf2_rank
from core.sbox import build_sbox, is_bijective
from core.metrics import nonlinearity, sac, bic, dap, lap


def analysis_menu():
    st.header("🧪 Cryptographic Analysis")

    st.markdown("""
    Analisis dilakukan terhadap S-box hasil konstruksi affine dan inverse GF(2⁸).
    Hasil dibandingkan dengan karakteristik S-box AES standar.
    """)

    # =========================
    # INPUT
    # =========================
    col1, col2 = st.columns(2)

    with col1:
        dec = st.number_input(
            "Affine First Row (0–255)",
            min_value=0,
            max_value=255,
            value=7
        )
        first_row = np.array(list(map(int, f"{dec:08b}")))

    with col2:
        const = st.number_input(
            "Affine Constant (0–255)",
            min_value=0,
            max_value=255,
            value=99
        )
        constant = np.array(list(map(int, f"{const:08b}")))[::-1]

    affine = generate_affine_matrix(first_row)

    if gf2_rank(affine) != 8:
        st.error("Affine matrix tidak valid (Rank ≠ 8)")
        return

    sbox = build_sbox(affine, constant)

    if not is_bijective(sbox):
        st.error("S-box tidak bijective")
        return

    # =========================
    # METRICS
    # =========================
    if st.button("📊 Analyze S-box"):
        nl = nonlinearity(sbox)
        sac_val = sac(sbox)
        bic_val = bic(sbox)
        dap_val = dap(sbox)
        lap_val = lap(sbox)

    df = pd.DataFrame({
        "Metric": [
            "Nonlinearity (avg)",
            "SAC",
            "BIC",
            "DAP",
            "LAP"
        ],
        "Value": [
            round(nl, 2),
            round(sac_val, 4),
            round(bic_val, 4),
            round(dap_val, 6),
            round(lap_val, 6)
        ],
        "AES Reference": [
            "≈112",
            "≈0.5",
            "≈0",
            "≤ 0.0156",
            "≤ 0.125"
        ]
    })

    st.subheader("📈 Cryptographic Metrics")
    st.dataframe(df)

    st.success("Analisis selesai. Nilai mendekati referensi AES menunjukkan S-box kuat.")
