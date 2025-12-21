import streamlit as st
import numpy as np
import pandas as pd
from collections import defaultdict
import math
from datetime import datetime
import os
import json
import plotly.graph_objects as go
import plotly.express as px

# --- HELPERS (sesuai lampiran user) ---
def get_bit(value, n):
    return (value >> n) & 1

def hamming_weight(n):
    return bin(n).count('1')

def boolean_nonlinearity(bits):
    """Menghitung NL untuk satu deret bit (Boolean Function) menggunakan FWHT."""
    size = len(bits)
    n = int(np.log2(size))
    # map 0 -> 1, 1 -> -1
    arr = np.array([1 if b == 0 else -1 for b in bits], dtype=np.int32)
    # in-place FWHT
    h = 1
    while h < size:
        for i in range(0, size, h * 2):
            for j in range(i, i + h):
                x = arr[j]
                y = arr[j + h]
                arr[j] = x + y
                arr[j + h] = x - y
        h *= 2
    max_walsh = np.max(np.abs(arr))
    return int((2 ** (n - 1)) - 0.5 * max_walsh)

# --- ADD: basic validation helper (fix NameError) ---
def validate_sbox_bijective(sbox_values):
    """
    Validasi dasar S-box: bijective dan domain/range lengkap 0..255.
    Return: (is_bijective: bool, unique_count: int)
    """
    vals = list(sbox_values)
    unique_values = set(vals)
    unique_count = len(unique_values)
    is_bijective = (unique_count == 256) and (min(unique_values) == 0) and (max(unique_values) == 255)
    return is_bijective, unique_count

# --- REPLACED FUNCTIONS (sesuai lampiran user) ---

def calculate_nonlinearity(sbox_values):
    """
    Hitung Nonlinearity seperti lampiran:
    - untuk setiap mask m (1..255) gabungkan bit output dengan parity(sbox[x] & m)
    - hitung boolean_nonlinearity pada vektor itu
    - kembalikan minimal (int), rata-rata, dan list nilai untuk kompatibilitas
    """
    size = 256
    nl_values = []
    for m in range(1, size):
        combined_bits = [hamming_weight(sbox_values[x] & m) % 2 for x in range(size)]
        nl_values.append(boolean_nonlinearity(combined_bits))
    min_nl = int(min(nl_values))
    avg_nl = float(np.mean(nl_values))
    return min_nl, avg_nl, nl_values

def calculate_bic_nl(sbox_values):
    """
    BIC-NL: Nonlinearity dari XOR pasangan output bits (menggunakan boolean_nonlinearity)
    """
    n = 8
    nl_values = []
    for j in range(n):
        for k in range(j + 1, n):
            combined_bits = [ (get_bit(sbox_values[x], j) ^ get_bit(sbox_values[x], k)) for x in range(256) ]
            nl_values.append(boolean_nonlinearity(combined_bits))
    return int(min(nl_values)) if nl_values else 0

def build_linear_approximation_table(sbox_values):
    """
    Pembentukan Linear Approximation Table (LAT):
    LAT[a][b] = |#{x: a·x = b·S(x)}| - 128
    dimana a adalah input mask, b adalah output mask
    """
    sbox = np.array(sbox_values, dtype=np.uint32)
    lat = np.zeros((256, 256), dtype=np.int32)
    
    for a in range(256):
        for b in range(256):
            count = 0
            for x in range(256):
                # Hitung a · x (dot product)
                input_sum = 0
                for bit_pos in range(8):
                    if (a >> bit_pos) & 1:
                        input_sum ^= (x >> bit_pos) & 1
                
                # Hitung b · S(x)
                output_sum = 0
                for bit_pos in range(8):
                    if (b >> bit_pos) & 1:
                        output_sum ^= (sbox[x] >> bit_pos) & 1
                
                if input_sum == output_sum:
                    count += 1
            
            lat[a][b] = count - 128
    
    return lat

def calculate_lap(lat):
    """
    Linear Approximation Probability (LAP):
    LAP = max|LAT[a][b]| / 256 untuk semua a, b (kecuali a=0 atau b=0)
    Mengukur resistensi terhadap linear cryptanalysis
    """
    max_lat = 0
    for a in range(1, 256):
        for b in range(1, 256):
            max_lat = max(max_lat, abs(lat[a][b]))
    
    lap = max_lat / 256
    return lap, max_lat

def build_difference_distribution_table(sbox_values):
    """
    Pembentukan Difference Distribution Table (DDT):
    DDT[Δx][Δy] = #{x: S(x) ⊕ S(x ⊕ Δx) = Δy}
    """
    sbox = np.array(sbox_values, dtype=np.uint32)
    ddt = np.zeros((256, 256), dtype=np.int32)
    
    for delta_x in range(256):
        for x in range(256):
            x_prime = x ^ delta_x
            delta_y = sbox[x] ^ sbox[x_prime]
            ddt[delta_x][delta_y] += 1
    
    return ddt

def calculate_differential_uniformity(ddt):
    """
    Differential Uniformity (DU):
    DU = max(DDT[Δx][Δy]) untuk semua Δx ≠ 0, Δy
    Nilai kecil (DU ≤ 4) menunjukkan resistensi terhadap differential cryptanalysis
    """
    du = 0
    for delta_x in range(1, 256):
        du = max(du, np.max(ddt[delta_x]))
    
    return du

def calculate_dap(ddt):
    """
    Differential Approximation Probability (DAP):
    DAP = max(DDT[Δx][Δy]) / 256 untuk semua Δx ≠ 0, Δy
    """
    du = calculate_differential_uniformity(ddt)
    dap = du / 256
    return dap, du

def calculate_sac(sbox_values):
    """
    SAC sesuai lampiran:
    S(i,j) = (1/2^n) * sum_x [ f_j(x) XOR f_j(x ⊕ e_i) ]
    return (avg_sac, per_input_avg_list)
    """
    size = 256
    n = 8
    sac_matrix = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for x in range(size):
            diff = sbox_values[x] ^ sbox_values[x ^ (1 << i)]
            for j in range(n):
                sac_matrix[i, j] += ((diff >> j) & 1)
        sac_matrix[i, :] = sac_matrix[i, :] / float(size)

    avg_sac = float(np.mean(sac_matrix))
    per_input_avg = [float(np.mean(sac_matrix[i, :])) for i in range(n)]
    return avg_sac, per_input_avg, sac_matrix

def calculate_bic_sac(sbox_values):
    """
    BIC-SAC sesuai lampiran:
    - untuk setiap pair (j,k): g(x)=f_j XOR f_k
    - hitung SAC_g = avg over input bits i of Pr[g(x)!=g(x^e_i)]
    - BIC-SAC = average SAC_g over all pairs
    """
    n = 8
    size = 256
    pair_sac = []
    for j in range(n):
        for k in range(j + 1, n):
            g = np.array([ (get_bit(sbox_values[x], j) ^ get_bit(sbox_values[x], k)) for x in range(size) ], dtype=np.uint8)
            sac_per_i = []
            for i in range(n):
                diff_count = 0
                for x in range(size):
                    if g[x] != g[x ^ (1 << i)]:
                        diff_count += 1
                sac_per_i.append(diff_count / float(size))
            sac_g = float(np.mean(sac_per_i))
            pair_sac.append(sac_g)
    bic_sac = float(np.mean(pair_sac)) if pair_sac else 0.0
    return bic_sac

def calculate_final_score(sac_value, bic_sac_value):
    """
    Final score S per lampiran:
    S = ( |SAC - 0.5| + |BIC_SAC - 0.5| ) / 2
    Nilai S lebih dekat ke 0 = lebih baik.
    """
    return (abs(sac_value - 0.5) + abs(bic_sac_value - 0.5)) / 2

def calculate_avalanche_degree(sbox_values):
    """
    Avalanche Degree (AD):
    Rata-rata jumlah bit output yang berubah ketika satu bit input di-flip.
    Ideal: sekitar 4 (half of 8 bits).
    """
    sbox = np.array(sbox_values, dtype=np.uint32)
    avalanche_degrees = []
    
    for bit_pos in range(8):
        total_changes = 0
        
        for x in range(256):
            x_flip = x ^ (1 << bit_pos)
            output_diff = sbox[x] ^ sbox[x_flip]
            hamming_weight = bin(output_diff).count('1')
            total_changes += hamming_weight
        
        avg_changes = total_changes / 256
        avalanche_degrees.append(avg_changes)
    
    avg_avalanche = np.mean(avalanche_degrees)
    return avg_avalanche, avalanche_degrees

def calculate_transparency_order(sbox_values):
    """
    Transparency Order (TO):
    Mengukur ketahanan terhadap serangan side-channel berbasis korelasi.
    TO = rata-rata jumlah pasangan (x, x⊕Δ) dimana output berubah tepat 4 bit
    untuk setiap Δ ≠ 0.
    
    Nilai TO rendah (< 0.3) menunjukkan resistensi lebih baik.
    """
    sbox = np.array(sbox_values, dtype=np.uint32)
    to_values = []
    
    for delta in range(1, 256):
        four_bit_change = 0
        
        for x in range(256):
            x_prime = x ^ delta
            output_diff = sbox[x] ^ sbox[x_prime]
            hamming_weight = bin(output_diff).count('1')
            
            if hamming_weight == 4:
                four_bit_change += 1
        
        to = four_bit_change / 256
        to_values.append(to)
    
    avg_to = np.mean(to_values)
    return avg_to, to_values

def calculate_correlation_immunity(sbox_values):
    """
    Correlation Immunity (CI):
    Mengukur ketergantungan statistik antara output S-box dan subset bit input.
    
    CI = 0: Output bergantung pada semua bit input
    CI > 0: Output independen dari subset tertentu bit input
    
    Untuk S-box 8-bit general purpose, biasanya CI = 0.
    """
    # Implementasi sederhana: Check apakah output independent dari single input bit
    sbox = np.array(sbox_values, dtype=np.uint32)
    ci = 0
    
    for bit_pos in range(8):
        # Count output distribution when input bit is 0 vs 1
        output_0 = defaultdict(int)
        output_1 = defaultdict(int)
        
        for x in range(256):
            if (x >> bit_pos) & 1 == 0:
                output_0[sbox[x]] += 1
            else:
                output_1[sbox[x]] += 1
        
        # Check jika distribusi sama (independent)
        if output_0 == output_1:
            ci += 1
    
    return ci

def calculate_strength_values(metrics):
    """
    Kalkulasi Strength Values:
    SV (Paper) = (NL/128)*2 + (1-LAP)*8 + (1-DAP)*4 + (8-DU)/2 + (1-TO)*2
    Extended Score = SV + (SAC*2) + (CI*0.5)
    """
    nl, _, _ = metrics['nonlinearity']
    lap, _ = metrics['lap']
    dap, du = metrics['dap']
    _, to_values = metrics['transparency_order']
    avg_to = np.mean(to_values)
    avg_sac = metrics['sac'][0]
    ci = metrics['correlation_immunity']
    
    sv_paper = (nl / 128) * 2 + (1 - lap) * 8 + (1 - dap) * 4 + (8 - du) / 2 + (1 - avg_to) * 2
    extended_score = sv_paper + (avg_sac * 2) + (ci * 0.5)
    
    # Count excellent criteria
    excellent_count = 0
    if nl >= 100:
        excellent_count += 1
    if avg_sac >= 0.4:
        excellent_count += 1
    if lap <= 0.0625:
        excellent_count += 1
    if dap <= 0.02:
        excellent_count += 1
    if du <= 4:
        excellent_count += 1
    if avg_to <= 0.3:
        excellent_count += 1
    if ci > 0:
        excellent_count += 1
    
    return sv_paper, extended_score, excellent_count

# --- ADD: persistent storage helpers ---
def _saved_tests_filepath():
    # data folder at project root: ../data relative to this file
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, "saved_tests.json")

def load_saved_tests_from_disk():
    """Load saved_tests from JSON file into session_state if not already loaded."""
    if "saved_tests_loaded" in st.session_state and st.session_state.saved_tests_loaded:
        return
    path = _saved_tests_filepath()
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            st.session_state.saved_tests = saved
        except Exception:
            st.session_state.saved_tests = []
    else:
        st.session_state.saved_tests = []
    st.session_state.saved_tests_loaded = True

def save_saved_tests_to_disk():
    """Persist st.session_state.saved_tests to JSON file."""
    path = _saved_tests_filepath()
    saved = st.session_state.get("saved_tests", [])
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(saved, f, indent=2, ensure_ascii=False)
    except Exception:
        # fail silently; session state still holds data
        pass

def sbox_testing_menu():
    st.header("🧪 S-Box Testing dan Evaluasi Kriptografi")
    
    # Ensure persistent saved_tests loaded once per app run
    load_saved_tests_from_disk()
    
    testing_tabs = st.tabs([
        "Quick Summary",
        "Detailed Analysis",
        "Visualizations",
        "Interpretasi",
        "Compare"
    ])
    
    # Quick Summary Tab
    with testing_tabs[0]:
        st.subheader("Quick Summary")
        
        if "validation_results" not in st.session_state:
            st.info("💡 Jalankan validasi terlebih dahulu di menu 'S-Box' → 'Validation'")
        else:
            sbox_values = st.session_state.validation_results['sbox']
            
            if st.button("🔍 Run Comprehensive Tests", key="run_tests"):
                with st.spinner("Melakukan pengujian komprehensif..."):
                    # 1. Validasi Dasar
                    is_bijective, unique_count = validate_sbox_bijective(sbox_values)
                    
                    # 2. Nonlinearity
                    nonlinearity = calculate_nonlinearity(sbox_values)
                    
                    # 3. BIC-NL
                    bic_nl = calculate_bic_nl(sbox_values)
                    
                    # 4. LAT dan LAP
                    lat = build_linear_approximation_table(sbox_values)
                    lap_result = calculate_lap(lat)
                    
                    # 5. DDT dan Differential Metrics
                    ddt = build_difference_distribution_table(sbox_values)
                    dap_result = calculate_dap(ddt)
                    
                    # 6. SAC dan Avalanche
                    sac_result = calculate_sac(sbox_values)
                    avalanche = calculate_avalanche_degree(sbox_values)
                    
                    # 7. BIC-SAC
                    bic_sac = calculate_bic_sac(sbox_values)
                    
                    # 7.5 Final S score (new)
                    avg_sac = sac_result[0]
                    s_value = calculate_final_score(avg_sac, bic_sac)
                    
                    # 8. Transparency Order
                    to_result = calculate_transparency_order(sbox_values)
                    
                    # 9. Correlation Immunity
                    ci = calculate_correlation_immunity(sbox_values)
                    
                    # Simpan hasil (inklud S)
                    st.session_state.test_results = {
                        'bijective': is_bijective,
                        'nonlinearity': nonlinearity,
                        'bic_nl': bic_nl,
                        'lat': lat,
                        'lap': lap_result,
                        'ddt': ddt,
                        'dap': dap_result,
                        'sac': sac_result,
                        'avalanche': avalanche,
                        'bic_sac': bic_sac,
                        's_value': s_value,                # <-- new
                        'transparency_order': to_result,
                        'correlation_immunity': ci
                    }
                
                # Save a summary of the results for comparison
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                saved_tests = st.session_state.get("saved_tests", [])
                # determine decimal index of the first_row if available
                first_row = st.session_state.get('validation_results', {}).get('first_row')
                decimal_index = int("".join(map(str, first_row)), 2) if first_row is not None else None
                saved_tests.append({
                    "decimal": decimal_index,
                    "s_value": s_value,
                    "nl": nonlinearity[0],
                    "du": dap_result[1],
                    "to": float(to_result[0]),
                    "lap": lap_result[0],
                    "sac": float(sac_result[0]),
                    "timestamp": timestamp
                })
                st.session_state.saved_tests = saved_tests
                save_saved_tests_to_disk()

                # Display Results
                st.markdown("### ✅ Validasi Dasar")
                if is_bijective:
                    st.success(f"✅ S-Box adalah BIJECTIVE (Unique values: {unique_count})")
                else:
                    st.error(f"❌ S-Box BUKAN bijective (Unique values: {unique_count})")
                
                st.markdown("### 📊 Core Metrics")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    nl, avg_nl, _ = nonlinearity
                    st.metric("Nonlinearity (NL)", f"{nl}")
                    st.metric("BIC-NL", f"{bic_nl}")
                    st.metric("LAP", f"{lap_result[0]:.6f}")
                    st.metric("CI", f"{ci}")
                
                with col2:
                    avg_sac = sac_result[0]
                    st.metric("SAC", f"{avg_sac:.6f}")
                    st.metric("BIC-SAC", f"{bic_sac:.6f}")
                    st.metric("DAP", f"{dap_result[0]:.6f}")
                
                with col3:
                    du = dap_result[1]
                    avg_ad, _ = avalanche
                    avg_to, _ = to_result
                    st.metric("DU", f"{du}")
                    st.metric("AD", f"{avg_ad:.6f}")
                    st.metric("TO", f"{avg_to:.6f}")
                    

                # Display Final S score under Strength Assessment
                st.markdown("---")
                st.metric("Final S Score (lower is better)", f"{s_value:.6f}")

                st.markdown("---")
                st.markdown("### 📈 Strength Assessment")
                
                metrics = {
                    'nonlinearity': nonlinearity,
                    'bic_nl': bic_nl,
                    'lap': lap_result,
                    'dap': dap_result,
                    'sac': sac_result,
                    'avalanche': avalanche,
                    'bic_sac': bic_sac,
                    'transparency_order': to_result,
                    'correlation_immunity': ci
                }
                
                sv_paper, extended_score, excellent_count = calculate_strength_values(metrics)
                
                assess_col1, assess_col2, assess_col3 = st.columns(3)
                
                with assess_col1:
                    st.metric("SV (Paper)", f"{sv_paper:.6f}")
                
                with assess_col2:
                    st.metric("Extended Score", f"{extended_score:.6f}")
                
                with assess_col3:
                    st.metric("Excellent Criteria", f"{excellent_count}/7")
                
                if excellent_count >= 6:
                    st.success(f"✅ **EXCELLENT** - {excellent_count}/7")
                elif excellent_count >= 4:
                    st.info(f"✅ **GOOD** - {excellent_count}/7")
                elif excellent_count >= 2:
                    st.warning(f"⚠️ **FAIR** - {excellent_count}/7")
                else:
                    st.error(f"❌ **POOR** - {excellent_count}/7")
                
                # Download Results
                st.markdown("---")
                st.markdown("### 📥 Download Results")
                
                report_text = f"""S-Box Cryptographic Evaluation Report
=====================================

1. BASIC VALIDATION
-------------------
Bijective Property: {'YES' if is_bijective else 'NO'}
Unique Values: {unique_count}/256
Domain Coverage: {'Complete' if is_bijective else 'Incomplete'}

2. NONLINEARITY METRICS
----------------------
Nonlinearity (NL): {nl}
BIC-NL: {bic_nl}
Average NL: {avg_nl:.6f}

3. LINEAR RESISTANCE
-------------------
Linear Approximation Probability (LAP): {lap_result[0]:.6f}
Max LAT Value: {lap_result[1]}

4. DIFFERENTIAL RESISTANCE
-------------------------
Differential Uniformity (DU): {du}
Differential Approximation Probability (DAP): {dap_result[0]:.6f}

5. AVALANCHE PROPERTIES
---------------------
Strict Avalanche Criterion (SAC): {avg_sac:.6f}
BIC-SAC: {bic_sac:.6f}
Avalanche Degree (AD): {avg_ad:.6f}

6. SIDE-CHANNEL RESISTANCE
--------------------------
Transparency Order (TO): {avg_to:.6f}

7. STATISTICAL INDEPENDENCE
---------------------------
Correlation Immunity (CI): {ci}

8. STRENGTH ASSESSMENT
---------------------
SV (Paper Formula): {sv_paper:.6f}
Extended Score: {extended_score:.6f}
Excellent Criteria: {excellent_count}/7

9. FINAL SCORE
--------------
S (|SAC-0.5| + |BIC_SAC-0.5|) / 2 = {s_value:.6f}

INTERPRETATION
==============
[See 'Interpretasi' tab for detailed analysis]
"""
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.download_button(
                        label="📥 Download Report (.txt)",
                        data=report_text,
                        file_name="sbox_evaluation_report.txt",
                        mime="text/plain"
                    )
                
                with col2:
                    json_report = {
                        "basic": {"bijective": is_bijective, "unique_values": unique_count},
                        "nonlinearity": {"nl": int(nl), "bic_nl": int(bic_nl), "avg_nl": float(avg_nl)},
                        "linear_resistance": {"lap": float(lap_result[0]), "max_lat": int(lap_result[1])},
                        "differential_resistance": {"du": int(du), "dap": float(dap_result[0])},
                        "avalanche": {"sac": float(avg_sac), "bic_sac": float(bic_sac), "ad": float(avg_ad)},
                        "side_channel": {"to": float(avg_to)},
                        "statistical": {"ci": int(ci)},
                        "strength": {"sv_paper": float(sv_paper), "extended_score": float(extended_score), "excellent_criteria": excellent_count},
                        "final_score": {"S": float(s_value)}   # <-- new
                    }
                    
                    st.download_button(
                        label="📥 Download Report (.json)",
                        data=json.dumps(json_report, indent=2),
                        file_name="sbox_evaluation_report.json",
                        mime="application/json"
                    )
            
    
    # Detailed Analysis Tab
    with testing_tabs[1]:
        st.subheader("Detailed Test Results")
        
        if "test_results" not in st.session_state:
            st.info("💡 Jalankan tests terlebih dahulu di tab 'Quick Summary'")
        else:
            results = st.session_state.test_results
            sbox_values = st.session_state.validation_results['sbox']

            # 1. Selectbox untuk memilih parameter
            selected_test = st.selectbox(
                "Select test to view details:",
                ["Nonlinearity (NL)", "Strict Avalanche Criterion (SAC)", "BIC-NL", "BIC-SAC", "Linear Approximation (LAP)", "Differential Approximation (DAP)"]
            )

            st.markdown("---")

            # --- Rincian masing-masing Parameter ---
            if selected_test == "Nonlinearity (NL)":
                st.subheader("Nonlinearity Test")
                st.write("Nonlinearity measures the minimum distance between the Boolean functions representing the S-box and all affine functions. Higher values indicate better resistance to linear cryptanalysis.")
                st.info("Ideal value: 112 for 8-bit S-boxes")

                min_nl, avg_nl, nl_per_bit = results['nonlinearity']
                
                # Baris Metrik Utama
                col1, col2, col3 = st.columns(3)
                col1.metric("Overall NL", min_nl)
                col2.metric("Min NL", min_nl)
                col3.metric("Max NL", int(max(nl_per_bit)))

                # Tabel Rincian per Output Bit
                st.markdown("##### Nonlinearity per output bit:")
                nl_data = []
                for i, val in enumerate(nl_per_bit):
                    nl_data.append({
                        "Output Bit": f"f_{i}",
                        "NL Value": int(val),
                        "Status": "✅" if val >= 112 else "⚠️"
                    })
                st.table(pd.DataFrame(nl_data))

            elif selected_test == "Strict Avalanche Criterion (SAC)":
                st.subheader("Strict Avalanche Criterion (SAC) Test")
                st.write("SAC measures if changing 1 input bit results in changing half of the output bits on average. This ensures high diffusion.")
                st.info("Ideal value: 0.5")

                avg_sac, sac_per_bit, sac_matrix = results['sac']
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Overall SAC", f"{avg_sac:.5f}")
                col2.metric("Min SAC", f"{min(sac_per_bit):.5f}")
                col3.metric("Max SAC", f"{max(sac_per_bit):.5f}")

                st.markdown("##### SAC per input bit flip:")
                sac_data = [{"Input Bit Flip": f"e_{i}", "SAC Value": f"{v:.5f}", "Status": "✅" if abs(v-0.5) < 0.05 else "⚠️"} 
                            for i, v in enumerate(sac_per_bit)]
                st.table(pd.DataFrame(sac_data))

            elif selected_test == "BIC-NL":
                st.subheader("BIC-NL Test")
                st.write("Bit Independence Criterion for Nonlinearity ensures that all pairs of output bits combined via XOR remain highly non-linear.")
                
                bic_nl = results['bic_nl']
                st.metric("Minimum BIC-NL", bic_nl)

                st.info("BIC-NL checks all 28 pairs of output bits (f_j ⊕ f_k).")

            elif selected_test == "BIC-SAC":
                st.subheader("BIC-SAC Test")
                st.write("Bit Independence Criterion for SAC ensures that output bits change independently of each other when an input bit is flipped.")
                
                bic_sac = results['bic_sac']
                # bic_sac may be stored as tuple or float depending on earlier code; normalize
                if isinstance(bic_sac, tuple) or isinstance(bic_sac, list):
                    bic_sac_val = float(bic_sac[0])
                else:
                    bic_sac_val = float(bic_sac)
                st.metric("Overall BIC-SAC", f"{bic_sac_val:.5f}")
                st.info("Ideal value: 0.5")

            elif selected_test == "Linear Approximation (LAP)":
                st.subheader("Linear Approximation Probability (LAP)")
                st.write("Measures the maximum probability of a linear relationship between input and output bits.")
                
                lap, max_lat = results['lap']
                col1, col2 = st.columns(2)
                col1.metric("LAP Value", f"{lap:.6f}")
                col2.metric("Max LAT Value", max_lat)
                st.info("Standard AES: 0.062500. Smaller is better.")

            elif selected_test == "Differential Approximation (DAP)":
                st.subheader("Differential Approximation Probability (DAP)")
                st.write("Measures the probability of an input difference resulting in a specific output difference.")
                
                dap, du = results['dap']
                col1, col2 = st.columns(2)
                col1.metric("DAP Value", f"{dap:.6f}")
                col2.metric("Differential Uniformity (DU)", du)
                st.info("Standard AES: 0.015625 (DU=4). Smaller is better.")

    # --- VISUALIZATIONS TAB (REPLACEMENT) ---
    with testing_tabs[2]:
        st.header("📈 Visualisasi Hasil Pengujian")
        
        if "test_results" not in st.session_state:
            st.info("💡 Jalankan pengujian di tab 'Quick Summary' terlebih dahulu untuk melihat visualisasi.")
        else:
            results = st.session_state.test_results
            
            # Pilihan Visualisasi
            viz_choice = st.selectbox(
                "Pilih tipe visualisasi:",
                ["Strength Radar Chart", "SAC Distribution Heatmap", "Comparison vs AES Standard"],
                key="sbox_viz_choice"
            )

            st.markdown("---")

            if viz_choice == "Strength Radar Chart":
                st.subheader("S-Box Security Profile (Normalized)")
                st.caption("Grafik ini membandingkan skor S-box Anda dengan nilai referensi ideal. Semakin luas area biru, semakin seimbang kekuatannya.")
                
                # Mengambil data
                nl_val = results['nonlinearity'][0]
                sac_val = results['sac'][0]
                lap_val = results['lap'][0]
                dap_val = results['dap'][0]
                bic_nl_val = results['bic_nl']

                # Normalisasi data (skala 0.0 - 1.0+)
                # - NL: Ideal 112 (untuk 8-bit)
                # - SAC: Ideal 0.5 (dihitung seberapa dekat, 1.0 = sempurna)
                # - LAP: Ideal <= 0.0625 (dibalik, makin kecil makin bagus)
                # - DAP: Ideal <= 0.015625 (dibalik)
                
                metrics_dict = {
                    'Nonlinearity': min(nl_val / 112, 1.2),
                    'SAC (Consistency)': max(0, 1 - abs(sac_val - 0.5) / 0.25), # 0.5 is ideal
                    'Linear Res. (1/LAP)': min(0.0625 / max(lap_val, 0.0001), 1.2),
                    'Diff. Res. (1/DAP)': min(0.015625 / max(dap_val, 0.0001), 1.2),
                    'BIC-NL': min(bic_nl_val / 112, 1.2)
                }
                
                values = list(metrics_dict.values())
                labels = list(metrics_dict.keys())

                # Membuat Radar Chart
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=values + [values[0]], # Menutup loop
                    theta=labels + [labels[0]],
                    fill='toself',
                    fillcolor='rgba(31, 119, 180, 0.4)',
                    line=dict(color='#1f77b4', width=2),
                    name='Current S-Box'
                ))

                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 1.2], ticksuffix="x Ideal"),
                        bgcolor='rgba(0,0,0,0)'
                    ),
                    showlegend=False,
                    height=500,
                    margin=dict(t=40, b=40, l=40, r=40)
                )
                st.plotly_chart(fig, use_container_width=True)

            elif viz_choice == "SAC Distribution Heatmap":
                st.subheader("Strict Avalanche Criterion (SAC) Matrix")
                st.write("Heatmap ini menunjukkan rata-rata perubahan bit output ketika bit input tertentu diubah.")

                # Mengambil data full 8x8 matrix dari results['sac']
                try:
                    avg_sac, per_input_avg, sac_matrix = results['sac']

                    # Ensure sac_matrix is a plain 2D array / DataFrame to avoid plotting issues
                    sac_arr = np.array(sac_matrix, dtype=float)
                    x_labels = [f'Output Bit {j}' for j in range(sac_arr.shape[1])]
                    y_labels = [f'Input Bit {i}' for i in range(sac_arr.shape[0])]

                    df_sac = pd.DataFrame(sac_arr, index=y_labels, columns=x_labels)

                    fig = px.imshow(
                        df_sac.values,
                        labels=dict(x="Output Bit", y="Input Bit", color="SAC Value"),
                        x=df_sac.columns.tolist(),
                        y=df_sac.index.tolist(),
                        color_continuous_scale='RdBu_r',
                        range_color=[0.4, 0.6],
                        text_auto='.3f'
                    )

                    fig.update_layout(
                        title="SAC Matrix (Input bit flipped → Output bit change probability)",
                        margin=dict(t=40, b=40, l=40, r=40),
                        height=600
                    )

                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error rendering SAC heatmap: {e}")
                
                st.info("Keterangan: Warna **Merah/Biru Tua** menandakan penyimpangan dari nilai ideal 0.5. Warna **Putih/Pucat** menandakan properti Avalanche yang baik.")

            elif viz_choice == "Comparison vs AES Standard":
                st.subheader("Current S-Box vs AES Standard (FIPS 197)")
                
                # Data perbandingan
                categories = ['Nonlinearity (NL)', 'SAC', 'BIC-NL']
                # Nilai AES (Rijndael)
                aes_values_main = [112, 0.5, 112] 
                current_values_main = [
                    results['nonlinearity'][0],
                    results['sac'][0],
                    results['bic_nl']
                ]
                
                # Chart 1: Metrics Besar (NL, BIC)
                fig1 = go.Figure(data=[
                    go.Bar(name='AES Standard', x=categories, y=aes_values_main, marker_color='#adb5bd'),
                    go.Bar(name='Current S-Box', x=categories, y=current_values_main, marker_color='#1f77b4')
                ])
                fig1.update_layout(title="General Metrics (Higher is Better)", barmode='group', height=400)
                st.plotly_chart(fig1, use_container_width=True)
                
                st.markdown("---")
                
                # Chart 2: Metrics Probabilitas (Kecil) - LAP & DAP
                cat_prob = ['LAP (Linear Prob)', 'DAP (Differential Prob)']
                aes_prob = [0.0625, 0.015625] # 2^-4, 2^-6
                cur_prob = [results['lap'][0], results['dap'][0]]
                
                fig2 = go.Figure(data=[
                    go.Bar(name='AES Standard', x=cat_prob, y=aes_prob, marker_color='#adb5bd'),
                    go.Bar(name='Current S-Box', x=cat_prob, y=cur_prob, marker_color='#d62728')
                ])
                fig2.update_layout(
                    title="Vulnerability Probabilities (Lower is Better)", 
                    barmode='group', 
                    height=400,
                    yaxis_type="log", # Log scale agar DAP terlihat
                    yaxis_title="Probability (Log Scale)"
                )
                st.plotly_chart(fig2, use_container_width=True)
                st.caption("Catatan: Grafik probabilitas menggunakan skala Logaritmik.")
    
    # Interpretasi Tab
    with testing_tabs[3]:
        st.subheader("Interpretasi Hasil Pengujian")
        
        if "test_results" not in st.session_state:
            st.info("💡 Jalankan tests terlebih dahulu di tab 'Quick Summary'")
        else:
            results = st.session_state.test_results
            nl, _, _ = results['nonlinearity']
            lap, _ = results['lap']
            dap, du = results['dap']
            avg_sac = results['sac'][0]
            avg_ad, _ = results['avalanche']
            bic_sac = results['bic_sac']
            avg_to, _ = results['transparency_order']
            ci = results['correlation_immunity']
            
            st.markdown(f"""
## Analisis Komprehensif S-Box

### 1️⃣ Properti Nonlinearity (NL = {nl})
**Definisi:** Mengukur jarak minimum dari output ke fungsi linear Boolean.
- **Nilai baik:** NL ≥ 100 (untuk 8-bit S-box)
- **Interpretasi hasil:** NL = {nl} mengindikasikan {'SANGAT BAIK' if nl >= 100 else 'CUKUP' if nl >= 80 else 'LEMAH'} resistensi terhadap linear cryptanalysis.
- **Implikasi:** S-box {'TAHAN' if nl >= 100 else 'RENTAN'} terhadap serangan linear approximation.

### 2️⃣ Linear Approximation Probability (LAP = {lap:.6f})
**Definisi:** Probabilitas maksimum dari linear approximation untuk setiap input/output mask.
- **Nilai baik:** LAP ≤ 0.0625 (1/16)
- **Interpretasi hasil:** LAP = {lap:.6f} menunjukkan {'SANGAT BAIK' if lap <= 0.0625 else 'CUKUP' if lap <= 0.125 else 'LEMAH'} keseimbangan linear.
- **Implikasi:** Resistensi {'KUAT' if lap <= 0.0625 else 'SEDANG' if lap <= 0.125 else 'LEMAH'} terhadap linear cryptanalysis.

### 3️⃣ Strict Avalanche Criterion (SAC = {avg_sac:.6f})
**Definisi:** Proporsi bit output yang berubah ketika satu bit input di-flip.
- **Nilai ideal:** SAC = 0.5 (setengah bit output berubah)
- **Interpretasi hasil:** SAC = {avg_sac:.6f} menunjukkan {'IDEAL' if 0.45 <= avg_sac <= 0.55 else 'BAIK' if 0.4 <= avg_sac <= 0.6 else 'KURANG'}' avalanche effect.
- **Implikasi:** Perubahan input {'TERSEBAR MERATA' if 0.45 <= avg_sac <= 0.55 else 'CUKUP MERATA' if 0.4 <= avg_sac <= 0.6 else 'TIDAK MERATA'} ke output.

### 4️⃣ Differential Uniformity (DU = {du})
**Definisi:** Maksimum jumlah pasangan (x, x⊕Δx) dengan output difference Δy yang sama.
- **Nilai baik:** DU ≤ 4 (optimal untuk 8-bit)
- **Interpretasi hasil:** DU = {du} adalah {'OPTIMAL' if du <= 4 else 'BAIK' if du <= 6 else 'LEMAH'}.
- **Implikasi:** Resistensi {'KUAT' if du <= 4 else 'SEDANG' if du <= 6 else 'LEMAH'} terhadap differential cryptanalysis.

### 5️⃣ Differential Approximation Probability (DAP = {dap:.6f})
**Definisi:** Probabilitas maksimum dari differential approximation.
- **Nilai baik:** DAP ≤ 0.02
- **Interpretasi hasil:** DAP = {dap:.6f} menunjukkan {'SANGAT BAIK' if dap <= 0.02 else 'BAIK' if dap <= 0.04 else 'LEMAH'} resistensi diferensial.
- **Implikasi:** Keseimbangan differential {'SANGAT BAIK' if dap <= 0.02 else 'BAIK' if dap <= 0.04 else 'LEMAH'}.

### 6️⃣ Keterkaitan Antar Metrik

**Nonlinearity ↔ LAP:**
- NL yang tinggi mengurangi LAP
- Keduanya mengukur resistensi linear dari sudut berbeda
- Hasil: Semakin tinggi NL, semakin baik LAP

**DU ↔ DAP:**
- DU langsung menentukan DAP (DAP = DU/256)
- DU optimal ≤ 4 menghasilkan DAP ≤ 0.015625
- Hasil: DU = {du} → DAP = {dap:.6f}

**SAC ↔ Avalanche Degree:**
- SAC ideal = 0.5 ketika AD ≈ 4 (setengah dari 8 bit)
- Nilai AD = {avg_ad:.6f} pada SAC = {avg_sac:.6f} menunjukkan {'KONSISTENSI BAIK' if 3.5 <= avg_ad <= 4.5 else 'KETIDAKSESUAIAN'}

**BIC-SAC ↔ Output Independence:**
- BIC-SAC mengukur independensi bit output
- Nilai BIC-SAC = {bic_sac:.6f} menunjukkan {'INDEPENDENSI BAIK' if bic_sac >= 0.4 else 'KORELASI ANTAR BIT'}

### 7️⃣ Kekuatan S-Box

✅ **Kekuatan:**
{f'- NL = {nl} ≥ 100 (Nonlinearity baik)' if nl >= 100 else f'- NL = {nl} (Nonlinearity cukup)'}
{f'- LAP = {lap:.6f} ≤ 0.0625 (Linear resistance baik)' if lap <= 0.0625 else f'- LAP = {lap:.6f} (Linear resistance cukup)'}
{f'- DU = {du} ≤ 4 (Differential uniformity optimal)' if du <= 4 else f'- DU = {du} (Differential uniformity cukup)'}
{f'- DAP = {dap:.6f} ≤ 0.02 (Differential probability baik)' if dap <= 0.02 else f'- DAP = {dap:.6f} (Differential probability cukup)'}

### 8️⃣ Kelemahan S-Box

❌ **Kelemahan:**
{f'- SAC = {avg_sac:.6f} (Avalanche effect tidak ideal)' if not (0.45 <= avg_sac <= 0.55) else ''}
{f'- TO = {avg_to:.6f} (Side-channel resistance sedang)' if avg_to >= 0.2 else ''}
{f'- CI = {ci} (Tidak ada correlation immunity)' if ci == 0 else ''}

### 9️⃣ Rekomendasi Penggunaan

S-Box ini {'DIREKOMENDASIKAN' if nl >= 100 and lap <= 0.0625 and du <= 4 else 'CUKUP DIREKOMENDASIKAN' if nl >= 80 and lap <= 0.125 and du <= 6 else 'TIDAK DIREKOMENDASIKAN'} untuk:
- **Enkripsi blok:** {'✅ YA' if nl >= 100 and du <= 4 else '⚠️ DENGAN HATI-HATI' if nl >= 80 else '❌ TIDAK'}
- **Resistensi linear:** {'✅ KUAT' if lap <= 0.0625 else '⚠️ SEDANG' if lap <= 0.125 else '❌ LEMAH'}
- **Resistensi diferensial:** {'✅ KUAT' if du <= 4 else '⚠️ SEDANG' if du <= 6 else '❌ LEMAH'}
- **Side-channel:** {'✅ BAIK' if avg_to <= 0.2 else '⚠️ SEDANG' if avg_to <= 0.3 else '❌ LEMAH'}

### 🔟 Kesimpulan

S-Box ini memiliki profil kriptografi {'SANGAT BAIK' if nl >= 100 and lap <= 0.0625 and du <= 4 else 'BAIK' if nl >= 80 and lap <= 0.125 and du <= 6 else 'SEDANG' if nl >= 60 else 'LEMAH'} dengan:
- Resistensi linear: {'KUAT' if lap <= 0.0625 else 'SEDANG' if lap <= 0.125 else 'LEMAH'}
- Resistensi diferensial: {'KUAT' if du <= 4 else 'SEDANG' if du <= 6 else 'LEMAH'}
- Avalanche property: {'IDEAL' if 0.45 <= avg_sac <= 0.55 else 'BAIK' if 0.4 <= avg_sac <= 0.6 else 'KURANG'}
- Side-channel resistance: {'BAIK' if avg_to <= 0.2 else 'SEDANG' if avg_to <= 0.3 else 'LEMAH'}

Untuk aplikasi kriptografi production, pastikan semua metrik memenuhi standar industri.
            """)
    
    # =========================
    # COMPARE Tab (baru)
    # =========================
    with testing_tabs[4]:
        st.subheader("Compare Test Results")
        st.markdown("Bandingkan hasil testing sebelumnya dengan hasil testing saat ini. Penentu kekuatan: Final S (semakin kecil lebih kuat).")

        # Past tests list
        saved = st.session_state.get("saved_tests", [])
        current = st.session_state.get("test_results", None)

        col_left, col_right = st.columns([1,2])

        with col_left:
            st.markdown("### Saved Tests")
            if not saved:
                st.info("Belum ada hasil testing tersimpan. Jalankan 'Run Comprehensive Tests' untuk menyimpan hasil.")
            else:
                df_list = []
                for idx, item in enumerate(saved):
                    df_list.append({
                        "Index": idx,
                        "Decimal": item.get("decimal"),
                        "S": item.get("s_value"),
                        "NL": item.get("nl"),
                        "DU": item.get("du"),
                        "Timestamp": item.get("timestamp")
                    })
                df_saved = pd.DataFrame(df_list)
                st.dataframe(df_saved[['Index','Decimal','S','NL','DU','Timestamp']], use_container_width=True)
                
                sel_idx = st.number_input("Select saved test index to compare", min_value=0, max_value=len(saved)-1, value=0, step=1)
        
        with col_right:
            st.markdown("### Current Test")
            if current is None:
                st.info("Belum ada hasil testing saat ini. Jalankan 'Run Comprehensive Tests' dahulu.")
            else:
                # show current summary
                cur = current
                cur_s = cur.get('s_value') if isinstance(cur, dict) else None
                if cur_s is None:
                    cur_s = st.session_state.get('test_results', {}).get('s_value')
                # display current metrics (allow missing values)
                try:
                    st.metric("Final S (current)", f"{cur_s:.6f}" if cur_s is not None else "N/A")
                except Exception:
                    st.metric("Final S (current)", "N/A")
                nl_cur = cur.get('nonlinearity')[0] if cur.get('nonlinearity') else None
                du_cur = cur.get('dap')[1] if cur.get('dap') else None
                st.write(f"NL: {nl_cur}, DU: {du_cur}")
                
                # If a saved selection exists, show comparison
                if saved:
                    other = saved[int(sel_idx)]
                    st.markdown("### Selected Saved Test")
                    st.write(f"Decimal Value: {other.get('decimal')}")
                    st.metric("Final S (saved)", f"{other.get('s_value'):.6f}")
                    st.write(f"NL: {other.get('nl')}, DU: {other.get('du')}, Timestamp: {other.get('timestamp')}")
                    
                    # Comparison logic: lower S wins
                    cur_s_val = float(cur_s) if (cur_s is not None) else float('inf')
                    saved_s_val = float(other.get('s_value', float('inf')))
                    
                    if cur_s_val < saved_s_val:
                        st.success(f"Current S-Box is STRONGER (S current = {cur_s_val:.6f} < S saved = {saved_s_val:.6f})")
                    elif cur_s_val > saved_s_val:
                        st.warning(f"Saved S-Box is STRONGER (S saved = {saved_s_val:.6f} < S current = {cur_s_val:.6f})")
                    else:
                        st.info(f"Both S-Boxes have equal Final S = {cur_s_val:.6f}")
                    
                    # Show simple diff table for key metrics
                    comp_rows = [
                        {"Metric":"Final S","Current":cur_s_val,"Saved":saved_s_val},
                        {"Metric":"NL","Current":nl_cur,"Saved":other.get('nl')},
                        {"Metric":"DU","Current":du_cur,"Saved":other.get('du')},
                        {"Metric":"LAP","Current":cur.get('lap')[0] if cur.get('lap') else None,"Saved":other.get('lap')},
                        {"Metric":"SAC","Current":cur.get('sac')[0] if cur.get('sac') else None,"Saved":other.get('sac')}
                    ]
                    st.table(pd.DataFrame(comp_rows))
                    
                    # Option to mark saved as baseline
                    if st.button("Set selected saved as baseline", key="set_baseline"):
                        st.session_state['baseline_test'] = int(sel_idx)
                        st.success("Baseline updated.")
                        save_saved_tests_to_disk()
                    
                    baseline = st.session_state.get('baseline_test', None)
                    if baseline is not None and baseline < len(saved):
                        st.info(f"Current baseline: index {baseline} (Decimal {saved[baseline].get('decimal')})")
        # end compare tab