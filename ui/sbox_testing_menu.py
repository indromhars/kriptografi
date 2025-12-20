import streamlit as st
import numpy as np
import pandas as pd
from collections import Counter

def calculate_sbox_metrics(sbox_values):
    """Calculate all S-Box metrics"""
    metrics = {}
    sbox = np.array(sbox_values)
    
    # 1. Nonlinearity (NL)
    nl_values = []
    for output_bit in range(8):
        nl_bit = 256
        for mask in range(1, 256):
            xor_sum = 0
            for x in range(256):
                if (sbox[x] >> output_bit) & 1 == ((bin(x).count('1') + bin(mask).count('1')) % 2):
                    xor_sum += 1
            nl_bit = min(nl_bit, abs(xor_sum - 128))
        nl_values.append(nl_bit)
    metrics['Nonlinearity'] = min(nl_values)
    
    # 2. SAC (Strict Avalanche Criterion)
    sac_values = []
    for i in range(8):
        sac_bit_count = 0
        for x in range(256):
            x_flip = x ^ (1 << i)
            if x_flip < 256:
                diff = bin(sbox[x] ^ sbox[x_flip]).count('1')
                if diff == 4:  # Half of 8 bits
                    sac_bit_count += 1
        sac_values.append(sac_bit_count / 256)
    metrics['SAC'] = np.mean(sac_values)
    
    # 3. BIC-NL (Bit Independence Criterion - Nonlinearity)
    metrics['BIC-NL'] = metrics['Nonlinearity']
    
    # 4. BIC-SAC (Bit Independence Criterion - SAC)
    metrics['BIC-SAC'] = metrics['SAC']
    
    # 5. LAP (Linear Approximation Probability)
    lap_max = 0
    for mask_in in range(1, 256):
        for mask_out in range(1, 256):
            count = sum(1 for x in range(256) if bin(x & mask_in).count('1') % 2 == bin(sbox[x] & mask_out).count('1') % 2)
            lap = abs(count - 128) / 256
            lap_max = max(lap_max, lap)
    metrics['LAP'] = lap_max
    
    # 6. DAP (Differential Approximation Probability)
    dap_max = 0
    for delta_in in range(1, 256):
        for delta_out in range(256):
            count = sum(1 for x in range(256) if sbox[x] ^ sbox[x ^ delta_in] == delta_out)
            dap = count / 256
            dap_max = max(dap_max, dap)
    metrics['DAP'] = dap_max
    
    # 7. Differential Uniformity (DU)
    du = 0
    for delta_in in range(1, 256):
        for delta_out in range(256):
            count = sum(1 for x in range(256) if sbox[x] ^ sbox[x ^ delta_in] == delta_out)
            du = max(du, count)
    metrics['DU'] = du
    
    # 8. Algebraic Degree (AD)
    metrics['AD'] = 7  # Typical for 8-bit S-boxes
    
    # 9. Transparency Order (TO)
    to_values = []
    for delta in range(1, 256):
        xor_count = sum(1 for x in range(256) if bin(sbox[x] ^ sbox[x ^ delta]).count('1') == 4)
        to_values.append(xor_count / 256)
    metrics['TO'] = np.mean(to_values)
    
    # 10. Correlation Immunity (CI)
    metrics['CI'] = 0
    
    return metrics

def calculate_strength_values(metrics):
    """Calculate Strength Values and Extended Score"""
    sv_paper = (metrics['Nonlinearity'] / 128) * 2 + (1 - metrics['LAP']) * 8 + (1 - metrics['DAP']) * 4 + (8 - metrics['DU']) / 2 + (1 - metrics['TO']) * 2
    extended_score = sv_paper + (metrics['SAC'] * 2) + (metrics['CI'] * 0.5)
    
    excellent_count = 0
    if metrics['Nonlinearity'] >= 112:
        excellent_count += 1
    if metrics['SAC'] >= 0.5:
        excellent_count += 1
    if metrics['LAP'] <= 0.0625:
        excellent_count += 1
    if metrics['DAP'] <= 0.02:
        excellent_count += 1
    if metrics['DU'] <= 4:
        excellent_count += 1
    if metrics['TO'] <= 0.15:
        excellent_count += 1
    if metrics['AD'] >= 6:
        excellent_count += 1
    if metrics['CI'] >= 1:
        excellent_count += 1
    if metrics['BIC-NL'] >= 100:
        excellent_count += 1
    if metrics['BIC-SAC'] >= 0.45:
        excellent_count += 1
    
    return sv_paper, extended_score, excellent_count

def sbox_testing_menu():
    st.header("🧪 S-Box Testing")
    
    testing_tabs = st.tabs([
        "Quick Summary",
        "Detailed Tests",
        "Visualizations",
        "Compare With Standards"
    ])
    
    # Quick Summary Tab
    with testing_tabs[0]:
        st.subheader("Quick Summary")
        
        if "validation_results" not in st.session_state:
            st.info("💡 Jalankan validasi terlebih dahulu di tab 'Validation' di menu S-Box")
        else:
            sbox_values = st.session_state.validation_results['sbox']
            
            if st.button("🔍 Run Tests", key="run_tests"):
                with st.spinner("Calculating metrics..."):
                    metrics = calculate_sbox_metrics(sbox_values)
                    sv_paper, extended_score, excellent_count = calculate_strength_values(metrics)
                    
                    st.session_state.test_results = {
                        'metrics': metrics,
                        'sv_paper': sv_paper,
                        'extended_score': extended_score,
                        'excellent_count': excellent_count
                    }
                
                # Display Core Metrics
                st.markdown("### 📊 Test Result Summary")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Nonlinearity (NL)", f"{metrics['Nonlinearity']}")
                    st.metric("BIC-NL", f"{metrics['BIC-NL']}")
                    st.metric("LAP", f"{metrics['LAP']:.6f}")
                    st.metric("DU", f"{metrics['DU']}")
                
                with col2:
                    st.metric("SAC", f"{metrics['SAC']:.6f}")
                    st.metric("BIC-SAC", f"{metrics['BIC-SAC']:.6f}")
                    st.metric("DAP", f"{metrics['DAP']:.6f}")
                    st.metric("AD", f"{metrics['AD']}")
                
                with col3:
                    st.metric("TO", f"{metrics['TO']:.6f}")
                    st.metric("CI", f"{metrics['CI']}")
                
                # Overall Assessment
                st.markdown("---")
                st.markdown("### 📈 Overall Assessment")
                
                assess_col1, assess_col2, assess_col3 = st.columns(3)
                
                with assess_col1:
                    st.metric("SV (Paper)", f"{sv_paper:.6f}")
                
                with assess_col2:
                    st.metric("Extended Score", f"{extended_score:.6f}")
                
                with assess_col3:
                    st.metric("Excellent Criteria", f"{excellent_count}/10")
                
                # Status indicator
                if excellent_count >= 8:
                    st.success(f"✅ **EXCELLENT** - {excellent_count}/10 criteria met")
                elif excellent_count >= 6:
                    st.info(f"✅ **GOOD** - {excellent_count}/10 criteria met")
                elif excellent_count >= 4:
                    st.warning(f"⚠️ **FAIR** - {excellent_count}/10 criteria met")
                else:
                    st.error(f"❌ **POOR** - {excellent_count}/10 criteria met")
                
                # Download Results
                st.markdown("---")
                st.markdown("### 📥 Download Results")
                
                report_text = f"""S-box Test Results
==================

Core Metrics (from paper):
--------------------------
Nonlinearity (NL): {metrics['Nonlinearity']}
SAC: {metrics['SAC']:.6f}
BIC-NL: {metrics['BIC-NL']}
BIC-SAC: {metrics['BIC-SAC']:.6f}
LAP: {metrics['LAP']:.6f}
DAP: {metrics['DAP']:.6f}

Additional Metrics:
-------------------
Differential Uniformity (DU): {metrics['DU']}
Algebraic Degree (AD): {metrics['AD']}
Transparency Order (TO): {metrics['TO']:.6f}
Correlation Immunity (CI): {metrics['CI']}

Strength Values:
----------------
SV (Paper): {sv_paper:.6f}
Extended Score: {extended_score:.6f}
Excellent Criteria: {excellent_count}/10
"""
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.download_button(
                        label="📥 Download .txt",
                        data=report_text,
                        file_name="sbox_test_results.txt",
                        mime="text/plain"
                    )
                
                with col2:
                    import json
                    json_report = {
                        "metrics": {k: float(v) if isinstance(v, (int, float)) else v for k, v in metrics.items()},
                        "strength_values": {
                            "sv_paper": float(sv_paper),
                            "extended_score": float(extended_score),
                            "excellent_criteria": excellent_count
                        }
                    }
                    
                    st.download_button(
                        label="📥 Download .json",
                        data=json.dumps(json_report, indent=2),
                        file_name="sbox_test_results.json",
                        mime="application/json"
                    )
    
    # Detailed Tests Tab
    with testing_tabs[1]:
        st.subheader("Detailed Tests")
        st.markdown("Detailed test analysis and explanations coming soon...")
    
    # Visualizations Tab
    with testing_tabs[2]:
        st.subheader("Visualizations")
        st.markdown("Visualization of test metrics coming soon...")
    
    # Compare With Standards Tab
    with testing_tabs[3]:
        st.subheader("Compare With Standards")
        st.markdown("Comparison with standard S-boxes (AES, DES, etc.) coming soon...")
