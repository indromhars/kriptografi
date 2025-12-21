import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from core.metrics import calculate_sbox_metrics

def sbox_testing_menu():
    st.header("🧪 S-Box Testing & Cryptographic Evaluation")
    
    if "validation_results" not in st.session_state:
        st.info("💡 Jalankan validasi terlebih dahulu di menu S-Box Construction")
        return

    sbox_values = st.session_state.validation_results['sbox']
    
    # Menambahkan tab Comparison agar tidak kosong
    tabs = st.tabs(["📊 Quick Summary", "🔍 Detailed Analysis", "📈 Visualizations", "⚖️ Comparison"])

    # --- TAB 0: QUICK SUMMARY ---
    with tabs[0]:
        if st.button("🚀 Run Full Analysis"):
            with st.spinner("Calculating complex metrics... (LAP takes time)"):
                res = calculate_sbox_metrics(sbox_values)
                st.session_state.test_results = res
            st.rerun()

        if "test_results" in st.session_state:
            res = st.session_state.test_results
            c1, c2, c3 = st.columns(3)
            c1.metric("Nonlinearity (NL)", res['nl_min'])
            c1.metric("BIC-NL", res['bic_nl'])
            c2.metric("SAC (Avg)", f"{res['sac_avg']:.5f}")
            c2.metric("BIC-SAC", f"{res['bic_sac']:.5f}")
            c3.metric("LAP", f"{res['lap']:.5f}")
            c3.metric("DAP", f"{res['dap']:.5f}")
            
            st.divider()
            st.metric("Final S Score (Lower is better)", f"{res['final_s']:.6f}")

    # --- TAB 1: DETAILED ANALYSIS ---
    with tabs[1]:
        if "test_results" in st.session_state:
            res = st.session_state.test_results
            selected = st.selectbox("Select test to view details:", 
                ["Nonlinearity (NL)", "Strict Avalanche Criterion (SAC)", "BIC-NL/SAC", "Linear/Differential"])
            
            if selected == "Nonlinearity (NL)":
                st.subheader("Nonlinearity Test")
                st.info("Ideal value: 112 for 8-bit S-boxes")
                nl_df = pd.DataFrame({
                    "Output Bit": [f"f_{i}" for i in range(8)],
                    "NL Value": res['nl_per_bit'],
                    "Status": ["✅" if x >= 112 else "⚠️" for x in res['nl_per_bit']]
                })
                st.table(nl_df)

            elif selected == "Strict Avalanche Criterion (SAC)":
                st.subheader("SAC Detail per Input Bit")
                st.write("Target ideal: 0.5000")
                sac_df = pd.DataFrame({
                    "Input Bit Flip": [f"e_{i}" for i in range(8)],
                    "SAC Value": [f"{x:.5f}" for x in res['sac_per_bit']],
                    "Status": ["✅" if abs(x-0.5) < 0.05 else "⚠️" for x in res['sac_per_bit']]
                })
                st.table(sac_df)
            
            # ... bisa tambahkan detail untuk BIC atau LAP di sini ...

    # --- TAB 2: VISUALIZATIONS ---
    with tabs[2]:
        if "test_results" in st.session_state:
            res = st.session_state.test_results
            v_choice = st.selectbox("Select visualization:", ["Overview Radar Chart", "SAC Heatmap (8x8)"])
            
            if v_choice == "Overview Radar Chart":
                metrics = {
                    'NL': res['nl_min'] / 112,
                    'SAC': 1 - abs(res['sac_avg'] - 0.5) / 0.5,
                    'LAP (Inv)': 0.0625 / max(res['lap'], 0.001),
                    'DAP (Inv)': 0.0156 / max(res['dap'], 0.001),
                    'BIC-NL': res['bic_nl'] / 112
                }
                fig = go.Figure(go.Scatterpolar(r=list(metrics.values()) + [list(metrics.values())[0]], 
                                              theta=list(metrics.keys()) + [list(metrics.keys())[0]], fill='toself'))
                fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1.2])), title="Normalized Strength (1.0 = Ideal)")
                st.plotly_chart(fig)

            elif v_choice == "SAC Heatmap (8x8)":
                st.subheader("Full 8x8 SAC Matrix")
                # Ini yang memastikan heatmap tidak gepeng (8 baris x 8 kolom)
                fig = px.imshow(res['sac_matrix'], 
                                labels=dict(x="Output Bit Index", y="Input Bit Flip", color="Probability"),
                                x=[f'Out {i}' for i in range(8)], 
                                y=[f'In {i}' for i in range(8)],
                                color_continuous_scale='RdBu_r', 
                                range_color=[0.4, 0.6], 
                                text_auto='.3f')
                fig.update_layout(width=700, height=700)
                st.plotly_chart(fig)

    # --- TAB 3: COMPARISON (Sesuai Data Jurnal Halaman 17) ---
    with tabs[3]:
        st.subheader("Comparison with AES Standard")
        if "test_results" in st.session_state:
            res = st.session_state.test_results
            
            # Data dari Tabel 19 di Jurnal
            comparison_data = {
                "Metric": ["NL", "SAC", "BIC-NL", "BIC-SAC", "LAP", "DAP"],
                "AES Standard [1]": [112, 0.50488, 112, 0.50460, 0.0625, 0.01563],
                "Current S-Box": [
                    res['nl_min'], 
                    round(res['sac_avg'], 5), 
                    res['bic_nl'], 
                    round(res['bic_sac'], 5), 
                    round(res['lap'], 4), 
                    round(res['dap'], 5)
                ]
            }
            df_comp = pd.DataFrame(comparison_data)
            st.table(df_comp)

            # Bar Chart Comparison
            st.markdown("### Visualization: Current vs AES")
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(x=df_comp["Metric"], y=df_comp["AES Standard [1]"], name='AES Standard'))
            fig_comp.add_trace(go.Bar(x=df_comp["Metric"], y=df_comp["Current S-Box"], name='Current S-Box'))
            fig_comp.update_layout(barmode='group', height=400)
            st.plotly_chart(fig_comp, use_container_width=True)

            # Kesimpulan Otomatis
            if res['sac_avg'] < 0.50488 and res['nl_min'] >= 112:
                st.success("✨ S-Box saat ini secara kriptografi LEBIH KUAT atau setara dibanding AES standar berdasarkan nilai SAC.")
            else:
                st.warning("⚠️ S-Box saat ini memiliki performa di bawah atau setara dengan AES standar.")