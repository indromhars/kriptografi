import streamlit as st
import numpy as np
import pandas as pd
from core.affine import (
    generate_affine_matrix,
    matrix_properties,
    index_to_first_row
)

def display_matrix_properties(first_row, props):
    """Display matrix properties in a detailed UI format"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Index Value", int("".join(map(str, first_row)), 2))
        st.metric("Rank", props["Rank"])
    
    with col2:
        st.metric("Ones Count", props["Ones Count"])
        st.metric("Zeros Count", props["Zeros Count"])
    
    with col3:
        binary_str = "".join(map(str, first_row))
        st.markdown(f"**Binary**\n\n`{binary_str}`")
    
    # Status highlight
    st.markdown("---")
    if props["Rank"] == 8:
        st.success("✅ **STATUS: VALID** - Matrix is invertible and can be used for encryption")
    else:
        st.error("❌ **STATUS: INVALID** - Matrix is not invertible and cannot be used for encryption")
    
    # Detailed properties section
    st.markdown("#### 📋 Detailed Properties")
    
    props_text = f"""
    - **Determinant**: {props.get("Determinant", "N/A")}
    - **Invertible**: {'Yes ✓' if props["Rank"] == 8 else 'No ✗'}
    - **Dimension**: 8×8
    """
    st.markdown(props_text)

def affine_menu():
    st.header("🔍 Affine Matrix Exploration")

    # Main explanation
    with st.expander("ℹ️ Tentang Affine Matrix Encryption", expanded=False):
        st.markdown("""
        ### Penjelasan Affine Matrix
        
        **Affine Matrix** adalah matriks yang digunakan dalam enkripsi untuk transformasi linier data. 
        Proses kerjanya adalah:
        
        1. **Input**: Binary string (8-bit) yang merepresentasikan baris pertama matriks
        2. **Ekspansi**: Baris pertama diperluas menjadi matriks 8×8 dengan operasi shift cyclic
        3. **Validasi**: Matriks divalidasi dengan menghitung rank (harus = 8 untuk valid)
        4. **Aplikasi**: Matriks valid dapat digunakan untuk enkripsi/dekripsi data
        
        ### Properti Penting
        - **Rank**: Dimensi ruang kolom, harus 8 untuk matriks invertible
        - **Determinant**: Nilai skalar yang menentukan invertibilitas
        - **Ones/Zeros Count**: Jumlah bit 1 dan 0 dalam baris pertama
        """)

    submenu = st.tabs([
        "Custom Input",
        "Example Matrices",
        "Index-based Access",
        "Range Explorer"
    ])

    # =========================
    # CUSTOM INPUT
    # =========================
    with submenu[0]:
        st.subheader("Custom Input")

        with st.expander("📚 Cara Kerja Custom Input", expanded=False):
            st.markdown("""
            ### Proses Perhitungan Custom Input
            
            #### 1. **Konversi Input ke Binary**
            - Jika menggunakan Binary String: langsung diparsing
            - Jika menggunakan Individual Bits: dikombinasikan dari 8 selectbox
            - Jika menggunakan Decimal: dikonversi ke 8-bit binary dengan formula `bin(dec)`
            
            Contoh: Decimal 143 → Binary 10001111
            
            #### 2. **Generasi Matriks dari Baris Pertama**
            ```
            Baris Pertama: [1, 0, 0, 0, 1, 1, 1, 1]
            
            Matriks 8×8 dibuat dengan shift cyclic:
            Row 0: [1, 0, 0, 0, 1, 1, 1, 1]
            Row 1: [1, 1, 0, 0, 0, 1, 1, 1]  ← shift left 1
            Row 2: [1, 1, 1, 0, 0, 0, 1, 1]  ← shift left 2
            ... dst sampai Row 7
            ```
            
            #### 3. **Kalkulasi Properti Matriks**
            - **Rank**: Jumlah baris/kolom independen (max 8)
            - **Ones Count**: Menghitung jumlah bit '1' dalam baris pertama
            - **Zeros Count**: Menghitung jumlah bit '0' dalam baris pertama
            - **Determinant**: Nilai yang menentukan apakah matriks invertible
            
            #### 4. **Validasi Status**
            - **VALID**: Rank = 8 (matriks invertible, dapat digunakan untuk enkripsi)
            - **INVALID**: Rank < 8 (matriks singular, tidak dapat digunakan)
            """)

        input_type = st.radio(
            "Input Method",
            ["Binary String", "Individual Bits", "Decimal Value"]
        )

        first_row = None
        valid = False

        if input_type == "Binary String":
            binary = st.text_input(
                "First Row (8-bit binary)",
                value="00000111"
            )
            if len(binary) != 8:
                st.warning("⚠️ Binary string harus tepat 8 bit")
            elif not all(c in '01' for c in binary):
                st.warning("⚠️ Binary string hanya boleh berisi 0 dan 1")
            else:
                first_row = np.array(list(map(int, binary)))
                valid = True

        elif input_type == "Individual Bits":
            cols = st.columns(8)
            bits = []
            for i, c in enumerate(cols):
                bits.append(c.selectbox(f"b{i}", [0, 1]))
            first_row = np.array(bits)
            valid = True

        else:
            dec = st.number_input(
                "Decimal Value (0-255)",
                min_value=0,
                max_value=255,
                value=7
            )
            first_row = np.array(list(map(int, f"{dec:08b}")))
            valid = True

        if valid and first_row is not None:
            matrix = generate_affine_matrix(first_row)
            props = matrix_properties(matrix)

            st.markdown("### Generated Matrix")
            st.dataframe(pd.DataFrame(matrix))

            st.markdown("### Matrix Properties")
            display_matrix_properties(first_row, props)
            
            # Save button
        if st.button("💾 Save for S-Box", key="save_custom"):
                    st.session_state.saved_first_row = first_row
                    # Clear previous test results when user saves a new first_row
                    if "test_results" in st.session_state:
                        del st.session_state["test_results"]
                    st.success("✅ Matrix berhasil disimpan! Buka menu S-Box untuk menggunakannya.")
        
        elif input_type == "Binary String" and first_row is None:
            st.info("ℹ️ Masukkan binary string yang valid untuk melanjutkan")

    # =========================
    # EXAMPLE MATRICES
    # =========================
    with submenu[1]:
        st.subheader("Example Matrices")

        with st.expander("📚 Cara Kerja Example Matrices", expanded=False):
            st.markdown("""
            ### Proses Perhitungan Example Matrices
            
            #### Matriks Predefined
            Menu ini menyediakan contoh matriks yang sudah diketahui hasilnya:
            
            1. **AES Default (Index 143)**
               - Decimal: 143 → Binary: 10001111
               - Digunakan sebagai referensi standar AES
               
            2. **All Zeros (Index 0)**
               - Decimal: 0 → Binary: 00000000
               - Seluruh baris pertama adalah 0
               - Status: INVALID (Rank < 8)
               
            3. **All Ones (Index 255)**
               - Decimal: 255 → Binary: 11111111
               - Seluruh baris pertama adalah 1
               - Status: Tergantung sifat matriks yang dihasilkan
            
            #### Proses yang Sama
            Setelah memilih contoh, sistem:
            1. Mengkonversi index ke binary (8-bit)
            2. Menghasilkan matriks 8×8 dengan cyclic shift
            3. Menghitung properti dan validasi
            4. Menampilkan hasil dalam format tabel dan metrics
            """)

        examples = {
            "AES Default": 0x8F,
            "All Zeros": 0,
            "All Ones": 255
        }
        choice = st.radio("Choose Example", examples.keys())
        first_row = index_to_first_row(examples[choice])
        matrix = generate_affine_matrix(first_row)
        
        st.markdown("#### Matrix")
        st.dataframe(pd.DataFrame(matrix))
        
        props = matrix_properties(matrix)
        st.markdown("#### Properties")
        display_matrix_properties(first_row, props)
        
        # Save button
        if st.button("💾 Save for S-Box", key="save_example"):
            st.session_state.saved_first_row = first_row
            if "test_results" in st.session_state:
                del st.session_state["test_results"]
            st.success("✅ Matrix berhasil disimpan! Buka menu S-Box untuk menggunakannya.")

    # =========================
    # INDEX-BASED ACCESS
    # =========================
    with submenu[2]:
        st.subheader("Index-based Access")

        with st.expander("📚 Cara Kerja Index-based Access", expanded=False):
            st.markdown("""
            ### Proses Perhitungan Index-based Access
            
            #### Konsep Index
            Setiap kombinasi 8-bit dapat direpresentasikan sebagai angka desimal (0-255):
            - Index 0 = Binary 00000000
            - Index 1 = Binary 00000001
            - Index 143 = Binary 10001111
            - Index 255 = Binary 11111111
            
            #### Proses Konversi Index → Matriks
            ```
            1. Input Index (0-255)
            2. Konversi ke 8-bit Binary
               Index 143 → 10001111
            3. Gunakan binary sebagai baris pertama
            4. Ekspansi menjadi matriks 8×8 dengan cyclic shift
            5. Hitung properti dan validasi
            ```
            
            #### Kegunaan
            - Mengakses matriks tertentu langsung melalui indeksnya
            - Sistematis menjelajahi 256 kemungkinan matriks affine
            - Mempelajari pola validitas di range tertentu
            """)

        idx = st.number_input(
            "Affine Matrix Index",
            min_value=0,
            max_value=2**8 - 1,
            value=7,
            key="affine_idx"
        )
        first_row = index_to_first_row(idx)
        matrix = generate_affine_matrix(first_row)
        
        st.markdown("#### Matrix")
        st.dataframe(pd.DataFrame(matrix))
        
        props = matrix_properties(matrix)
        st.markdown("#### Properties")
        display_matrix_properties(first_row, props)
        
        # Clear previous test results when user changes the index (without needing to press Save)
        prev_idx = st.session_state.get("_last_affine_idx", None)
        if prev_idx is None:
            st.session_state["_last_affine_idx"] = idx
        elif prev_idx != idx:
            if "test_results" in st.session_state:
                del st.session_state["test_results"]
            st.session_state["_last_affine_idx"] = idx

        # Save button
        if st.button("💾 Save for S-Box", key="save_index"):
            st.session_state.saved_first_row = first_row
            if "test_results" in st.session_state:
                del st.session_state["test_results"]
            st.success("✅ Matrix berhasil disimpan! Buka menu S-Box untuk menggunakannya.")

    # =========================
    # RANGE EXPLORER
    # =========================
    with submenu[3]:
        st.subheader("Range Explorer")

        with st.expander("📚 Cara Kerja Range Explorer", expanded=False):
            st.markdown("""
            ### Proses Perhitungan Range Explorer
            
            #### Fungsi
            Menganalisis range indeks secara batch untuk melihat pola validitas matriks.
            
            #### Proses Batch
            ```
            1. Input Start Index dan End Index
            2. Loop untuk setiap index dalam range:
               a. Konversi index ke 8-bit binary
               b. Generate matriks 8×8
               c. Hitung properties
               d. Simpan hasil dalam tabel
            3. Tampilkan hasil lengkap dalam dataframe
            ```
            
            #### Interpretasi Hasil
            - **Index**: Nilai desimal (0-255)
            - **First Row**: Representasi binary dari index
            - **Rank**: 8 = Valid, <8 = Invalid
            - **Ones**: Jumlah bit '1' dalam first row
            - **Zeros**: Jumlah bit '0' dalam first row
            
            #### Contoh Analisis
            Jika range 0-10:
            - Index 0 (00000000): Ones=0, Zeros=8 → INVALID
            - Index 7 (00000111): Ones=3, Zeros=5 → tergantung rank
            - Index 255 (11111111): Ones=8, Zeros=0 → tergantung rank
            """)

        start = st.number_input("Start Index", 0, 255, 0)
        end = st.number_input("End Index", 0, 255, 10)

        if start > end:
            st.error("❌ Start Index harus lebih kecil atau sama dengan End Index")
        else:
            rows = []
            for i in range(start, end + 1):
                row = index_to_first_row(i)
                mat = generate_affine_matrix(row)
                props = matrix_properties(mat)
                rows.append({
                    "Index": i,
                    "First Row": "".join(map(str, row)),
                    "Rank": props["Rank"],
                    "Ones": props["Ones Count"],
                    "Zeros": props["Zeros Count"]
                })

            st.dataframe(pd.DataFrame(rows))