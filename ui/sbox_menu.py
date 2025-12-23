import streamlit as st
import numpy as np
import pandas as pd
from core.affine import (
    generate_affine_matrix,
    matrix_properties,
    index_to_first_row
)

# Multiplicative Inverse Table (GF(2^8))
MULTIPLICATIVE_INVERSE = [
    0, 1, 141, 246, 203, 82, 123, 209, 232, 79, 41, 192, 176, 225, 229, 199,
    116, 180, 170, 75, 153, 43, 96, 95, 88, 63, 253, 204, 255, 64, 238, 178,
    58, 110, 90, 241, 85, 77, 168, 201, 193, 10, 152, 21, 48, 68, 162, 194,
    44, 69, 146, 108, 243, 57, 102, 66, 242, 53, 32, 111, 119, 187, 89, 25,
    29, 254, 55, 103, 45, 49, 245, 105, 167, 100, 171, 19, 84, 37, 233, 9,
    237, 92, 5, 202, 76, 36, 135, 191, 24, 62, 34, 240, 81, 236, 97, 23,
    22, 94, 175, 211, 73, 166, 54, 67, 244, 71, 145, 223, 51, 147, 33, 59,
    121, 183, 151, 133, 16, 181, 186, 60, 182, 112, 208, 6, 161, 250, 129, 130,
    131, 126, 127, 128, 150, 115, 190, 86, 155, 158, 149, 217, 247, 2, 185, 164,
    222, 106, 50, 109, 216, 138, 132, 114, 42, 20, 159, 136, 249, 220, 137, 154,
    251, 124, 46, 195, 143, 184, 101, 72, 38, 200, 18, 74, 206, 231, 210, 98,
    12, 224, 31, 239, 17, 117, 120, 113, 165, 142, 118, 61, 189, 188, 134, 87,
    11, 40, 47, 163, 218, 212, 228, 15, 169, 39, 83, 4, 27, 252, 172, 230,
    122, 7, 174, 99, 197, 219, 226, 234, 148, 139, 196, 213, 157, 248, 144, 107,
    177, 13, 214, 235, 198, 14, 207, 173, 8, 78, 215, 227, 93, 80, 30, 179,
    91, 35, 56, 52, 104, 70, 3, 140, 221, 156, 125, 160, 205, 26, 65, 28
]

def apply_affine_transformation(input_byte, matrix, constant, show_steps=False):
    """
    Apply affine transformation: 
    1. Lookup input in multiplicative inverse table
    2. Convert to binary vector (LSB = rightmost)
    3. Multiply with affine matrix (mod 2)
    4. XOR with constant
    5. Read result from bottom to top (LSB to MSB)
    """
    # Step 1: Lookup multiplicative inverse
    inverse_value = MULTIPLICATIVE_INVERSE[input_byte]
    
    # Step 2: Convert inverse to binary vector (LSB to MSB, read from right to left)
    inverse_binary = format(inverse_value, '08b')
    inverse_vec = np.array([int(b) for b in inverse_binary[::-1]])  # Reverse untuk bottom-to-top reading
    
    # Step 3: Matrix multiplication (mod 2)
    output_vec = np.dot(matrix, inverse_vec) % 2
    
    # Step 4: Convert to decimal
    matrix_result = int(''.join(map(str, output_vec[::-1])), 2)  # Reverse back untuk decimal
    
    # Step 5: XOR with constant
    final_output = matrix_result ^ constant
    
    if show_steps:
        return final_output, {
            'input': input_byte,
            'inverse': inverse_value,
            'inverse_binary': inverse_binary,
            'inverse_vec': inverse_vec,
            'output_vec': output_vec,
            'matrix_result': matrix_result,
            'constant': constant,
            'final_output': final_output
        }
    
    return final_output

def display_transformation_steps(input_byte, matrix, constant):
    """Display detailed steps of transformation"""
    output, steps = apply_affine_transformation(input_byte, matrix, constant, show_steps=True)
    
    st.markdown(f"### 📊 Detailed Transformation Steps for Input {input_byte} (0x{input_byte:02X})")
    
    # Step 1: Multiplicative Inverse Lookup
    st.markdown("#### Step 1: Multiplicative Inverse Lookup")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Input:** {input_byte} (0x{input_byte:02X})")
    with col2:
        st.write(f"**Inverse Value:** {steps['inverse']} (0x{steps['inverse']:02X})")
    
    st.markdown(f"**Binary of Inverse:** `{steps['inverse_binary']}`")
    st.info(f"💡 From multiplicative inverse table: MULTIPLICATIVE_INVERSE[{input_byte}] = {steps['inverse']}")
    
    # Step 2: Convert to Binary Vector
    st.markdown("#### Step 2: Convert to Binary Vector (Read from Bottom to Top)")
    st.write("Binary (right to left):", " ".join(steps['inverse_binary'][::-1]))
    st.write(f"Vector: {steps['inverse_vec']}")
    
    # Step 3: Matrix Multiplication
    st.markdown("#### Step 3: Affine Matrix × Input Vector (mod 2)")
    st.write("Matrix × Vector = Output Vector")
    st.write(f"Output Vector: {steps['output_vec']}")
    
    # Step 4: Convert to Decimal
    st.markdown("#### Step 4: Convert Output Vector to Decimal")
    output_binary = ''.join(map(str, steps['output_vec'][::-1]))
    st.write(f"Binary (read from bottom to top): `{output_binary}`")
    st.write(f"Decimal: {steps['matrix_result']}")
    
    # Step 5: XOR with Constant
    st.markdown("#### Step 5: XOR with Constant")
    const_binary = format(steps['constant'], '08b')
    result_binary = format(steps['final_output'], '08b')
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**Matrix Result:** {steps['matrix_result']}")
        st.write(f"Binary: `{format(steps['matrix_result'], '08b')}`")
    with col2:
        st.write(f"**XOR**")
    with col3:
        st.write(f"**Constant:** {steps['constant']}")
        st.write(f"Binary: `{const_binary}`")
    
    st.write(f"**Final Result:** {steps['final_output']} (0x{steps['final_output']:02X})")
    st.write(f"Binary: `{result_binary}`")

def display_affine_configuration(first_row, constant):
    """Display the affine configuration summary"""
    st.markdown("---")
    st.markdown("### ⚙️ Configuration Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **Matrix Used:**
        - First Row: `{''.join(map(str, first_row))}` (Index: {int(''.join(map(str, first_row)), 2)})
        - Dimension: 8×8
        """)
    
    with col2:
        st.info(f"""
        **Constant Used (C):**
        - Binary: `{format(constant, '08b')}`
        - Decimal: {constant}
        - Hex: 0x{constant:02X}
        """)

def sbox_menu():
    st.header("🔐 S-Box Construction")
    
    # Main explanation
    with st.expander("ℹ️ Tentang S-Box dan Affine Transformation", expanded=False):
        st.markdown("""
        ### Proses Affine Transformation untuk S-Box
        
        **Rumus Lengkap:**
        ```
        Output = (K × InverseLookup(Input)) ⊕ C (mod 2)
        ```
        
        Dimana:
        - **K**: Affine Matrix 8×8
        - **Input**: Nilai byte (0-255)
        - **InverseLookup**: Lookup di multiplicative inverse table
        - **⊕**: Operasi XOR (bitwise exclusive or)
        - **C**: Constant (AES standard atau custom)
        - **mod 2**: Semua operasi dalam binary (0 atau 1)
        
        #### Langkah-langkah Detail:
        1. **Lookup Multiplicative Inverse**: Cari inverse dari input di table
        2. **Konversi ke Binary**: Ubah inverse menjadi 8-bit binary, baca dari bawah ke atas
        3. **Matrix Multiplication**: Kalikan matrix K dengan binary vector (mod 2)
        4. **Konversi ke Desimal**: Ubah hasil vector kembali ke desimal (baca dari bawah ke atas)
        5. **XOR dengan Constant**: Hasil di-XOR dengan constant C
        
        #### Contoh:
        - Input: 15
        - Inverse dari 15: 199 (dari table)
        - 199 binary: 11000111 → dibaca dari bawah: [1,1,1,0,0,0,1,1]
        - Kalikan dengan matrix K
        - Hasil: [1,1,0,1,0,1,1,0]
        - Baca dari bawah ke atas: 11010110 = 214 (desimal)
        - XOR dengan C (99): 214 ⊕ 99 = final output
        """)
    
    submenu = st.tabs([
        "Build S-Box",
        "Multiplicative Inverse Table",
        "Validation",
        "Export S-Box"
    ])
    
    # =========================
    # BUILD S-BOX
    # =========================
    with submenu[0]:
        st.subheader("Build S-Box")
        
        st.markdown("### Step 1: Select or Create Affine Matrix")
        
        matrix_choice = st.radio(
            "Matrix Selection",
            ["Import from Affine Matrix", "Use Custom Affine Matrix", "Enter First Row (Auto Generated Matrix)"]
        )
        
        first_row = None
        matrix = None
        
        if matrix_choice == "Import from Affine Matrix":
            st.markdown("#### Import Matrix from Affine Matrix Menu")
            st.info("📥 Gunakan matrix yang sudah dibuat dari menu Affine Matrix")
            
            if "saved_first_row" in st.session_state:
                first_row_saved = st.session_state.saved_first_row
                st.success(f"✅ Matrix ditemukan: First Row = {''.join(map(str, first_row_saved))} (Index: {int(''.join(map(str, first_row_saved)), 2)})")
                
                if st.checkbox("Gunakan matrix ini", value=True):
                    first_row = first_row_saved
                    matrix = generate_affine_matrix(first_row)
                    props = matrix_properties(matrix)
                    
                    st.markdown("#### Imported Affine Matrix (K)")
                    st.dataframe(pd.DataFrame(matrix))
                    
                    if props["Rank"] == 8:
                        st.success(f"✅ Matrix is VALID (Rank = 8)")
                    else:
                        st.error(f"❌ Matrix is INVALID (Rank = {props['Rank']})")
            else:
                st.warning("⚠️ Belum ada matrix yang disimpan dari menu Affine Matrix")
                st.info("💡 Buat matrix di menu 'Affine Matrix' terlebih dahulu, kemudian gunakan tombol 'Save for S-Box' untuk mengimpor di sini")
        
        elif matrix_choice == "Use Custom Affine Matrix":
            st.markdown("#### Enter Custom Affine Matrix (8×8)")
            st.info("📝 Masukkan setiap baris dari matriks affine Anda (8 bit per baris)")
            
            matrix_rows = []
            valid_matrix = True
            
            for row_num in range(8):
                binary_input = st.text_input(
                    f"Row {row_num} (8-bit binary)",
                    value="1" + "0" * 7 if row_num == 0 else "0" * 8,
                    key=f"matrix_row_{row_num}",
                    max_chars=8
                )
                
                if len(binary_input) != 8 or not all(c in '01' for c in binary_input):
                    st.warning(f"⚠️ Row {row_num}: Please enter exactly 8 bits (0 or 1)")
                    valid_matrix = False
                else:
                    row_data = list(map(int, binary_input))
                    matrix_rows.append(row_data)
            
            if valid_matrix and len(matrix_rows) == 8:
                matrix = np.array(matrix_rows)
                first_row = matrix[0]
                props = matrix_properties(matrix)
                
                st.markdown("#### Generated Affine Matrix (K)")
                st.dataframe(pd.DataFrame(matrix))
                
                if props["Rank"] == 8:
                    st.success(f"✅ Matrix is VALID (Rank = 8)")
                else:
                    st.error(f"❌ Matrix is INVALID (Rank = {props['Rank']}). Please use a different matrix.")
            else:
                st.error("❌ Please enter all 8 rows with valid binary strings")
                matrix = None
                first_row = None
        
        else:
            # Auto generated from first row
            first_row_input = st.radio(
                "First Row Input Method",
                ["Binary String", "Decimal Value"]
            )
            
            if first_row_input == "Binary String":
                binary = st.text_input(
                    "Enter First Row (8-bit binary)",
                    value="00000111"
                )
                if len(binary) == 8 and all(c in '01' for c in binary):
                    first_row = np.array(list(map(int, binary)))
                else:
                    st.warning("⚠️ Please enter a valid 8-bit binary string")
            else:
                dec = st.number_input(
                    "Decimal Value (0-255)",
                    min_value=0,
                    max_value=255,
                    value=7
                )
                first_row = np.array(list(map(int, f"{dec:08b}")))
            
            if first_row is not None:
                matrix = generate_affine_matrix(first_row)
                props = matrix_properties(matrix)
                
                st.markdown("#### Generated Affine Matrix (K)")
                st.dataframe(pd.DataFrame(matrix))
                
                if props["Rank"] == 8:
                    st.success(f"✅ Matrix is VALID (Rank = 8)")
                else:
                    st.error(f"❌ Matrix is INVALID (Rank = {props['Rank']})")
        
        # Step 2: Select Constant
        st.markdown("---")
        st.markdown("### Step 2: Enter 8-bit Constant (C)")
        
        constant_type = st.radio(
            "Constant Type",
            ["C_AES", "Custom Constant"]
        )
        
        constant = None
        
        if constant_type == "C_AES":
            st.info("Using AES standard constant: 0x63 (99 decimal)")
            constant = 0x63
        else:
            const_input = st.radio(
                "Constant Input Method",
                ["Binary String", "Decimal Value", "Hexadecimal Value"]
            )
            
            if const_input == "Binary String":
                binary_const = st.text_input(
                    "Enter Constant (8-bit binary)",
                    value="01100011"
                )
                if len(binary_const) == 8 and all(c in '01' for c in binary_const):
                    constant = int(binary_const, 2)
                else:
                    st.warning("⚠️ Please enter a valid 8-bit binary string")
            elif const_input == "Decimal Value":
                constant = st.number_input(
                    "Decimal Value (0-255)",
                    min_value=0,
                    max_value=255,
                    value=99
                )
            else:
                hex_const = st.text_input(
                    "Hexadecimal Value (0x00-0xFF)",
                    value="0x63"
                )
                try:
                    constant = int(hex_const, 16)
                    if constant > 255:
                        st.warning("⚠️ Value must be between 0x00 and 0xFF")
                        constant = None
                except ValueError:
                    st.warning("⚠️ Please enter a valid hexadecimal value")
        
        # Display configuration
        if first_row is not None and matrix is not None and constant is not None:
            # Save to session state for validation and export
            st.session_state.saved_first_row = first_row
            st.session_state.saved_constant = constant
            # Clear any previous test results when configuration/index changes
            if "test_results" in st.session_state:
                del st.session_state["test_results"]
            
            display_affine_configuration(first_row, constant)
            
            # Step 3: Example Transformations with Details
            st.markdown("---")
            st.markdown("### Step 3: Example Transformations")
            st.markdown("See how specific input values are transformed:")
            
            # Search index feature
            st.markdown("#### 🔍 Search Specific Input")
            search_col1, search_col2 = st.columns([3, 1])
            
            with search_col1:
                search_input = st.number_input(
                    "Enter input value to search (0-255)",
                    min_value=0,
                    max_value=255,
                    value=15,
                    key="search_input"
                )
            
            with search_col2:
                st.write("")
                st.write("")
                search_button = st.button("🔎 Search", use_container_width=True)
            
            # Display search result
            if search_button or search_input is not None:
                search_output = apply_affine_transformation(search_input, matrix, constant)
                
                st.markdown("---")
                st.markdown("#### Search Result")
                
                result_col1, result_col2, result_col3 = st.columns(3)
                
                with result_col1:
                    st.metric("Input (Decimal)", search_input)
                
                with result_col2:
                    st.metric("Input (Hex)", f"0x{search_input:02X}")
                
                with result_col3:
                    st.metric("Output (Decimal)", search_output)
                
                result_col1, result_col2 = st.columns(2)
                
                with result_col1:
                    st.write(f"**Output (Hex):** `0x{search_output:02X}`")
                
                with result_col2:
                    st.write(f"**Output (Binary):** `{format(search_output, '08b')}`")
                
                # Show detailed steps
                if st.checkbox(f"Show detailed transformation steps", key=f"show_search_steps"):
                    display_transformation_steps(search_input, matrix, constant)
            
            st.markdown("---")
            
            # Example inputs tabs
            st.markdown("#### 📊 Example Inputs (0, 15, 255)")
            
            # Example inputs
            example_inputs = [0, 15, 255]
            
            # Display example transformations
            tabs_examples = st.tabs([f"Input {i}" for i in example_inputs])
            
            for idx, (tab, input_val) in enumerate(zip(tabs_examples, example_inputs)):
                with tab:
                    output_val = apply_affine_transformation(input_val, matrix, constant)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(f"Input", f"{input_val} (0x{input_val:02X})")
                    with col2:
                        st.metric(f"Output", f"{output_val} (0x{output_val:02X})")
                    
                    # Show detailed steps
                    if st.checkbox(f"Show detailed steps for input {input_val}", key=f"show_steps_{input_val}"):
                        display_transformation_steps(input_val, matrix, constant)
            
            # Generate full S-Box
            st.markdown("---")
            if st.checkbox("Generate Full S-Box Table (0-255)", value=False):
                st.markdown("#### Complete S-Box Table")
                
                with st.spinner("Generating S-Box..."):
                    sbox_table = []
                    for i in range(256):
                        output = apply_affine_transformation(i, matrix, constant)
                        sbox_table.append({"Input": i, "Output": output})
                    
                    sbox_df = pd.DataFrame(sbox_table)
                    st.dataframe(sbox_df, use_container_width=True)
                    
                    # Display as matrix (16x16)
                    st.markdown("#### S-Box as 16×16 Matrix")
                    sbox_matrix = np.zeros((16, 16), dtype=int)
                    for i in range(256):
                        output = apply_affine_transformation(i, matrix, constant)
                        sbox_matrix[i // 16, i % 16] = output
                    
                    st.dataframe(pd.DataFrame(sbox_matrix), use_container_width=True)
    
    # =========================
    # MULTIPLICATIVE INVERSE TABLE
    # =========================
    with submenu[1]:
        st.subheader("Multiplicative Inverse Table")
        st.markdown("""
        Tabel ini berisi nilai inverse multiplicative dalam GF(2^8).
        Digunakan sebagai langkah pertama dalam affine transformation.
        """)
        
        # Display as table
        inverse_table = []
        for i in range(256):
            inverse_table.append({"Index": i, "Inverse": MULTIPLICATIVE_INVERSE[i]})
        
        st.dataframe(pd.DataFrame(inverse_table), use_container_width=True)
        
        # Display as 16x16 matrix
        st.markdown("#### As 16×16 Matrix")
        inverse_matrix = np.zeros((16, 16), dtype=int)
        for i in range(256):
            inverse_matrix[i // 16, i % 16] = MULTIPLICATIVE_INVERSE[i]
        
        st.dataframe(pd.DataFrame(inverse_matrix), use_container_width=True)
    
    # =========================
    # VALIDATION
    # =========================
    with submenu[2]:
        st.subheader("Validation")
        st.markdown("""
        Validasi S-Box mengecek:
        1. **Permutasi**: Semua nilai 0-255 muncul tepat 1 kali (tidak ada duplikat)
        2. **Distribusi Bit**: Setiap bit position memiliki 128 nilai 0 dan 128 nilai 1
        
        Keduanya harus terpenuhi untuk S-Box yang valid.
        """)
        
        if st.button("🔍 Run Validation", key="run_validation"):
            if "saved_first_row" in st.session_state and st.session_state.get("saved_constant"):
                saved_first_row = st.session_state.saved_first_row
                saved_constant = st.session_state.saved_constant
                saved_matrix = generate_affine_matrix(saved_first_row)
                
                st.markdown("### Validation Results")
                
                # Display Matrix Information
                st.markdown("#### 📊 Configuration Matrix")
                
                config_col1, config_col2 = st.columns(2)
                
                with config_col1:
                    st.markdown("**Affine Matrix K:**")
                    st.dataframe(pd.DataFrame(saved_matrix))
                
                with config_col2:
                    st.markdown("**Matrix Properties:**")
                    props = matrix_properties(saved_matrix)
                    matrix_info = f"""
                    - **First Row (Index):** {int(''.join(map(str, saved_first_row)), 2)}
                    - **First Row (Binary):** `{''.join(map(str, saved_first_row))}`
                    - **Rank:** {props["Rank"]}
                    - **Constant C:** {saved_constant} (0x{saved_constant:02X})
                    - **Constant (Binary):** `{format(saved_constant, '08b')}`
                    """
                    st.markdown(matrix_info)
                
                st.markdown("---")
                
                # Generate complete S-Box
                sbox_values = []
                sbox_binary_list = []
                
                for i in range(256):
                    output = apply_affine_transformation(i, saved_matrix, saved_constant)
                    sbox_values.append(output)
                    sbox_binary_list.append(format(output, '08b'))
                
                # Check 1: Bijectivity (based on matrix rank)
                st.markdown("#### Check 1: Bijectivity (Matrix Rank)")
                
                rank = props["Rank"]
                is_bijective = rank == 8
                
                bij_col1, bij_col2, bij_col3 = st.columns(3)
                
                with bij_col1:
                    st.metric("Matrix Rank", rank)
                
                with bij_col2:
                    st.metric("Expected", 8)
                
                with bij_col3:
                    st.metric("Is Bijective", "✅ Yes" if is_bijective else "❌ No")
                
                if is_bijective:
                    st.success("""
                    ✅ **BIJECTIVITY CHECK PASSED**
                    
                    Affine matrix memiliki rank 8, yang berarti transformasi adalah bijective.
                    Setiap input yang berbeda akan menghasilkan output yang berbeda.
                    """)
                else:
                    st.error(f"""
                    ❌ **BIJECTIVITY CHECK FAILED**
                    
                    Affine matrix memiliki rank {rank}, bukan 8!
                    Ini berarti transformasi TIDAK bijective.
                    S-Box tidak valid karena matrix tidak full rank.
                    """)
                
                st.markdown("---")
                
                # Check 2: Permutation (no duplicates)
                st.markdown("#### Check 2: Permutation (No Duplicates)")
                
                unique_values = set(sbox_values)
                is_permutation = len(unique_values) == 256
                
                perm_col1, perm_col2, perm_col3 = st.columns(3)
                
                with perm_col1:
                    st.metric("Unique Values", len(unique_values))
                
                with perm_col2:
                    st.metric("Expected", 256)
                
                with perm_col3:
                    st.metric("Is Permutation", "✅ Yes" if is_permutation else "❌ No")
                
                if not is_permutation:
                    st.error("""
                    ❌ **PERMUTATION CHECK FAILED**
                    
                    S-Box mengandung nilai duplikat!
                    Ini berarti beberapa output value muncul lebih dari sekali.
                    """)
                    
                    # Show duplicates
                    from collections import Counter
                    value_counts = Counter(sbox_values)
                    duplicates = {val: count for val, count in value_counts.items() if count > 1}
                    
                    if duplicates:
                        st.markdown("**Nilai yang Duplikat:**")
                        dup_data = []
                        for val, count in sorted(duplicates.items()):
                            dup_data.append({
                                "Value": val,
                                "Hex": f"0x{val:02X}",
                                "Count": count,
                                "Muncul di Input": [str(i) for i, v in enumerate(sbox_values) if v == val]
                            })
                        st.dataframe(pd.DataFrame(dup_data), use_container_width=True)
                else:
                    st.success("✅ **PERMUTATION CHECK PASSED** - Tidak ada duplikat")
                
                st.markdown("---")
                
                # Analyze each bit position
                bit_analysis = {i: {'zeros': 0, 'ones': 0, 'indices': []} for i in range(8)}
                
                for idx, sbox_val in enumerate(sbox_values):
                    sbox_binary = sbox_binary_list[idx]
                    for bit_pos, bit in enumerate(sbox_binary):
                        if bit == '0':
                            bit_analysis[bit_pos]['zeros'] += 1
                        else:
                            bit_analysis[bit_pos]['ones'] += 1
                        bit_analysis[bit_pos]['indices'].append((idx, int(bit)))
                
                # Check 3: Bit Distribution
                st.markdown("#### Check 3: Bit Distribution Analysis")
                
                analysis_data = []
                all_balanced = True
                
                for bit_pos in range(8):
                    zeros = bit_analysis[bit_pos]['zeros']
                    ones = bit_analysis[bit_pos]['ones']
                    is_balanced = (zeros == 128 and ones == 128)
                    
                    if not is_balanced:
                        all_balanced = False
                    
                    # Create bit string representation
                    bit_string = ''.join([str(b) for _, b in bit_analysis[bit_pos]['indices']])
                    
                    analysis_data.append({
                        "Bit Pos": bit_pos,
                        "Zeros (0)": zeros,
                        "Ones (1)": ones,
                        "Total": zeros + ones,
                        "Balanced": "✅" if is_balanced else "❌",
                        "Bit Pattern": bit_string[:32] + "..." if len(bit_string) > 32 else bit_string
                    })
                
                analysis_df = pd.DataFrame(analysis_data)
                st.dataframe(analysis_df, use_container_width=True)
                
                st.markdown("---")
                
                # Detailed bit analysis with splitting
                st.markdown("#### Detailed Bit Splitting Analysis")
                
                detail_tabs = st.tabs([f"Bit {i}" for i in range(8)])
                
                for bit_pos, tab in enumerate(detail_tabs):
                    with tab:
                        st.markdown(f"##### Bit Position {bit_pos} Analysis")
                        
                        zeros = bit_analysis[bit_pos]['zeros']
                        ones = bit_analysis[bit_pos]['ones']
                        is_balanced = (zeros == 128 and ones == 128)
                        
                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("0 Count", zeros)
                        with col2:
                            st.metric("1 Count", ones)
                        with col3:
                            st.metric("Balance", "✅ Yes" if is_balanced else "❌ No")
                        with col4:
                            st.metric("Total", zeros + ones)
                        
                        # Bit pattern visualization
                        st.markdown("**Bit Pattern (Input 0-255):**")
                        bit_pattern = ''.join([str(b) for _, b in bit_analysis[bit_pos]['indices']])
                        
                        # Split into chunks for readability
                        chunk_size = 64
                        for chunk_idx in range(0, len(bit_pattern), chunk_size):
                            chunk = bit_pattern[chunk_idx:chunk_idx+chunk_size]
                            start_idx = chunk_idx
                            end_idx = min(chunk_idx + chunk_size - 1, 255)
                            st.text(f"Input {start_idx:3d}-{end_idx:3d}: {chunk}")
                
                st.markdown("---")
                
                # Overall status
                st.markdown("#### Final Validation Status")
                
                # Combined validation result - ALL conditions must be met
                is_valid = is_bijective and is_permutation and all_balanced
                
                if is_valid:
                    st.success("""
                    ✅ **VALIDATION PASSED - SBOX VALID**
                    
                    S-Box memenuhi SEMUA kriteria:
                    1. ✅ Bijectivity (Matrix Rank = 8)
                    2. ✅ Permutasi sempurna (semua nilai 0-255 muncul tepat 1 kali)
                    3. ✅ Distribusi bit seimbang (setiap bit position: 128 zeros, 128 ones)
                    
                    S-Box ini AMAN dan OPTIMAL untuk digunakan dalam enkripsi.
                    """)
                elif is_bijective and is_permutation and not all_balanced:
                    st.warning("""
                    ⚠️ **PARTIAL VALIDATION - SBOX KURANG OPTIMAL**
                    
                    S-Box memenuhi 2 dari 3 kriteria:
                    1. ✅ Bijectivity (Matrix Rank = 8)
                    2. ✅ Permutasi sempurna (semua nilai 0-255 muncul tepat 1 kali)
                    3. ❌ Distribusi bit tidak seimbang
                    
                    S-Box masih bisa digunakan, tapi distribusi bit yang tidak seimbang
                    dapat mempengaruhi properti kriptografi.
                    """)
                elif is_bijective and not is_permutation:
                    st.error("""
                    ❌ **VALIDATION FAILED - SBOX INVALID**
                    
                    S-Box memiliki masalah kritis:
                    1. ✅ Bijectivity (Matrix Rank = 8)
                    2. ❌ Ada duplikat (beberapa nilai muncul lebih dari sekali)
                    3. ❌ Distribusi bit tidak seimbang (karena duplikat)
                    
                    S-Box TIDAK VALID karena ada duplikat. 
                    Gunakan matrix dan constant yang berbeda.
                    """)
                elif not is_bijective:
                    st.error(f"""
                    ❌ **VALIDATION FAILED - SBOX INVALID**
                    
                    S-Box TIDAK VALID karena matrix tidak bijective:
                    1. ❌ Bijectivity GAGAL (Matrix Rank = {rank}, expected 8)
                    2. {'✅' if is_permutation else '❌'} Permutasi {'terpenuhi' if is_permutation else 'gagal'}
                    3. {'✅' if all_balanced else '❌'} Distribusi bit {'seimbang' if all_balanced else 'tidak seimbang'}
                    
                    S-Box TIDAK VALID. Matrix harus memiliki rank 8 untuk bijective mapping.
                    """)
                else:
                    st.error("""
                    ❌ **VALIDATION FAILED - SBOX INVALID**
                    
                    S-Box memiliki multiple masalah dan TIDAK VALID.
                    """)
                
                # Visualization - Always show chart using actual bit distribution
                st.markdown("#### Bit Distribution Chart")
                
                # Calculate bit distribution CORRECTLY from ALL 256 outputs (including duplicates)
                # This shows the actual distribution of bits in the generated outputs
                chart_zeros_per_bit = [0] * 8
                chart_ones_per_bit = [0] * 8
                
                for sbox_val in sbox_values:
                    sbox_binary = format(sbox_val, '08b')
                    for bit_pos, bit in enumerate(sbox_binary):
                        if bit == '0':
                            chart_zeros_per_bit[bit_pos] += 1
                        else:
                            chart_ones_per_bit[bit_pos] += 1
                
                chart_data = pd.DataFrame({
                    'Bit Position': list(range(8)),
                    'Zeros': chart_zeros_per_bit,
                    'Ones': chart_ones_per_bit
                })
                
                st.bar_chart(chart_data.set_index('Bit Position'))
                
                # Summary statistics
                st.markdown("#### Summary Statistics")
                
                # Calculate actual statistics
                total_unique = len(unique_values)
                total_duplicates = 256 - total_unique
                
                # Recalculate bit statistics from actual unique values only
                actual_zeros_per_bit = [0] * 8
                actual_ones_per_bit = [0] * 8
                
                # Only count bits from actual outputs that appear in sbox
                for sbox_val in unique_values:
                    sbox_binary = format(sbox_val, '08b')
                    for bit_pos, bit in enumerate(sbox_binary):
                        if bit == '0':
                            actual_zeros_per_bit[bit_pos] += 1
                        else:
                            actual_ones_per_bit[bit_pos] += 1
                
                summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                
                with summary_col1:
                    st.metric("Unique Values", total_unique)
                
                with summary_col2:
                    st.metric("Duplicate Count", total_duplicates)
                
                with summary_col3:
                    avg_zeros = sum(actual_zeros_per_bit) / 8 if total_unique > 0 else 0
                    st.metric("Avg Zeros (Unique)", f"{avg_zeros:.1f}")
                
                with summary_col4:
                    avg_ones = sum(actual_ones_per_bit) / 8 if total_unique > 0 else 0
                    st.metric("Avg Ones (Unique)", f"{avg_ones:.1f}")
                
                # Additional statistics table
                st.markdown("#### Detailed Statistics")
                
                stats_table = []
                for i in range(8):
                    stats_table.append({
                        "Bit Position": i,
                        "Unique Zeros": actual_zeros_per_bit[i],
                        "Unique Ones": actual_ones_per_bit[i],
                        "Expected (Balanced)": "128 / 128" if total_unique == 256 else f"{128 * total_unique // 256} / {128 * total_unique // 256}"
                    })
                
                st.dataframe(pd.DataFrame(stats_table), use_container_width=True)
                
                # Store validation results for export
                st.session_state.validation_results = {
                    'sbox': sbox_values,
                    'sbox_binary': sbox_binary_list,
                    'bit_analysis': bit_analysis,
                    'is_bijective': is_bijective,
                    'is_permutation': is_permutation,
                    'is_balanced': all_balanced,
                    'is_valid': is_valid,
                    'matrix': saved_matrix,
                    'constant': saved_constant,
                    'first_row': saved_first_row,
                    'matrix_rank': rank
                }
                # Clear previous test results because validation produced a (potentially) new S-box
                if "test_results" in st.session_state:
                    del st.session_state["test_results"]
            else:
                st.warning("⚠️ Harap buat S-Box terlebih dahulu di tab 'Build S-Box'")
    
    # =========================
    # EXPORT S-BOX
    # =========================
    with submenu[3]:
        st.subheader("Export S-Box")
        st.markdown("Export S-Box dalam berbagai format untuk digunakan dalam aplikasi lain.")
        
        if "validation_results" in st.session_state:
            sbox_values = st.session_state.validation_results['sbox']
            
            st.markdown("### Format Export")
            
            export_format = st.radio(
                "Pilih format export:",
                ["Python List", "C Array", "Hex Values", "Binary Values", "JSON", "CSV", "Excel"]
            )
            
            st.markdown("---")
            
            if export_format == "Python List":
                st.markdown("#### Python List Format")
                
                # Format as Python list
                python_code = "sbox = [\n"
                for i in range(0, 256, 16):
                    row = sbox_values[i:i+16]
                    python_code += "    " + ", ".join(str(v) for v in row) + ",\n"
                python_code += "]"
                
                st.code(python_code, language="python")
            
            elif export_format == "C Array":
                st.markdown("#### C Array Format")
                
                # Format as C array
                c_code = "unsigned char sbox[] = {\n"
                for i in range(0, 256, 16):
                    row = sbox_values[i:i+16]
                    c_code += "    " + ", ".join(f"0x{v:02X}" for v in row) + ",\n"
                c_code += "};"
                
                st.code(c_code, language="c")
            
            elif export_format == "Hex Values":
                st.markdown("#### Hexadecimal Values")
                
                hex_values = [f"0x{v:02X}" for v in sbox_values]
                hex_output = " ".join(hex_values)
                
                st.text_area("Hex Values:", value=hex_output, height=150, disabled=True)
            
            elif export_format == "Binary Values":
                st.markdown("#### Binary Values")
                
                binary_values = [format(v, '08b') for v in sbox_values]
                binary_output = " ".join(binary_values)
                
                st.text_area("Binary Values:", value=binary_output, height=150, disabled=True)
            
            elif export_format == "JSON":
                st.markdown("#### JSON Format")
                
                import json
                json_data = {
                    "sbox": sbox_values,
                    "matrix_first_row": st.session_state.saved_first_row.tolist() if "saved_first_row" in st.session_state else [],
                    "constant": st.session_state.saved_constant if "saved_constant" in st.session_state else None,
                    "validation": {
                        "is_valid": st.session_state.validation_results.get('is_valid', False),
                        "is_bijective": st.session_state.validation_results.get('is_bijective', False),
                        "is_permutation": st.session_state.validation_results.get('is_permutation', False),
                        "is_balanced": st.session_state.validation_results.get('is_balanced', False),
                        "matrix_rank": st.session_state.validation_results.get('matrix_rank', 0)
                    }
                }
                
                json_str = json.dumps(json_data, indent=2)
                st.text_area("JSON Data:", value=json_str, height=300, disabled=True)
            
            elif export_format == "CSV":
                st.markdown("#### CSV Format")
                
                csv_output = "Index,Decimal,Hex,Binary\n"
                for i, val in enumerate(sbox_values):
                    csv_output += f"{i},{val},0x{val:02X},{format(val, '08b')}\n"
                
                st.text_area("CSV Data:", value=csv_output, height=150, disabled=True)
            
            elif export_format == "Excel":
                st.markdown("#### Excel Format (16x16 Grid)")
                sbox_matrix = np.array(sbox_values).reshape(16, 16)
                cols = [f"{i:X}" for i in range(16)] 
                df_preview = pd.DataFrame(sbox_matrix, columns=cols)
                st.write("Preview data:")
                st.dataframe(df_preview, use_container_width=True, hide_index=True)

            # Download buttons
            st.markdown("---")
            st.markdown("### Download")
            
            col1, col2, col3, col4 = st.columns(4)
            
            if export_format == "Python List":
                python_code = "sbox = [\n"
                for i in range(0, 256, 16):
                    row = sbox_values[i:i+16]
                    python_code += "    " + ", ".join(str(v) for v in row) + ",\n"
                python_code += "]"
                
                with col1:
                    st.download_button(
                        label="📥 Download .py",
                        data=python_code,
                        file_name="sbox.py",
                        mime="text/plain"
                    )
            
            elif export_format == "C Array":
                c_code = "unsigned char sbox[] = {\n"
                for i in range(0, 256, 16):
                    row = sbox_values[i:i+16]
                    c_code += "    " + ", ".join(f"0x{v:02X}" for v in row) + ",\n"
                c_code += "};"
                
                with col1:
                    st.download_button(
                        label="📥 Download .c",
                        data=c_code,
                        file_name="sbox.c",
                        mime="text/plain"
                    )
            
            elif export_format == "Hex Values":
                hex_values = [f"0x{v:02X}" for v in sbox_values]
                hex_output = " ".join(hex_values)
                
                with col1:
                    st.download_button(
                        label="📥 Download .txt",
                        data=hex_output,
                        file_name="sbox_hex.txt",
                        mime="text/plain"
                    )
            
            elif export_format == "Binary Values":
                binary_values = [format(v, '08b') for v in sbox_values]
                binary_output = " ".join(binary_values)
                
                with col1:
                    st.download_button(
                        label="📥 Download .txt",
                        data=binary_output,
                        file_name="sbox_binary.txt",
                        mime="text/plain"
                    )
            
            elif export_format == "JSON":
                import json
                json_data = {
                    "sbox": sbox_values,
                    "matrix_first_row": st.session_state.saved_first_row.tolist() if "saved_first_row" in st.session_state else [],
                    "constant": st.session_state.saved_constant if "saved_constant" in st.session_state else None,
                    "validation": {
                        "is_valid": st.session_state.validation_results.get('is_valid', False),
                        "is_bijective": st.session_state.validation_results.get('is_bijective', False),
                        "is_permutation": st.session_state.validation_results.get('is_permutation', False),
                        "is_balanced": st.session_state.validation_results.get('is_balanced', False),
                        "matrix_rank": st.session_state.validation_results.get('matrix_rank', 0)
                    }
                }
                
                json_str = json.dumps(json_data, indent=2)
                
                with col1:
                    st.download_button(
                        label="📥 Download .json",
                        data=json_str,
                        file_name="sbox.json",
                        mime="application/json"
                    )
            
            elif export_format == "CSV":
                csv_output = "Index,Decimal,Hex,Binary\n"
                for i, val in enumerate(sbox_values):
                    csv_output += f"{i},{val},0x{val:02X},{format(val, '08b')}\n"
                
                with col1:
                    st.download_button(
                        label="📥 Download .csv",
                        data=csv_output,
                        file_name="sbox.csv",
                        mime="text/csv"
                    )
            
            elif export_format == "Excel":
                    import io
                    output = io.BytesIO()
                    sbox_matrix_dl = np.array(sbox_values).reshape(16, 16)
                    
                    df_to_save = pd.DataFrame(sbox_matrix_dl) 

                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df_to_save.to_excel(writer, sheet_name='S-Box', index=False, header=False) 
                    
                    st.download_button(
                        label="📥 Download .xlsx",
                        data=output.getvalue(),
                        file_name="sbox.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
        
        else:
            st.info("💡 Jalankan validasi terlebih dahulu di tab 'Validation' untuk mengexport S-Box")
