import streamlit as st

def home_menu():
    st.header("🏠 Welcome to Affine Matrix Encryption Tool")
    
    st.markdown("""
    Selamat datang di aplikasi **Affine Matrix Encryption Tool**. Aplikasi ini dirancang untuk membantu
    Anda memahami dan mengeksplorasi konsep enkripsi menggunakan matriks affine dan konstruksi S-Box.
    """)
    
    # Quick Stats
    st.markdown("---")
    st.markdown("### 📊 Quick Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Matrices", "256", "Possible 8-bit combinations")
    
    with col2:
        st.metric("Valid Matrices", "~128", "With Rank = 8")
    
    with col3:
        st.metric("Constant Values", "256", "For S-Box transformation")
    
    with col4:
        st.metric("S-Box Size", "256×1", "Substitution values")
    
    # Main Features
    st.markdown("---")
    st.markdown("### ✨ Fitur Utama")
    
    tab1, tab2, tab3 = st.tabs(["🔍 Affine Matrix", "🔐 S-Box", "📚 Konsep"])
    
    with tab1:
        st.markdown("""
        #### 🔍 Affine Matrix Exploration
        
        Modul ini memungkinkan Anda untuk:
        
        **1. Custom Input**
        - Masukkan binary string 8-bit secara manual
        - Gunakan individual bit selector (8 selectbox)
        - Konversi dari nilai desimal (0-255)
        - Lihat matriks yang dihasilkan dengan cyclic shift
        
        **2. Example Matrices**
        - AES Default (Index 143)
        - All Zeros (Index 0)
        - All Ones (Index 255)
        - Pre-defined contoh untuk pembelajaran
        
        **3. Index-based Access**
        - Akses matriks tertentu melalui indeksnya (0-255)
        - Konversi otomatis ke binary representation
        - Validasi matriks secara real-time
        
        **4. Range Explorer**
        - Analisis batch untuk range indeks tertentu
        - Lihat pola validitas dalam jumlah besar
        - Export hasil dalam format tabel
        
        #### Properti yang Dianalisis:
        - **Index Value**: Representasi desimal dari baris pertama
        - **Rank**: Dimensi matriks (harus 8 untuk valid)
        - **Determinant**: Nilai yang menentukan invertibilitas
        - **Ones/Zeros Count**: Jumlah bit 1 dan 0
        - **Status**: VALID atau INVALID berdasarkan rank
        """)
    
    with tab2:
        st.markdown("""
        #### 🔐 S-Box Construction
        
        Modul ini membantu Anda membuat S-Box (Substitution Box) untuk enkripsi blok:
        
        **1. Build S-Box**
        - Pilih atau buat custom affine matrix
        - Pilih constant (AES atau custom)
        - Lihat transformasi contoh (input 0, 15, 255)
        - Generate tabel S-Box lengkap (256 nilai)
        
        **2. Multiplicative Inverse Table**
        - Tabel inverse untuk operasi GF(2^8)
        - Penting untuk dekripsi
        - Coming soon...
        
        **3. Validation**
        - Validasi properti S-Box
        - Cek linearitas dan non-linearitas
        - Coming soon...
        
        **4. Export S-Box**
        - Download dalam format berbeda
        - C code generation
        - Coming soon...
        
        #### Rumus Affine Transformation:
        ```
        Output = (Affine_Matrix × Input) ⊕ Constant
        ```
        """)
    
    with tab3:
        st.markdown("""
        #### 📚 Konsep Penting
        
        **Apa itu Affine Matrix?**
        
        Matriks affine adalah matriks yang digunakan untuk transformasi linier dalam enkripsi.
        Untuk enkripsi 8-bit, matriks harus berukuran 8×8 dan memiliki rank penuh (= 8).
        
        **Mengapa Rank = 8 penting?**
        
        Rank yang sama dengan dimensi matriks menjamin bahwa matriks tersebut **invertible**.
        Ini berarti untuk setiap output yang dihasilkan, kita bisa mengembalikannya ke input asli.
        
        **Cyclic Shift (Pergeseran Siklis)**
        
        Baris pertama matriks diperluas menjadi 8 baris dengan melakukan cyclic left shift:
        ```
        Baris 0: [1, 0, 0, 0, 1, 1, 1, 1]
        Baris 1: [0, 0, 0, 1, 1, 1, 1, 1]  ← shift left 1
        Baris 2: [0, 0, 1, 1, 1, 1, 1, 0]  ← shift left 2
        ...
        ```
        
        **Binary Representation (0-255)**
        
        Setiap kombinasi 8-bit dapat direpresentasikan sebagai angka desimal:
        - 00000000 = 0
        - 00000001 = 1
        - 10001111 = 143
        - 11111111 = 255
        
        **S-Box (Substitution Box)**
        
        S-Box adalah tabel lookup 256 nilai yang digunakan untuk substitusi nonlinear.
        Dibuat dengan menerapkan affine transformation ke setiap input (0-255).
        
        **Affine Transformation**
        
        Transformasi affine menggabungkan:
        1. Transformasi Linear: Input dikalikan dengan matrix
        2. Transformasi Affine: Hasil di-XOR dengan constant
        
        Ini menciptakan sifat non-linear yang penting untuk keamanan enkripsi.
        """)
    
    # How to Use
    st.markdown("---")
    st.markdown("### 🚀 Cara Menggunakan")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Untuk Affine Matrix Exploration:**
        
        1. Klik menu "🔍 Affine Matrix" di navbar
        2. Pilih tab yang sesuai (Custom, Example, dll)
        3. Masukkan input sesuai petunjuk
        4. Lihat hasil matriks dan propertinya
        5. Baca penjelasan detail di expander
        
        **Tips:**
        - Mulai dari Example Matrices untuk melihat contoh
        - Coba berbagai input untuk memahami pola
        - Gunakan Range Explorer untuk analisis batch
        """)
    
    with col2:
        st.markdown("""
        **Untuk S-Box Construction:**
        
        1. Klik menu "🔐 S-Box" di navbar
        2. Buka tab "Build S-Box"
        3. Pilih metode matrix (custom atau auto)
        4. Pilih constant (AES atau custom)
        5. Lihat example transformations
        6. Generate full S-Box jika diperlukan
        
        **Tips:**
        - Gunakan AES constant untuk standar
        - Validasi matrix sebelum S-Box
        - Export hasil untuk keperluan lain
        """)
    
    # Learning Path
    st.markdown("---")
    st.markdown("### 📖 Learning Path Rekomendasi")
    
    st.info("""
    **Pemula:**
    1. Baca penjelasan di Home (halaman ini)
    2. Jelajahi Example Matrices
    3. Coba Custom Input dengan nilai sederhana
    4. Lihat properti dan pahami artinya
    
    **Menengah:**
    1. Pahami konsep cyclic shift dan rank
    2. Coba berbagai input binary
    3. Gunakan Range Explorer untuk melihat pola
    4. Analisis hubungan antara input dan properti
    
    **Advanced:**
    1. Buat custom affine matrix
    2. Eksplorasi S-Box construction
    3. Analisis non-linearitas S-Box
    4. Pahami aplikasi dalam enkripsi nyata
    """)
    
    # Quick Links
    st.markdown("---")
    st.markdown("### 🔗 Navigasi Cepat")
    
    nav_col1, nav_col2, nav_col3 = st.columns(3)
    
    with nav_col1:
        if st.button("🔍 Ke Affine Matrix", use_container_width=True):
            st.session_state.current_page = "affine"
            st.rerun()
    
    with nav_col2:
        if st.button("🔐 Ke S-Box", use_container_width=True):
            st.session_state.current_page = "sbox"
            st.rerun()
    
    with nav_col3:
        if st.button("📚 Dokumentasi Lengkap", use_container_width=True):
            st.session_state.current_page = "docs"
            st.rerun()
    
    # Footer Info
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>💡 <strong>Pro Tip:</strong> Klik expander di setiap menu untuk membaca penjelasan detail tentang cara kerja setiap fitur.</p>
        <p style="margin-top: 1rem; color: #999; font-size: 0.85rem;">
            Affine Matrix Encryption Tool | v1.0 | Untuk Pembelajaran Kriptografi
        </p>
    </div>
    """, unsafe_allow_html=True)
