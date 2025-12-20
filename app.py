import streamlit as st
from ui.home_menu import home_menu
from ui.affine_menu import affine_menu
from ui.sbox_menu import sbox_menu
from ui.sbox_testing import sbox_testing_menu

# Page configuration
st.set_page_config(
    page_title="Kriptografi Affine Matrix",
    page_icon="🔐",
    layout="wide"
)

# Custom CSS untuk navbar
st.markdown("""
    <style>
    /* Header styling */
    .header-container {
        text-align: center;
        padding: 1.5rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 0.5rem;
        color: white;
        margin-bottom: 1rem;
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
        letter-spacing: 0.05em;
    }
    
    .header-subtitle {
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.9);
        margin: 0.5rem 0;
    }
    
    /* Navbar styling */
    [data-testid="stHorizontalBlock"] {
        gap: 1rem;
    }
    
    /* Button styling */
    .nav-button {
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        border: none;
        cursor: pointer;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .nav-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    </style>
""", unsafe_allow_html=True)

# Header dengan title
st.markdown("""
    <div class="header-container">
        <div style="font-size: 3rem; margin-bottom: 0.5rem;">🔐</div>
        <div class="header-title">Affine Matrix Encryption</div>
        <div class="header-subtitle">Eksplorasi dan Konstruksi S-Box untuk Kriptografi</div>
    </div>
""", unsafe_allow_html=True)

# Navbar di bagian atas
st.markdown("---")
nav_col1, nav_col2, nav_col3, nav_col4, nav_col5, nav_col6 = st.columns(6)

with nav_col1:
    if st.button("🏠 Home", use_container_width=True, key="nav_home"):
        st.session_state.current_page = "home"

with nav_col2:
    if st.button("🔍 Affine Matrix", use_container_width=True, key="nav_affine"):
        st.session_state.current_page = "affine"

with nav_col3:
    if st.button("🔐 S-Box", use_container_width=True, key="nav_sbox"):
        st.session_state.current_page = "sbox"

with nav_col4:
    if st.button("🧪 S-Box Testing", use_container_width=True, key="nav_testing"):
        st.session_state.current_page = "testing"

with nav_col5:
    if st.button("📊 Documentation", use_container_width=True, key="nav_docs"):
        st.session_state.current_page = "docs"

with nav_col6:
    if st.button("ℹ️ About", use_container_width=True, key="nav_about"):
        st.session_state.current_page = "about"

st.markdown("---")

# Initialize session state
if "current_page" not in st.session_state:
    st.session_state.current_page = "home"

# Page routing
if st.session_state.current_page == "home":
    home_menu()
elif st.session_state.current_page == "affine":
    affine_menu()
elif st.session_state.current_page == "sbox":
    sbox_menu()
elif st.session_state.current_page == "testing":
    sbox_testing_menu()
elif st.session_state.current_page == "docs":
    st.header("📊 Documentation")
    st.markdown("""
    ### Dokumentasi Affine Matrix Encryption
    
    Halaman ini berisi dokumentasi lengkap mengenai Affine Matrix Encryption.
    
    Coming soon...
    """)
elif st.session_state.current_page == "about":
    st.header("ℹ️ About")
    st.markdown("""
    ### Tentang Aplikasi
    
    Aplikasi ini adalah tools untuk eksplorasi dan konstruksi Affine Matrix dalam kriptografi.
    
    **Fitur:**
    - Affine Matrix Exploration dengan berbagai input method
    - S-Box Construction untuk enkripsi
    - Range Explorer untuk analisis batch
    - Validasi dan properti matriks
    
    **Dibuat untuk:** Tugas Kriptografi
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.9rem;">
    <p>Affine Matrix Encryption Tool v1.0 | Kriptografi</p>
</div>
""", unsafe_allow_html=True)
