import streamlit as st
import numpy as np
import binascii
from core.sbox import build_sbox
from core.affine import generate_affine_matrix, gf2_rank
from PIL import Image
import io
import math
import json

try:
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except Exception:
    _HAS_MATPLOTLIB = False

# ==========================================
# 1. KONSTANTA S-BOX
# ==========================================

SBOX_44 = (
    99,205,85,71,25,127,113,219,63,244,109,159,11,228,94,214,
    77,177,201,78,5,48,29,30,87,96,193,80,156,200,216,86,
    116,143,10,14,54,169,148,68,49,75,171,157,92,114,188,194,
    121,220,131,210,83,135,250,149,253,72,182,33,190,141,249,82,
    232,50,21,84,215,242,180,198,168,167,103,122,152,162,145,184,
    43,237,119,183,7,12,125,55,252,206,235,160,140,133,179,192,
    110,176,221,134,19,6,187,59,26,129,112,73,175,45,24,218,
    44,66,151,32,137,31,35,147,236,247,117,132,79,136,154,105,
    199,101,203,52,57,4,153,197,88,76,202,174,233,62,208,91,
    231,53,1,124,0,28,142,170,158,51,226,65,123,186,239,246,
    38,56,36,108,8,126,9,189,81,234,212,224,13,3,40,64,
    172,74,181,118,39,227,130,89,245,166,16,61,106,196,211,107,
    229,195,138,18,93,207,240,95,58,255,209,217,15,111,46,173,
    223,42,115,238,139,243,23,98,100,178,37,97,191,213,222,155,
    165,2,146,204,120,241,163,128,22,90,60,185,67,34,27,248,
    164,69,41,230,104,47,144,251,20,17,150,225,254,161,102,70
)

SBOX_STANDARD = (
    99,124,119,123,242,107,111,197,48,1,103,43,254,215,171,118,
    202,130,201,125,250,89,71,240,173,212,162,175,156,164,114,192,
    183,253,147,38,54,63,247,204,52,165,229,241,113,216,49,21,
    4,199,35,195,24,150,5,154,7,18,128,226,235,39,178,117,
    9,131,44,26,27,110,90,160,82,59,214,179,41,227,47,132,
    83,209,0,237,32,252,177,91,106,203,190,57,74,76,88,207,
    208,239,170,251,67,77,51,133,69,249,2,127,80,60,159,168,
    81,163,64,143,146,157,56,245,188,182,218,33,16,255,243,210,
    205,12,19,236,95,151,68,23,196,167,126,61,100,93,25,115,
    96,129,79,220,34,42,144,136,70,238,184,20,222,94,11,219,
    224,50,58,10,73,6,36,92,194,211,172,98,145,149,228,121,
    231,200,55,109,141,213,78,169,108,86,244,234,101,122,174,8,
    186,120,37,46,28,166,180,198,232,221,116,31,75,189,139,138,
    112,62,181,102,72,3,246,14,97,53,87,185,134,193,29,158,
    225,248,152,17,105,217,142,148,155,30,135,233,206,85,40,223,
    140,161,137,13,191,230,66,104,65,153,45,15,176,84,187,22
)

def generate_inverse_sbox(sbox):
    inv = [0] * 256
    for i in range(256):
        inv[sbox[i]] = i
    return tuple(inv)

INV_SBOX_44 = generate_inverse_sbox(SBOX_44)
INV_SBOX_STANDARD = generate_inverse_sbox(SBOX_STANDARD)
RCON = (0x00,0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1B,0x36,0x6C,0xD8,0xAB,0x4D,0x9A)
xtime = lambda a: (((a << 1) ^ 0x1B) & 0xFF) if (a & 0x80) else ((a << 1) & 0xFF)

# ==========================================
# 2. CORE AES ENGINE
# ==========================================

class AESCoreCustom:
    def __init__(self, key_hex, sbox=SBOX_STANDARD, inv_sbox=INV_SBOX_STANDARD):
        self.master_key = int(key_hex, 16)
        self.sbox = sbox
        self.inv_sbox = inv_sbox
        self.round_keys = []
        self.expand_key(self.master_key)

    def text2matrix(self, text_int):
        matrix = []
        for i in range(16):
            byte = (text_int >> (8 * (15 - i))) & 0xFF
            if i % 4 == 0: matrix.append([byte])
            else: matrix[i // 4].append(byte)
        return matrix

    def matrix2text(self, matrix):
        text = 0
        for i in range(4):
            for j in range(4):
                text |= (matrix[i][j] << (8 * (15 - (4 * i + j))))
        return text

    def expand_key(self, master_key):
        self.round_keys = self.text2matrix(master_key)
        for i in range(4, 44):
            self.round_keys.append([])
            if i % 4 == 0:
                word = [self.round_keys[i-1][1], self.round_keys[i-1][2], self.round_keys[i-1][3], self.round_keys[i-1][0]]
                for j in range(4): word[j] = self.sbox[word[j]]
                word[0] ^= RCON[i // 4]
                for j in range(4): self.round_keys[i].append(word[j] ^ self.round_keys[i-4][j])
            else:
                for j in range(4): self.round_keys[i].append(self.round_keys[i-4][j] ^ self.round_keys[i-1][j])

    def sub_bytes(self, s):
        for i in range(4):
            for j in range(4): s[i][j] = self.sbox[s[i][j]]

    def shift_rows(self, s):
        s[0][1], s[1][1], s[2][1], s[3][1] = s[1][1], s[2][1], s[3][1], s[0][1]
        s[0][2], s[1][2], s[2][2], s[3][2] = s[2][2], s[3][2], s[0][2], s[1][2]
        s[0][3], s[1][3], s[2][3], s[3][3] = s[3][3], s[0][3], s[1][3], s[2][3]

    def mix_columns(self, s):
        for i in range(4):
            a = s[i]
            t = a[0] ^ a[1] ^ a[2] ^ a[3]
            u = a[0]
            a[0] ^= t ^ xtime(a[0] ^ a[1])
            a[1] ^= t ^ xtime(a[1] ^ a[2])
            a[2] ^= t ^ xtime(a[2] ^ a[3])
            a[3] ^= t ^ xtime(a[3] ^ u)

    def add_round_key(self, state, key):
        for i in range(4):
            for j in range(4): state[i][j] ^= key[i][j]

    def encrypt(self, plaintext_hex):
        state = self.text2matrix(int(plaintext_hex, 16))
        self.add_round_key(state, self.round_keys[:4])
        for i in range(1, 10):
            self.sub_bytes(state); self.shift_rows(state); self.mix_columns(state)
            self.add_round_key(state, self.round_keys[4*i : 4*(i+1)])
        self.sub_bytes(state); self.shift_rows(state)
        self.add_round_key(state, self.round_keys[40:])
        return hex(self.matrix2text(state))[2:].zfill(32).upper()

    def inv_sub_bytes(self, s):
        for i in range(4):
            for j in range(4): s[i][j] = self.inv_sbox[s[i][j]]

    def inv_shift_rows(self, s):
        s[0][1], s[1][1], s[2][1], s[3][1] = s[3][1], s[0][1], s[1][1], s[2][1]
        s[0][2], s[1][2], s[2][2], s[3][2] = s[2][2], s[3][2], s[0][2], s[1][2]
        s[0][3], s[1][3], s[2][3], s[3][3] = s[1][3], s[2][3], s[3][3], s[0][3]

    def inv_mix_columns(self, s):
        for i in range(4):
            u = xtime(xtime(s[i][0] ^ s[i][2]))
            v = xtime(xtime(s[i][1] ^ s[i][3]))
            s[i][0] ^= u; s[i][1] ^= v; s[i][2] ^= u; s[i][3] ^= v
        self.mix_columns(s)

    def decrypt(self, ciphertext_hex):
        state = self.text2matrix(int(ciphertext_hex, 16))
        self.add_round_key(state, self.round_keys[40:])
        self.inv_shift_rows(state); self.inv_sub_bytes(state)
        for i in range(9, 0, -1):
            self.add_round_key(state, self.round_keys[4*i : 4*(i+1)])
            self.inv_mix_columns(state); self.inv_shift_rows(state); self.inv_sub_bytes(state)
        self.add_round_key(state, self.round_keys[:4])
        return hex(self.matrix2text(state))[2:].zfill(32).upper()

class SimpleAES_RGB:
    def __init__(self, key_text, sbox, inv_sbox):
        self.key = [ord(c) for c in key_text[:16].ljust(16)]
        self.sbox = list(sbox)
        self.inv_sbox = list(inv_sbox)

    def _diffuse_block(self, block):
        new_block = list(block)
        for i in range(1, 16): new_block[i] = (new_block[i] ^ new_block[i-1]) % 256
        return new_block

    def _undiffuse_block(self, block):
        new_block = list(block)
        for i in range(15, 0, -1): new_block[i] = (new_block[i] ^ new_block[i-1]) % 256
        return new_block

    def encrypt_bytes(self, byte_data):
        encrypted = []
        prev_cipher_block = self.key
        for i in range(0, len(byte_data), 16):
            block = byte_data[i:i+16]
            chained = [b ^ p for b, p in zip(block, prev_cipher_block)]
            sub = [self.sbox[b] for b in chained]
            diffused = self._diffuse_block(sub)
            cipher_block = [b ^ self.key[idx % 16] for idx, b in enumerate(diffused)]
            encrypted.extend(cipher_block)
            prev_cipher_block = cipher_block
        return encrypted

    def decrypt_bytes(self, byte_data):
        decrypted = []
        prev_cipher_block = self.key
        for i in range(0, len(byte_data), 16):
            block = byte_data[i:i+16]
            xor_back = [b ^ self.key[idx % 16] for idx, b in enumerate(block)]
            undiffused = self._undiffuse_block(xor_back)
            sub_inv = [self.inv_sbox[b] for b in undiffused]
            original = [s ^ p for s, p in zip(sub_inv, prev_cipher_block)]
            decrypted.extend(original)
            prev_cipher_block = block
        return decrypted

# ==========================================
# 3. ANALYSIS HELPERS
# ==========================================

def calc_entropy(arr):
    flat = arr.flatten()
    marg = np.histogram(flat, bins=256, range=(0,255), density=True)[0]
    return -np.sum(marg * np.log2(marg + 1e-9))

def calc_npcr_uaci(c1, c2):
    diff = (c1 != c2).astype(int)
    npcr = (np.sum(diff) / c1.size) * 100
    uaci = (np.sum(np.abs(c1.astype(float) - c2.astype(float))) / (255 * c1.size)) * 100
    return npcr, uaci

def run_image_analysis_dynamic(arr, flat_padded, pad_len, engine, enc_bytes, h, w, c, mode):
    """Analisis Keamanan Berdasarkan Mode (Encrypt vs Decrypt)"""
    with st.spinner(f"Running Analysis for {mode.upper()} context..."):
        
        st.markdown("---")
        if mode == "encrypt":
            st.subheader("📊 Encryption Security Analysis (NPCR, UACI, Entropy)")
            
            # 1. Gen Image Mod (P2)
            mod_int = arr.astype(np.int16)
            mod_int[0,0,0] = (int(mod_int[0,0,0]) + 1) % 256
            flat_mod = mod_int.flatten().astype(np.uint8)
            if pad_len > 0: flat_mod = np.pad(flat_mod, (0, pad_len), mode='constant')
            
            # 2. Enkripsi
            enc_mod = engine.encrypt_bytes(flat_mod.tolist())
            cipher_arr = np.array(enc_bytes[:arr.size]).reshape(h,w,c).astype(np.uint8)
            cipher_mod_arr = np.array(enc_mod[:arr.size]).reshape(h,w,c).astype(np.uint8)
            
            # 3. Hitung NPCR, UACI, Entropy
            npcr, uaci = calc_npcr_uaci(cipher_arr, cipher_mod_arr)
            ent_val = calc_entropy(cipher_arr)

            # --- UI OUTPUT ---
            m1, m2, m3 = st.columns(3)
            m1.metric("NPCR", f"{npcr:.4f}%")
            m2.metric("UACI", f"{uaci:.4f}%")
            m3.metric("Entropy", f"{ent_val:.4f}")

            st.write("**Histogram: Encrypted Image (Cipher Distribution)**")
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.hist(cipher_arr.ravel(), bins=256, range=(0,255), color='#1f77b4')
            st.pyplot(fig)

        elif mode == "decrypt":
            st.subheader("📊 Decryption Quality Analysis (Correlation, PSNR)")
            
            # 1. Dekripsi
            dec_bytes = engine.decrypt_bytes(enc_bytes)
            dec_arr = np.array(dec_bytes[:arr.size]).reshape(h,w,c).astype(np.uint8)
            
            # 2. Hitung PSNR & Correlation
            corr_val = np.corrcoef(arr.flatten(), dec_arr.flatten())[0, 1]
            mse = np.mean((arr.astype(float) - dec_arr.astype(float)) ** 2)
            psnr_val = 20 * math.log10(255.0 / math.sqrt(mse)) if mse > 0 else float('inf')

            # --- UI OUTPUT ---
            m1, m2 = st.columns(2)
            m1.metric("Correlation", f"{corr_val:.4f}")
            m2.metric("PSNR", "∞ dB" if psnr_val == float('inf') else f"{psnr_val:.2f} dB")

            st.write("**Histogram: Decrypted Image (Restored Distribution)**")
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.hist(dec_arr.ravel(), bins=256, range=(0,255), color='#2ca02c')
            st.pyplot(fig)

# ==========================================
# 4. MAIN APP LOGIC
# ==========================================

def aes_menu():
    st.header("🔐 AES Image Implementation with Custom S-Box")
    
    # 1. Siapkan List Pilihan S-Box
    sbox_options = ["S-Box Aktif (dari Testing/Upload)", "Affine-derived S-Box", "S-Box44 (Jurnal)", "S-Box Standar (AES)"]
    
    sbox_option = st.selectbox("Pilih S-Box Strategi:", 
                              sbox_options,
                              key="main_sbox_sel")

    # AUTO RESET LOGIC: Jika S-Box diganti, hapus state gambar
    if "prev_sbox" not in st.session_state: st.session_state.prev_sbox = sbox_option
    if st.session_state.prev_sbox != sbox_option:
        for k in ['aes_img_ready', 'aes_img_dec_ready', 'show_analysis', 'analysis_mode']:
            if k in st.session_state: del st.session_state[k]
        st.session_state.prev_sbox = sbox_option

    # --- LOGIKA PENENTUAN S-BOX ---
    s_u = None
    i_u = None

    if sbox_option == "S-Box Aktif (dari Testing/Upload)":
        if "validation_results" in st.session_state:
            # Mengambil S-Box yang tersimpan di memori (dari Excel/Generator)
            s_u = tuple(st.session_state.validation_results['sbox'])
            i_u = generate_inverse_sbox(s_u)
            source_info = st.session_state.validation_results.get('source', 'Generated')
            st.info(f"✅ Menggunakan S-Box Aktif dari: **{source_info}**")
        else:
            st.error("❌ Belum ada S-Box aktif di memori. Silakan Upload Excel atau Generate S-Box di menu Testing/Affine terlebih dahulu.")
            return # Berhenti jika tidak ada S-Box

    elif sbox_option == "Affine-derived S-Box":
        dec = st.number_input("Affine First Row (decimal)", 0, 255, 7)
        const = st.number_input("Affine Constant (decimal)", 0, 255, 99)
        first_row = np.array(list(map(int, f"{dec:08b}")))
        constant = np.array(list(map(int, f"{const:08b}")))[::-1]
        affine = generate_affine_matrix(first_row)
        if gf2_rank(affine) != 8:
            st.error("Affine matrix tidak valid"); return
        s_u = tuple(build_sbox(affine, constant))
        i_u = generate_inverse_sbox(s_u)

    elif sbox_option == "S-Box44 (Jurnal)":
        s_u, i_u = SBOX_44, INV_SBOX_44
    else:
        s_u, i_u = SBOX_STANDARD, INV_SBOX_STANDARD

    tabs = st.tabs(["Text Demo", "Image Demo"])

    # --- TAB TEXT (Simplified) ---
 # --- TAB TEXT (AES Demo) ---
    with tabs[0]:
        st.subheader("📝 Text Encryption & Decryption")
        
        key_t = st.text_input("Kunci (16 karakter)", value="kuncirahasia1234", key="tk")
        
        # --- Bagian Enkripsi ---
        st.markdown("#### 🔒 Encryption")
        pt_t = st.text_area("Masukkan Plaintext", value="Halo AES Demo", key="tp")
        
        if st.button("Encrypt Text", use_container_width=True):
            if len(key_t) != 16:
                st.error("Kunci harus tepat 16 karakter!")
            else:
                kh = binascii.hexlify(key_t.encode()).decode().upper()
                # Padding manual agar pas 16 byte (1 blok)
                ph = binascii.hexlify(pt_t.ljust(16)[:16].encode()).decode().upper()
                
                # Inisialisasi engine dengan S-Box terpilih
                eng = AESCoreCustom(kh, sbox=s_u, inv_sbox=i_u)
                cipher_result = eng.encrypt(ph)
                
                st.success("Teks Berhasil Dienkripsi!")
                st.code(cipher_result, language="text")
                st.info("💡 Salin kode di atas untuk mencoba fitur dekripsi di bawah.")

        st.markdown("---")

        # --- Bagian Dekripsi ---
        st.markdown("#### 🔓 Decryption")
        ct_t = st.text_input("Masukkan Ciphertext (Hex 32 karakter)", key="ct_input")
        
        if st.button("Decrypt Text", use_container_width=True):
            if len(key_t) != 16:
                st.error("Kunci harus tepat 16 karakter!")
            elif len(ct_t) != 32:
                st.error("Ciphertext harus 32 karakter Hex (16 byte)!")
            else:
                try:
                    kh = binascii.hexlify(key_t.encode()).decode().upper()
                    
                    # Inisialisasi engine dengan S-Box terpilih (dan Inverse S-Box nya)
                    eng = AESCoreCustom(kh, sbox=s_u, inv_sbox=i_u)
                    
                    # Proses dekripsi
                    dec_hex = eng.decrypt(ct_t)
                    # Konversi kembali dari Hex ke teks string
                    dec_text = binascii.unhexlify(dec_hex).decode(errors='ignore')
                    
                    st.success("Teks Berhasil Didekripsi!")
                    st.markdown(f"**Plaintext Hasil Dekripsi:**")
                    st.info(dec_text)
                except Exception as e:
                    st.error(f"Gagal mendekripsi: Pastikan format Hex benar. (Error: {e})")

    # --- TAB IMAGE ---
    with tabs[1]:
        img_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])
        
        # AUTO RESET LOGIC: Jika Gambar diganti
        if img_file:
            if "prev_img" not in st.session_state: st.session_state.prev_img = img_file.name
            if st.session_state.prev_img != img_file.name:
                for k in ['aes_img_ready', 'aes_img_dec_ready', 'show_analysis', 'analysis_mode']:
                    if k in st.session_state: del st.session_state[k]
                st.session_state.prev_img = img_file.name

        c_cfg1, c_cfg2 = st.columns(2)
        with c_cfg1: resize_val = st.selectbox("Resize", ["No Resize","128x128","64x64"], index=1)
        with c_cfg2: key_i = st.text_input("Kunci Image (16 char)", value="kuncirahasia1234", key="ik")

        if img_file:
            img_obj = Image.open(img_file).convert('RGB')
            if resize_val != "No Resize":
                s = int(resize_val.split("x")[0])
                img_obj = img_obj.resize((s, s))
            
            arr = np.array(img_obj)
            h, w, c = arr.shape
            flat = arr.flatten()
            pad_len = (16 - (len(flat) % 16)) % 16
            flat_padded = np.pad(flat, (0, pad_len), mode='constant') if pad_len > 0 else flat
            
            eng_img = SimpleAES_RGB(key_i, s_u, i_u)

            # TOMBOL PROSES
            st.write("")
            btn_c1, btn_c2 = st.columns(2)
            
            if btn_c1.button("🔒 Encrypt Image", use_container_width=True):
                st.session_state.aes_img_enc_bytes = eng_img.encrypt_bytes(flat_padded.tolist())
                st.session_state.aes_img_ready = True
                st.session_state.aes_img_dec_ready = False # Hilangkan dekripsi lama
                st.session_state.analysis_mode = "encrypt"
                st.session_state.show_analysis = False # Reset hasil analisis layar

            if btn_c2.button("🔓 Decrypt Image", use_container_width=True):
                if st.session_state.get('aes_img_ready'):
                    st.session_state.aes_img_dec_bytes = eng_img.decrypt_bytes(st.session_state.aes_img_enc_bytes)
                    st.session_state.aes_img_dec_ready = True
                    st.session_state.analysis_mode = "decrypt"
                    st.session_state.show_analysis = False
                else:
                    st.warning("Encrypt image first!")

            # DISPLAY
            st.write("")
            v_col1, v_col2 = st.columns(2)
            with v_col1:
                st.image(arr, caption="Original Image", use_container_width=True)
            
            if st.session_state.get('aes_img_ready'):
                with v_col2:
                    c_arr = np.array(st.session_state.aes_img_enc_bytes[:arr.size]).reshape(h,w,c).astype(np.uint8)
                    st.image(c_arr, caption="Encrypted Image", use_container_width=True)
                
                if st.session_state.get('aes_img_dec_ready'):
                    st.markdown("---")
                    st.subheader("Restored Image (Decrypted)")
                    d_arr = np.array(st.session_state.aes_img_dec_bytes[:arr.size]).reshape(h,w,c).astype(np.uint8)
                    st.image(d_arr, width=350)

                # TOMBOL UTILITIES (Muncul hanya jika ada data)
                st.markdown("### 🛠️ Utilities & Security Analysis")
                u_col1, u_col2, u_col3 = st.columns(3)
                
                with u_col1:
                    if st.button("📊 Run Security Analysis", use_container_width=True):
                        st.session_state.show_analysis = True
                
                with u_col2:
                    c_arr = np.array(st.session_state.aes_img_enc_bytes[:arr.size]).reshape(h,w,c).astype(np.uint8)
                    buf = io.BytesIO()
                    Image.fromarray(c_arr).save(buf, format="PNG")
                    st.download_button("📥 Download Encrypted", buf.getvalue(), "enc.png", "image/png", use_container_width=True)
                
                with u_col3:
                    if st.session_state.get('aes_img_dec_ready'):
                        d_arr = np.array(st.session_state.aes_img_dec_bytes[:arr.size]).reshape(h,w,c).astype(np.uint8)
                        buf_d = io.BytesIO()
                        Image.fromarray(d_arr).save(buf_d, format="PNG")
                        st.download_button("📥 Download Decrypted", buf_d.getvalue(), "dec.png", "image/png", use_container_width=True)
                    else:
                        st.button("📥 Decrypt First to Download", disabled=True, use_container_width=True)

                # LOGIKA TAMPILAN ANALISIS (FULL WIDTH)
                if st.session_state.get('show_analysis'):
                    run_image_analysis_dynamic(
                        arr, flat_padded, pad_len, eng_img, 
                        st.session_state.aes_img_enc_bytes, 
                        h, w, c, 
                        st.session_state.analysis_mode
                    )