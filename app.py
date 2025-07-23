import pandas as pd
import numpy as np
import joblib
import streamlit as st

# === CONFIG ===
st.set_page_config(page_title="Prediksi Petir - STMKG", layout="wide")

# === LOAD MODEL ===
@st.cache_resource
def load_model():
    return joblib.load("model_petir_xgb (3).pkl")

try:
    model, fitur_model = load_model()
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.stop()

# === CUSTOM CSS ===
st.markdown("""
    <style>
        .logo-title-container {
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 20px;
        }
        .logo-title-container img {
            width: 70px;
        }
        .main-title {
            font-size: 30px;
            font-weight: bold;
            color: #004080;
        }
        .stMetric label, .stMetric div {
            text-align: center !important;
        }
        .stProgress > div > div {
            background-color: #ffaa00;
        }
    </style>
""", unsafe_allow_html=True)

# === LOGO + JUDUL ===
st.markdown(
    f"""
    <div class="logo-title-container">
        <img src="logo_stmkg.png" alt="Logo STMKG">
        <div class="main-title">Prediksi Kejadian Petir Harian</div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("Masukkan parameter atmosfer untuk memprediksi kemungkinan petir menggunakan model machine learning.")

# === SIDEBAR ===
st.sidebar.image("logo_stmkg.png", width=150)
st.sidebar.header("🔧 Input Parameter")

hour = st.sidebar.selectbox("Jam (UTC)", options=[0, 12])
season = st.sidebar.selectbox("Musim", options=[1, 2, 3, 4],
    format_func=lambda x: {1: "DJF (Des-Jan-Feb)", 2: "MAM (Mar-Apr-Mei)", 3: "JJA (Jun-Jul-Agu)", 4: "SON (Sep-Okt-Nov)"}[x])
month = st.sidebar.slider("Bulan", 1, 12, 1)
KI = st.sidebar.number_input("KI Index", value=30.0)
SWEAT = st.sidebar.number_input("SWEAT Index", value=200.0)
LI = st.sidebar.number_input("Lifted Index", value=-2.0)
CAPE = st.sidebar.number_input("CAPE (J/kg)", value=1000.0)
TTI = st.sidebar.number_input("TTI Index", value=48.0)
SI = st.sidebar.number_input("Showalter Index", value=1.0)
PW = st.sidebar.number_input("Precipitable Water (mm)", value=40.0)

# === PREDIKSI ===
if st.sidebar.button("🔍 Prediksi Sekarang"):
    cos_month = np.cos(2 * np.pi * month / 12)
    input_data = pd.DataFrame([{
        'hour': hour, 'season': season, 'KI': KI, 'SWEAT': SWEAT,
        'LI': LI, 'CAPE': CAPE, 'TTI': TTI, 'SI': SI,
        'PW': PW, 'cos_month': cos_month
    }])[fitur_model]

    prob = model.predict_proba(input_data)[0, 1]
    klasifikasi = "⚡ POTENSI PETIR" if prob >= 0.5 else "✅ NON-PETIR"

    # === OUTPUT HASIL ===
    st.subheader("📊 Hasil Prediksi")
    col1, col2 = st.columns(2)
    col1.metric("Klasifikasi", klasifikasi)
    col2.metric("Probabilitas", f"{prob:.2f}")

    st.markdown("**Visualisasi Probabilitas**")
    st.progress(prob)

    with st.expander("ℹ️ Penjelasan Model", expanded=False):
        st.markdown("""
        - Probabilitas ≥ 0.5 → **diprediksi akan terjadi petir**
        - Model XGBoost dilatih menggunakan data atmosfer historis
        - Harap gunakan bersama analisis cuaca manual
        """)

