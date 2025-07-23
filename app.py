import pandas as pd
import numpy as np
import joblib
import streamlit as st

st.set_page_config(page_title="STMKG Lightning Prediction", layout="centered")

# === LOAD MODEL ===
@st.cache_resource
def load_model():
    return joblib.load("model_petir_xgb.pkl")

model, fitur_model = load_model()

# === STYLE ===
st.markdown("""
    <style>
        .logo-title-container {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 15px;
            margin-bottom: 20px;
        }
        .logo-title-container img {
            width: 70px;
        }
        .main-title {
            font-size: 28px;
            font-weight: bold;
            color: #004080;
        }
        .centered {
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# === LOGO & JUDUL ===
st.markdown(
    """
    <div class="logo-title-container">
        <img src="logo_stmkg.png" alt="Logo">
        <div class="main-title">Prediksi Kejadian Petir Harian</div>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown('<p class="centered">Masukkan parameter atmosfer di bawah untuk memprediksi potensi petir.</p>', unsafe_allow_html=True)

# === FORM INPUT DI TENGAH ===
with st.form("form_input", clear_on_submit=False):
    col1, col2 = st.columns(2)
    with col1:
        hour = st.selectbox("Jam (UTC)", options=[0, 12])
        season = st.selectbox("Musim", options=[1, 2, 3, 4], format_func=lambda x: {
            1: "DJF (Des-Jan-Feb)",
            2: "MAM (Mar-Apr-Mei)",
            3: "JJA (Jun-Jul-Agu)",
            4: "SON (Sep-Okt-Nov)"
        }[x])
        month = st.slider("Bulan", 1, 12, 1)
        KI = st.number_input("KI Index", value=30.0)
        SWEAT = st.number_input("SWEAT Index", value=200.0)
    with col2:
        LI = st.number_input("Lifted Index", value=-2.0)
        CAPE = st.number_input("CAPE (J/kg)", value=1000.0)
        TTI = st.number_input("TTI Index", value=48.0)
        SI = st.number_input("Showalter Index", value=1.0)
        PW = st.number_input("Precipitable Water (mm)", value=40.0)

    submitted = st.form_submit_button("🔍 Prediksi Sekarang")

# === PREDIKSI ===
if submitted:
    cos_month = np.cos(2 * np.pi * month / 12)
    input_data = pd.DataFrame([{
        'hour': hour, 'season': season, 'KI': KI, 'SWEAT': SWEAT,
        'LI': LI, 'CAPE': CAPE, 'TTI': TTI, 'SI': SI,
        'PW': PW, 'cos_month': cos_month
    }])[fitur_model]

    prob = model.predict_proba(input_data)[0, 1]
    klasifikasi = "⚡ POTENSI PETIR" if prob >= 0.35 else "✅ NON-PETIR"

    st.subheader("📊 Hasil Prediksi")
    col1, col2 = st.columns(2)
    col1.metric("Klasifikasi", klasifikasi)
    col2.metric("Probabilitas", f"{prob:.2f}")
    st.progress(prob)

    with st.expander("ℹ️ Tentang Model"):
        st.markdown("""
        - Model: **XGBoost Classifier**
        - Optimasi ketidakseimbangan data: **SMOTE-ENN**
        - Threshold klasifikasi: **0.35**
        - Fitur: Indeks atmosfer & musiman
        """)

