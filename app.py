# === 1. Import Library ===
import pandas as pd
import numpy as np
import joblib
import streamlit as st

# === 2. Load Model dan Fitur ===
model, fitur_model = joblib.load("model_petir_xgb_fix.pkl")

# === 3. Streamlit Interface ===
st.set_page_config(page_title="Prediksi Petir", layout="centered")
st.title("🌩️ Prediksi Kejadian Petir Harian")
st.markdown("Silakan masukkan nilai-nilai parameter atmosfer di bawah ini:")

# === 4. Form Input User ===
with st.form("form_input"):
    col1, col2 = st.columns(2)
    with col1:
        hour = st.selectbox("Jam (UTC)", options=[00, 12])
        season = st.selectbox("Periode Musim (1=DJF, 2=MAM, 3=JJA, 4=SON)", options=[1, 2, 3, 4])
        month = st.slider("Bulan", 1, 12, 1)
        KI = st.number_input("KI Index", value=30.0)
        SWEAT = st.number_input("SWEAT Index", value=200.0)
    with col2:
        LI = st.number_input("Lifted Index", value=-2.0)
        CAPE = st.number_input("CAPE (J/kg)", value=1000.0)
        TTI = st.number_input("TTI Index", value=48.0)
        SI = st.number_input("Showalter Index", value=1.0)
        PW = st.number_input("Precipitable Water (mm)", value=40.0)

    submitted = st.form_submit_button("🔍 Prediksi")

# === 5. Prediksi Probabilitas & Klasifikasi ===
if submitted:
    cos_month = np.cos(2 * np.pi * month / 12)

    # Buat DataFrame input user sesuai urutan fitur saat training
    X_input = pd.DataFrame([{
        'hour': hour,
        'season': season,
        'KI': KI,
        'SWEAT': SWEAT,
        'LI': LI,
        'CAPE': CAPE,
        'TTI': TTI,
        'SI': SI,
        'PW': PW,
        'cos_month': cos_month
    }])[fitur_model]  # pastikan urutannya sesuai

    prob = model.predict_proba(X_input)[0, 1]
    klasifikasi = "⚡ Petir" if prob >= 0.35 else "✅ Non-Petir"

    st.metric("Probabilitas Petir", f"{prob:.2f}")
    st.success(f"Hasil Klasifikasi: {klasifikasi}")
