import pandas as pd
import numpy as np
import joblib
import streamlit as st

# Load model
model = joblib.load("model_petir_xgb.pkl")

st.set_page_config(page_title="Prediksi Petir", layout="centered")
st.title("🌩️ Prediksi Kejadian Petir Harian")
st.markdown("Masukkan parameter atmosfer berikut untuk memprediksi kejadian petir.")

# Form input
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        hour = st.selectbox("Jam (UTC)", [0, 12])
        season = st.selectbox("Musim", [1, 2, 3, 4])
        month = st.slider("Bulan", 1, 12, 1)
        KI = st.number_input("KI Index", value=30.0)
        SWEAT = st.number_input("SWEAT Index", value=200.0)
    with col2:
        LI = st.number_input("Lifted Index", value=-2.0)
        CAPE = st.number_input("CAPE (J/kg)", value=1000.0)
        TTI = st.number_input("TTI Index", value=48.0)
        SI = st.number_input("Showalter Index", value=1.0)
        PW = st.number_input("Precipitable Water (mm)", value=40.0)

    submit = st.form_submit_button("🔍 Prediksi")

if submit:
    cos_month = np.cos(2 * np.pi * month / 12)
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
    }])

    prob = model.predict_proba(X_input)[0, 1]
    klasifikasi = "⚡ Petir" if prob >= 0.35 else "✅ Non-Petir"

    st.metric("Probabilitas Petir", f"{prob:.2f}")
    st.success(f"Klasifikasi: {klasifikasi}")
