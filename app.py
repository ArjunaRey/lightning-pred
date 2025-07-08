
# === 1. Import Library ===
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, roc_auc_score, precision_recall_curve
)
from sklearn.model_selection import TimeSeriesSplit
from imblearn.combine import SMOTEENN
from xgboost import XGBClassifier
import streamlit as st

# === 2. Latih Model Sekali di Awal ===
df = pd.read_excel("dataset_final_dengan_fitur_labilitas.xlsx")
df = df.dropna()
df['slot'] = pd.to_datetime(df['slot'])
df['tahun'] = df['slot'].dt.year

# Fitur musiman dan waktu
df['month'] = df['slot'].dt.month
df['season'] = ((df['month'] % 12 + 3) // 3)
df['hour'] = df['slot'].dt.hour
df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)

# Fitur dan Target
fitur = ['hour', 'season', 'KI', 'SWEAT', 'LI', 'CAPE', 'TTI', 'SI', 'PW', 'cos_month']
X = df[fitur]
y = df['label_petir_biner']

# Split Train-Test
train_df = df[df['tahun'] < 2024]
X_train = train_df[fitur]
y_train = train_df['label_petir_biner']

# SMOTE-ENN dan Latih Model
smote_enn = SMOTEENN(random_state=42)
X_train_res, y_train_res = smote_enn.fit_resample(X_train, y_train)
model = XGBClassifier(
    n_estimators=113,
    max_depth=6,
    learning_rate=0.0447,
    min_child_weight=9,
    subsample=0.501,
    colsample_bytree=0.917,
    gamma=0.346,
    reg_lambda=1.307,
    reg_alpha=0.902,
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train_res, y_train_res)

# === 3. Aplikasi Streamlit ===
st.set_page_config(page_title="Prediksi Petir", layout="centered")
st.title("🌩️ Prediksi Kejadian Petir Harian")
st.markdown("Silakan masukkan nilai-nilai parameter atmosfer di bawah ini.")

# Input dari user
with st.form("form_input"):
    col1, col2 = st.columns(2)
    with col1:
        hour = st.selectbox("Jam (UTC)", options=[0, 12])
        season = st.selectbox("Musim (1=DJF, 2=MAM, 3=JJA, 4=SON)", options=[1, 2, 3, 4])
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

if submitted:
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
