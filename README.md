# 🌩️ Aplikasi Prediksi Petir

Aplikasi ini menggunakan model Machine Learning (XGBoost) untuk memprediksi kemungkinan terjadinya petir berdasarkan parameter atmosfer dan waktu (jam, musim, bulan).

## 📥 Input dari Pengguna
- Jam (00 atau 12 UTC)
- Musim (DJF, MAM, JJA, SON)
- Bulan
- Parameter atmosfer:
  - KI Index
  - SWEAT Index
  - Lifted Index (LI)
  - CAPE (Convective Available Potential Energy)
  - TTI (Total Totals Index)
  - SI (Showalter Index)
  - PW (Precipitable Water)

## 🔎 Output
- Probabilitas petir
- Klasifikasi biner (⚡ Petir / ✅ Non-Petir)

## 🚀 Cara Menjalankan

### Lokal
Pastikan sudah menginstal streamlit:

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Online (Streamlit Cloud)
1. Upload project ini ke GitHub
2. Buka [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Klik **New app** dan pilih repositori ini
4. Pastikan file utama adalah `app.py`
5. Klik **Deploy**

---
Model telah dilatih dengan data meteorologi dan fitur musiman menggunakan `cos_month`, yang terbukti meningkatkan akurasi.