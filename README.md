# 📌 Deteksi Tuberkulosis pada Citra Rontgen Dada menggunakan CNN

Proyek ini membahas implementasi **Convolutional Neural Network (CNN)** untuk klasifikasi penyakit **Tuberkulosis (TBC)** pada citra **rontgen dada**, menggunakan pendekatan **transfer learning** dan integrasi dengan aplikasi web berbasis **Streamlit**.

---

## 🩺 Latar Belakang

Tuberkulosis merupakan penyakit menular yang berdampak besar, khususnya di negara berkembang seperti Indonesia. Metode deteksi konvensional seperti tes dahak dan rontgen dada masih memiliki banyak keterbatasan. Oleh karena itu, solusi berbasis AI diharapkan mampu mempercepat, mempermurah, dan meningkatkan akurasi diagnosis TBC.

---

## 🎯 Tujuan

- Meningkatkan akurasi deteksi TBC menggunakan CNN.
- Menerapkan transfer learning dengan berbagai arsitektur pre-trained.
- Membangun aplikasi web untuk klasifikasi TBC secara otomatis.
- Mengevaluasi performa model berdasarkan metrik evaluasi.

---

## 🧪 Dataset

Dataset terdiri dari:
- **Citra X-ray normal**: 3500 gambar  
- **Citra X-ray TBC**: 700 gambar  
Total: 4200 citra  
(Sumber: Montgomery, Shenzhen, Belarus, RSNA pneumonia challenge)

---

## 🧠 Arsitektur CNN yang Digunakan

- [x] **EfficientNetV2B0**
- [x] **MobileNetV3**
- [x] **NASNetMobile**
- [x] **DenseNet121**

Semua model menggunakan **pre-trained weights dari ImageNet**.

---

## ⚙️ Metodologi

1. **Pre-processing**: resize, augmentasi, dan normalisasi gambar.
2. **Pelatihan**: menggunakan transfer learning dan fine-tuning.
3. **Evaluasi**: berdasarkan akurasi, precision, recall, F1-score, dan confusion matrix.
4. **Implementasi**: aplikasi web berbasis [Streamlit](https://streamlit.io/).

---

## 🏆 Hasil Terbaik

Model **MobileNetV3** menunjukkan performa terbaik:

| Metrik     | Nilai     |
|------------|-----------|
| Akurasi    | 98.10%    |
| Presisi    | 100%      |
| Recall     | 97.71%    |
| F1-Score   | 99.84%    |

---

## 🌐 Aplikasi Web

🖥️ Kunjungi aplikasi klasifikasi TBC secara online:  
👉 [https://tbclassification.streamlit.app/](https://tbclassification.streamlit.app/)

Pengguna dapat mengunggah citra rontgen dada dan memperoleh hasil klasifikasi (Normal/TBC) secara otomatis dan interaktif melalui antarmuka web sederhana.

---

## 💻 Tools & Teknologi

- Python
- TensorFlow & Keras
- Streamlit
- Visual Studio Code
- NumPy, Matplotlib, Scikit-learn

---
