# 📘 SISTEM DETEKSI DAN KLASIFIKASI SAMPAH
## Pengolahan Citra Digital CC

### 👥 **Kelompok:**
- **Ketua:** Muhammad Taufiq Rahman Hakim (152023119)  
- **Anggota:** Muhammad Reza Faishal (152023113)

---

## 📝 **Deskripsi Program**
Program ini merupakan sistem lengkap untuk **deteksi dan klasifikasi sampah** menggunakan teknik **pengolahan citra digital**. Sistem terdiri dari 3 program utama yang saling terintegrasi untuk melakukan ekstraksi fitur, pelatihan model machine learning, dan interface pengguna untuk klasifikasi real-time.

---

## 🎯 **Tujuan Program**
Untuk **mengidentifikasi dan mengklasifikasikan objek sampah** ke dalam 3 kategori utama (**Plastik**, **Kertas**, **Organik**) menggunakan:
- Ekstraksi fitur citra (warna, bentuk, tekstur)
- Algoritma machine learning (KNN, SVM)
- Interface pengguna yang user-friendly

---

## 🧩 **Fitur yang Diekstraksi**

### 1. 🎨 **Fitur Warna (untuk Kertas)**
- Konversi ke ruang warna HSV
- Rata-rata nilai H, S, V
- Histogram warna 8 bin per channel

### 2. 📐 **Fitur Bentuk (untuk Organik)**
- Hu Moments (7 momen invariant)
- Aspect ratio
- Jumlah kontur
- Deteksi kontur dengan thresholding

### 3. 🌾 **Fitur Tekstur (untuk Plastik)**
- **LBP (Local Binary Pattern)**
  - Pattern uniform dengan P=8, R=1
  - Histogram 9 bin
- **GLCM (Gray Level Co-occurrence Matrix)**
  - Kontras
  - Homogenitas
  - Energi
  - Korelasi

---

## 📁 **Dataset**
- **60+ citra RGB** sampah
- 3 kategori: `plastic/`, `paper/`, `organic/`
- Format: `.jpg`, `.png`, `.jpeg`
- Resolusi: Otomatis resize ke 200x200 pixel

---

---

## 📊 **Alur Kerja Sistem**

1. **Preprocessing**
   - Resize gambar → 200x200
   - Konversi color space sesuai kebutuhan

2. **Ekstraksi Fitur**
   - Kertas → Fitur warna (HSV)
   - Organik → Fitur bentuk (kontur)
   - Plastik → Fitur tekstur (LBP + GLCM)

3. **Klasifikasi**
   - Rule-based system (GUI)
   - Machine learning (KNN/SVM)

4. **Output**
   - Kategori sampah terdeteksi
   - Visualisasi proses ekstraksi
   - Metrics evaluasi

---

## 📈 **Hasil Evaluasi**
- **Rule-based System:** Akurasi berdasarkan threshold manual
- **KNN Classifier:** Evaluasi dengan k=3 neighbors
- **SVM Classifier:** Linear kernel performance
- **Metrics:** Accuracy, Precision, Recall, F1-score

---

**© 2024 - Tugas Besar Pengolahan Citra Digital**
