# 📘 TUGAS BESAR PENGOLAHAN CITRA DIGITAL

### 👥 **Kelompok:**
- **Ketua:** Muhammad Taufiq Rahman Hakim (152023119)  
- **Anggota:** Muhammad Reza Faishal (152023113)

---

## 📝 **Deskripsi Program**

Program ini dirancang untuk melakukan **ekstraksi fitur dari citra RGB** yang berisi gambar-gambar objek **sampah**. Fitur yang diekstraksi bertujuan untuk mendukung proses pengenalan dan klasifikasi sampah ke dalam kategori tertentu.

---

## 🎯 **Tujuan Program**

Untuk **membedakan objek sampah** menjadi beberapa kategori menggunakan teknik **ekstraksi fitur citra** seperti warna, bentuk, dan tekstur. Fitur-fitur ini akan digunakan sebagai dasar untuk sistem klasifikasi berbasis machine learning atau analisis lanjutan.

---

## 🧩 **Jenis Fitur yang Diekstraksi**

1. ### 🎨 **Fitur Warna**
   - Histogram warna (RGB/HSV)
   - Rata-rata warna (mean RGB atau HSV)
   - Deviasi standar warna

2. ### 📐 **Fitur Bentuk**
   - Kontur objek
   - Luas (area)
   - Rasio aspek (aspect ratio)
   - Ekstent (extent)
   - Eccentricity

3. ### 🌾 **Fitur Tekstur**
   - GLCM (Gray Level Co-occurrence Matrix)
     - Kontras
     - Homogenitas
     - Entropi
   - LBP (Local Binary Pattern)

---

## 📁 **Dataset**
- Minimal **60 citra RGB**
- Dibagi menjadi beberapa kategori objek sampah (misalnya: plastik, kertas, organik)
- Format file: `.jpg`, `.png`, atau format gambar umum lainnya

---

## ⚙️ **Alur Kerja Program**

1. Membaca citra dari dataset
2. Melakukan preprocessing (resize, grayscale, dll.)
3. Mengekstrak fitur berdasarkan kategori:
   - Warna → untuk objek kertas
   - Bentuk → untuk objek organik
   - Tekstur → untuk objek plastik
4. Menyimpan hasil ekstraksi ke file `.csv` atau `.npy` untuk keperluan pelatihan model klasifikasi

---

## 📌 **Catatan**
- Program dibagi menjadi 3 bagian utama (warna, bentuk, dan tekstur).
- Masing-masing bagian menangani kategori objek sampah yang berbeda.
- Dapat diperluas ke sistem klasifikasi otomatis menggunakan SVM, KNN, atau CNN.

---

