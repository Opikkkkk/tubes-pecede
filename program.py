import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QTextEdit
)
from PyQt5.QtGui import QPixmap, QImage
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from PyQt5.QtCore import Qt
import os

def ekstrak_fitur_warna(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mean = cv2.mean(hsv)[:3]
    hist = []
    for i in range(3):
        h = cv2.calcHist([hsv], [i], None, [8], [0, 256])
        h = cv2.normalize(h, h).flatten()
        hist.extend(h)
    return mean, hist

def ekstrak_fitur_bentuk(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hu = cv2.HuMoments(cv2.moments(gray)).flatten()
    if contours:
        c = max(contours, key=cv2.contourArea)
        _, _, w, h = cv2.boundingRect(c)
        aspect_ratio = float(w) / h
        n_contour = len(contours)
        img_contour = image.copy()
        cv2.drawContours(img_contour, [c], -1, (0,255,0), 2)
        return img_contour, hu, aspect_ratio, n_contour
    return image, hu, 0, 0

def ekstrak_fitur_tekstur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    glcm_vals = [contrast, homogeneity, energy, correlation]
    lbp_img = np.uint8(255 * (lbp - lbp.min()) / (lbp.max() - lbp.min()))
    lbp_img = cv2.cvtColor(lbp_img, cv2.COLOR_GRAY2BGR)
    return lbp_img, hist, glcm_vals

def deteksi_kategori(mean, hist, hu, aspect_ratio, n_contour, hist_lbp, glcm_vals):
    # Aturan gabungan, threshold bisa Anda tuning dari data nyata
    # Paper: S rendah, histogram warna dominan, tekstur rendah, aspect ratio ~1, kontur sedikit
    if mean[1] < 50 and max(hist) > 0.4 and np.mean(hist_lbp) < 0.15 and n_contour < 4 and 0.8 < aspect_ratio < 1.2:
        return "PAPER"
    # Organic: Banyak kontur, aspect ratio besar, Hu Moments tertentu, tekstur sedang
    elif n_contour > 5 and aspect_ratio > 1.3 and np.mean(hist_lbp) < 0.25:
        return "ORGANIC"
    # Plastic: Tekstur tinggi, GLCM kontras tinggi, histogram warna tidak dominan
    elif np.mean(hist_lbp) > 0.15 or glcm_vals[0] > 0.1:
        return "PLASTIC"
    else:
        # fallback, misal jika tidak yakin
        return "TIDAK TERDETEKSI"

def uji_akurasi_dataset(dataset_dir):
    kategori_list = ["plastic", "paper", "organic"]
    benar = 0
    total = 0
    output_dir = "gambar_ekstraksi"
    os.makedirs(output_dir, exist_ok=True)
    for kategori in kategori_list:
        folder = os.path.join(dataset_dir, kategori)
        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            img = cv2.imread(fpath)
            if img is not None:
                img = cv2.resize(img, (200, 200))
                mean, hist = ekstrak_fitur_warna(img)
                img_contour, hu, aspect_ratio, n_contour = ekstrak_fitur_bentuk(img)
                lbp_img, hist_lbp, glcm_vals = ekstrak_fitur_tekstur(img)
                pred = deteksi_kategori(mean, hist, hu, aspect_ratio, n_contour, hist_lbp, glcm_vals)
                total += 1
                if pred.lower() == kategori:
                    benar += 1
                print(f"{fname}: label={kategori}, prediksi={pred}")
                # Simpan gambar hasil ekstraksi ke subfolder sesuai label
                base = os.path.splitext(fname)[0]
                label_dir = os.path.join(output_dir, kategori)
                os.makedirs(label_dir, exist_ok=True)
                cv2.imwrite(os.path.join(label_dir, f"{base}_asli_{kategori}_{pred}.png"), img)
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                hsv_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                cv2.imwrite(os.path.join(label_dir, f"{base}_hsv_{kategori}_{pred}.png"), hsv_vis)
                cv2.imwrite(os.path.join(label_dir, f"{base}_kontur_{kategori}_{pred}.png"), img_contour)
                cv2.imwrite(os.path.join(label_dir, f"{base}_lbp_{kategori}_{pred}.png"), lbp_img)
    akurasi = benar / total * 100 if total > 0 else 0
    print(f"\nAkurasi deteksi kategori dari dataset: {benar}/{total} = {akurasi:.2f}%")

class DeteksiSampahUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Deteksi & Ekstraksi Fitur Sampah")
        self.setGeometry(100, 100, 900, 700)
        self.layout = QVBoxLayout()

        self.btn = QPushButton("Pilih Gambar")
        self.btn.clicked.connect(self.pilih_gambar)
        self.layout.addWidget(self.btn)

        # Layout untuk gambar-gambar
        self.img_layout = QHBoxLayout()
        self.img_label_asli = QLabel("Citra Asli")
        self.img_label_warna = QLabel("Metode Warna (HSV)")
        self.img_label_bentuk = QLabel("Metode Bentuk (Kontur)")
        self.img_label_tekstur = QLabel("Metode Tekstur (LBP)")
        for lbl in [self.img_label_asli, self.img_label_warna, self.img_label_bentuk, self.img_label_tekstur]:
            lbl.setFixedSize(200, 200)
            lbl.setStyleSheet("border: 1px solid black;")
        self.img_layout.addWidget(self.img_label_asli)
        self.img_layout.addWidget(self.img_label_warna)
        self.img_layout.addWidget(self.img_label_bentuk)
        self.img_layout.addWidget(self.img_label_tekstur)
        self.layout.addLayout(self.img_layout)

        # Label keterangan di bawah gambar
        self.caption_layout = QHBoxLayout()
        self.caption_asli = QLabel("Citra Asli")
        self.caption_warna = QLabel("Metode Warna (HSV)")
        self.caption_bentuk = QLabel("Metode Bentuk (Kontur)")
        self.caption_tekstur = QLabel("Metode Tekstur (LBP)")
        for cap in [self.caption_asli, self.caption_warna, self.caption_bentuk, self.caption_tekstur]:
            cap.setAlignment(Qt.AlignCenter)
            cap.setFixedWidth(200)
        self.caption_layout.addWidget(self.caption_asli)
        self.caption_layout.addWidget(self.caption_warna)
        self.caption_layout.addWidget(self.caption_bentuk)
        self.caption_layout.addWidget(self.caption_tekstur)
        self.layout.addLayout(self.caption_layout)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.layout.addWidget(self.result_text)

        self.setLayout(self.layout)

    def pilih_gambar(self):
        path, _ = QFileDialog.getOpenFileName(self, "Pilih Gambar", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            img = cv2.imread(path)
            img = cv2.resize(img, (200, 200))

            # Ekstraksi semua fitur
            mean, hist = ekstrak_fitur_warna(img)
            img_contour, hu, aspect_ratio, n_contour = ekstrak_fitur_bentuk(img)
            lbp_img, hist_lbp, glcm_vals = ekstrak_fitur_tekstur(img)
            kategori = deteksi_kategori(mean, hist, hu, aspect_ratio, n_contour, hist_lbp, glcm_vals)

            # Gambar asli
            img_asli = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.set_img_to_label(self.img_label_asli, img_asli)

            # Gambar HSV (untuk visualisasi, tampilkan channel V sebagai grayscale)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hsv_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            self.set_img_to_label(self.img_label_warna, hsv_vis)

            # Gambar kontur
            img_contour_vis = cv2.cvtColor(img_contour, cv2.COLOR_BGR2RGB)
            self.set_img_to_label(self.img_label_bentuk, img_contour_vis)

            # Gambar LBP
            lbp_vis = cv2.cvtColor(lbp_img, cv2.COLOR_BGR2RGB)
            self.set_img_to_label(self.img_label_tekstur, lbp_vis)

            # Output per metode
            teks = (
                f"<span style='font-size:18pt; font-weight:bold; color:#2a7ae2;'>Kategori terdeteksi: {kategori}</span><br><br>"
                f"<b>=== METODE WARNA ===</b><br>"
                f"- Rata-rata HSV: H={mean[0]:.2f}, S={mean[1]:.2f}, V={mean[2]:.2f}<br>"
                f"- Histogram warna (8 nilai pertama): {', '.join(f'{h:.3f}' for h in hist[:8])}<br>"
                f"<br><b>=== METODE BENTUK ===</b><br>"
                f"- Hu Moments: {', '.join(f'{v:.4e}' for v in hu)}<br>"
                f"- Aspect Ratio: {aspect_ratio:.2f}<br>"
                f"- Jumlah kontur: {n_contour}<br>"
                f"<br><b>=== METODE TEKSTUR ===</b><br>"
                f"- Histogram LBP: {', '.join(f'{h:.3f}' for h in hist_lbp)}<br>"
                f"- GLCM:<br>"
                f"   Kontras: {glcm_vals[0]:.4f}<br>"
                f"   Homogenitas: {glcm_vals[1]:.4f}<br>"
                f"   Energi: {glcm_vals[2]:.4f}<br>"
                f"   Korelasi: {glcm_vals[3]:.4f}<br>"
            )
            self.result_text.setHtml(teks)

    def set_img_to_label(self, label, img):
        h, w, ch = img.shape
        bytes_per_line = ch * w
        qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qimg).scaled(200, 200))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DeteksiSampahUI()
    window.show()
    # Untuk uji akurasi, jalankan ini:
    uji_akurasi_dataset("dataset")
    sys.exit(app.exec_())