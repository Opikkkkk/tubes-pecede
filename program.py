import sys
import os
import cv2
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QComboBox
from PyQt5.QtGui import QPixmap

# ====== Ekstraksi Fitur Lebih Stabil ======
def extract_features(image):
    def preprocess(image):
        return cv2.resize(image, (200, 200))

    def hsv_features(image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mean = cv2.mean(hsv)[:3]
        hist = []
        for i in range(3):
            h = cv2.calcHist([hsv], [i], None, [16], [0, 256])
            h = cv2.normalize(h, h).flatten()
            hist.extend(h)
        return list(mean) + list(hist)

    def hu_moments(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        moments = cv2.moments(gray)
        hu = cv2.HuMoments(moments).flatten()
        return -np.sign(hu) * np.log10(np.abs(hu) + 1e-6)

    def shape_stats(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            _, _, w, h = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            aspect_ratio = float(w) / (h + 1e-6)
            return [aspect_ratio, area, perimeter]
        return [0, 0, 0]

    def lbp_texture(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(gray, 8, 1, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)
        return hist.tolist()

    def glcm_texture(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]
        return [contrast, energy, homogeneity, correlation]

    image = preprocess(image)
    features = hsv_features(image) + hu_moments(image).tolist() + shape_stats(image) + lbp_texture(image) + glcm_texture(image)
    return np.array(features).reshape(1, -1)

# ====== Training Model Lebih Akurat ======
def train_models():
    print("🧠 Melatih model dengan dataset...")
    dataset_path = "dataset"
    X, y = [], []

    for label in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, label)
        if os.path.isdir(class_path):
            for file in os.listdir(class_path):
                path = os.path.join(class_path, file)
                try:
                    img = cv2.imread(path)
                    if img is not None:
                        img = cv2.resize(img, (200, 200))
                        fitur = extract_features(img)
                        X.append(fitur.flatten())
                        y.append(label)
                except:
                    print(f"⚠️  Gagal membaca: {path}")

    X = np.array(X)
    y = np.array(y)

    if len(set(y)) < 2:
        print("❌ Gagal: Dataset harus memiliki minimal 2 kelas.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # KNN
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    knn.fit(X_train, y_train)
    joblib.dump(knn, "model_knn.pkl")

    # SVM
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
    svm.fit(X_train, y_train)
    joblib.dump(svm, "model_svm.pkl")

    # Evaluasi
    print("\n📊 Evaluasi KNN")
    print(classification_report(y_test, knn.predict(X_test)))
    print("Akurasi:", accuracy_score(y_test, knn.predict(X_test)))

    print("\n📊 Evaluasi SVM")
    print(classification_report(y_test, svm.predict(X_test)))
    print("Akurasi:", accuracy_score(y_test, svm.predict(X_test)))

# ====== GUI ======
class SampahClassifier(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Klasifikasi Sampah")
        self.setGeometry(100, 100, 400, 400)

        self.layout = QVBoxLayout()
        self.label = QLabel("Pilih gambar untuk diklasifikasikan:")
        self.img_label = QLabel()
        self.result_label = QLabel()
        self.model_selector = QComboBox()
        self.model_selector.addItems(["KNN", "SVM"])

        self.button = QPushButton("Pilih Gambar")
        self.button.clicked.connect(self.klasifikasikan_gambar)

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.img_label)
        self.layout.addWidget(QLabel("Pilih Model:"))
        self.layout.addWidget(self.model_selector)
        self.layout.addWidget(self.button)
        self.layout.addWidget(self.result_label)
        self.setLayout(self.layout)

        self.model_knn = joblib.load("model_knn.pkl")
        self.model_svm = joblib.load("model_svm.pkl")

    def klasifikasikan_gambar(self):
        path, _ = QFileDialog.getOpenFileName(self, "Pilih Gambar", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            image = cv2.imread(path)
            image = cv2.resize(image, (200, 200))
            fitur = extract_features(image)

            model_type = self.model_selector.currentText()
            model = self.model_knn if model_type == "KNN" else self.model_svm
            pred = model.predict(fitur)[0]

            self.img_label.setPixmap(QPixmap(path).scaledToWidth(300))
            self.result_label.setText(f"Hasil klasifikasi: {pred.upper()}")

            output_folder = os.path.join("hasil_klasifikasi", pred)
            os.makedirs(output_folder, exist_ok=True)
            filename = os.path.basename(path)
            cv2.imwrite(os.path.join(output_folder, filename), image)

# ====== RUN ======
if __name__ == "__main__":
    if not os.path.exists("model_knn.pkl") or not os.path.exists("model_svm.pkl"):
        train_models()
    app = QApplication(sys.argv)
    window = SampahClassifier()
    window.show()
    sys.exit(app.exec_())
