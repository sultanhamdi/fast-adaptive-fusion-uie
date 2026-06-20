# Lightweight Adaptive Fusion for Underwater Image Enhancement

Proyek ini adalah implementasi dari Tugas Akhir di program studi Rekayasa Kecerdasan Artifisial, Institut Teknologi Sepuluh Nopember (ITS). Fokus utama penelitian ini adalah mencari *sweet spot* antara **kualitas visual tingkat tinggi** untuk akurasi visi komputer dan **efisiensi waktu komputasi** untuk implementasi *real-time* (>15 FPS) pada robotika bawah laut (AUV/ROV).

Proyek ini mengevaluasi metode usulan **Lightweight Adaptive Fusion (LAF)** dan membandingkannya dengan dua metode *baseline* (satu berfokus pada kecepatan, satu berfokus pada kualitas).

---

##  Tujuan Penelitian

1. **Efisiensi Komputasi:** Menghindari iterasi level-piksel (memanfaatkan vektorisasi NumPy secara penuh) dan operasi piramida fusi yang berat.
2. **Kualitas Visual:** Menjaga warna natural dan ketajaman fitur bawah laut tanpa over-enhancement.
3. **Trade-off Optimal:** Mengungguli kecepatan metode berkualitas tinggi dan melampaui kualitas metode yang sangat cepat.

---

##  Struktur Direktori

```text
fast-adaptive-fusion-uie/
│
├── method/                                  # Implementasi Algoritma Enhancement
│   ├── lightweight_adaptive_fusion.py       # (Usulan) Sweet-spot kecepatan & kualitas
│   ├── fast_underwater_enhancer.py          # (Baseline 1) Fokus kecepatan
│   └── adaptive_multichannel_enhancer.py    # (Baseline 2) Fokus kualitas tinggi
│
├── evaluation_metrics.py                    # Script penghitungan metrik (UIQM, UCIQE, NIQE, PSNR, SSIM)
├── comparison_notebook.ipynb                # Notebook evaluasi komparatif dataset
├── app.py                                   # Server Flask untuk Live Demo Web App
├── templates/
│   └── index.html                           # Antarmuka web app (Dark mode, slider komparasi)
├── dataset/                                 # Dataset UIEB (890 pasang gambar)
│   ├── raw/                                 # Citra bawah laut asli (degraded)
│   └── reference/                           # Citra referensi (ground truth)
└── README.md                                # Dokumentasi ini
```

---

##  Metode Enhancement

Proyek ini membandingkan 3 metode yang berbeda:

### 1. Proposed Method: Lightweight Adaptive Fusion (LAF)
Metode usulan yang dirancang untuk mencapai *sweet spot* antara kecepatan dan kualitas.
*   **Fase A:** Koreksi warna via *Bounded Gray World*.
*   **Fase B:** Peningkatan detail dengan *CLAHE* pada saluran Luminance (YCbCr).
*   **Fase C:** *Single-level pixel-wise fusion* menggunakan Normalisasi *Saliency* (LAB distance) dan *Brightness* (Gaussian) secara multiplikatif.
*   **Kecepatan:** ~12 - 15 FPS (Memenuhi syarat *real-time*).

### 2. Baseline 1: Fast Underwater Enhancer
Berdasarkan *Fast Underwater Image Enhancement for Real Time Applications*.
*   Menerapkan CLAHE secara independen per-saluran, diikuti CLAHE Luminance, dan normalisasi histogram *percentile*.
*   Sangat cepat, namun terkadang menghasilkan citra yang over-enhanced atau artifisial.

### 3. Baseline 2: Adaptive Multichannel Enhancer
Berdasarkan *Underwater Image Enhancement Based on Multichannel Adaptive Compensation*.
*   Menggunakan pemrosesan Grid GACC, *Local Shannon Entropy* (LEGW), dan fusi Piramida Laplacian-Gaussian 5-level.
*   Menghasilkan kualitas sangat baik dan warna natural, namun sangat berat secara komputasi (<1 FPS).

---

##  Metrik Evaluasi

`evaluation_metrics.py` menyediakan dukungan untuk evaluasi metrik *No-Reference* maupun *Reference-Based*:

| Metrik | Tipe | Keterangan |
|--------|------|------------|
| **UIQM** | No-Reference | *Underwater Image Quality Measure*. Menilai ketajaman, kontras, dan warna. (Higher is better) |
| **UCIQE** | No-Reference | *Underwater Color Image Quality Evaluation*. Fokus pada Chroma, Saturation, dan Luminance. (Higher is better) |
| **NIQE~** | No-Reference | Pendekatan statistik (*MSCN Variance/Skewness/Kurtosis*) untuk menilai *naturalness*. (Lower is better/more natural) |
| **PSNR** | Reference | *Peak Signal-to-Noise Ratio*. Akurasi piksel. (Higher is better) |
| **SSIM** | Reference | *Structural Similarity Index*. Akurasi struktural. Maks 1.0. (Higher is better) |

*Catatan: Metode berbasis kontras ekstrim cenderung mendominasi skor UIQM/UCIQE. Oleh karena itu, metrik penyeimbang seperti NIQE dan perbandingan visual tetap diperlukan.*

---

##  Live Demo Web App

Proyek ini menyertakan aplikasi web interaktif berbasis Flask untuk mendemonstrasikan metode **Lightweight Adaptive Fusion**.

Fitur Web App:
*   **Upload Gambar Interaktif:** Mendukung drag-and-drop.
*   **Slider Komparasi (*Before-After*):** Menggeser slider untuk melihat perbandingan citra secara visual.
*   **Penghitungan Metrik Real-time:** Menampilkan perbandingan nilai UIQM, UCIQE, NIQE, beserta waktu pemrosesan (ms) dan estimasi FPS.

### Cara Menjalankan Web App
1. Pastikan modul yang dibutuhkan sudah terinstal:
   ```bash
   pip install flask opencv-python numpy scipy pandas matplotlib
   ```
2. Jalankan aplikasi Flask:
   ```bash
   python app.py
   ```
3. Buka *browser* pada: `http://127.0.0.1:5000`

---

##  Penggunaan (*Usage*)

### Evaluasi Notebook
Buka `comparison_notebook.ipynb` dengan Jupyter Notebook atau ekstensi IDE untuk melihat:
*   Visualisasi hasil enhancement.
*   Tabel agregat UIQM, UCIQE, NIQE, dan FPS di seluruh dataset.
*   *Scatter plot* untuk visualisasi trade-off antara kecepatan dan kualitas (*Speed vs Quality*).

### Penggunaan Langsung via Python
Anda dapat mengimpor dan menggunakan metode *enhancement* ke dalam script Anda sendiri:

```python
import cv2
from method.lightweight_adaptive_fusion import LightweightAdaptiveFusion

# Baca citra (BGR format via OpenCV)
img = cv2.imread("Underwater Image/Biru/100_img_.png")

# Inisialisasi enhancer
enhancer = LightweightAdaptiveFusion()

# Lakukan enhancement
result = enhancer.enhance(img)

# Simpan hasil
cv2.imwrite("enhanced_result.png", result)
```

---

##  Lisensi dan Kredit
Proyek ini dikembangkan sebagai bagian dari Tugas Akhir di Institut Teknologi Sepuluh Nopember (ITS). Jika Anda menggunakan kode atau bagian dari proyek ini untuk penelitian, mohon sertakan sitasi yang sesuai.
