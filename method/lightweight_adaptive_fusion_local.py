"""
LightweightAdaptiveFusion

Pipeline tiga langkah:
  A. Fast Color Correction   Bounded Gray World
  B. Detail Enhancement      CLAHE pada Luminance (YCbCr)
  C. Lightweight Fusion      Saliency + Brightness weight maps

Seluruh operasi berbasis numpy vectorization - zero pixel-level loop.

"""

from __future__ import annotations

import cv2
import numpy as np


class LightweightAdaptiveFusionLocal:
    """
    Real-time underwater image enhancer via lightweight adaptive fusion.

    """

    def __init__(
        self,
        max_red_gain: float = 2.5,
        clahe_clip: float = 2.0,
        clahe_tile: tuple[int, int] = (8, 8),
        sigma_bright: float = 0.25,
        eps: float = 1e-6,
    ) -> None:
        self._max_red_gain = max_red_gain
        self._eps = eps

        # Konstanta Gaussian brightness weight: −1/(2σ²)
        self._bright_coeff = -1.0 / (2.0 * sigma_bright ** 2)

        # CLAHE dibuat sekali, di-reuse di setiap frame
        self._clahe = cv2.createCLAHE(
            clipLimit=clahe_clip,
            tileGridSize=clahe_tile,
        )

    # Langkah A - Fast Local/Grid Color Correction (Grid Bounded Gray World)
    def _step_a_color_correction(self, image: np.ndarray) -> np.ndarray:
        """
        Koreksi warna adaptif lokal via Grid-based Bounded Gray World 
        dengan interpolasi bilinear untuk mencegah blocking artifacts.
        """
        h, w, c = image.shape
        img_f = image.astype(np.float32) / 255.0
        b, g, r = cv2.split(img_f)
        
        # 1. Kompensasi adaptif saluran merah (tetap dipertahankan)
        mean_b_glob = np.mean(b)
        mean_g_glob = np.mean(g)
        mean_r_glob = np.mean(r)
        r_compensated = r + (mean_g_glob - mean_r_glob) * (1.0 - r) * g
        fused_img = cv2.merge((b, g, r_compensated)) # Shape: (H, W, 3)

        # 2. Tentukan ukuran grid (misal 8x8 blok)
        grid_rows, grid_cols = 8, 8
        blk_h = h // grid_rows
        blk_w = w // grid_cols

        # Potong sedikit sisa gambar agar pas dibagi ukuran blok
        crop_h = grid_rows * blk_h
        crop_w = grid_cols * blk_w
        fused_cropped = fused_img[:crop_h, :crop_w, :]

        # 3. Vectorized Block Processing: Reshape untuk hitung mean per blok
        # Mengubah (crop_h, crop_w, 3) -> (grid_rows, blk_h, grid_cols, blk_w, 3)
        blocks = fused_cropped.reshape(grid_rows, blk_h, grid_cols, blk_w, 3)
        
        # Hitung rata-rata tiap channel di setiap blok (axis 1 dan 3 adalah blk_h dan blk_w)
        local_means = blocks.mean(axis=(1, 3)) # Shape: (grid_rows, grid_cols, 3)
        
        # Hitung rata-rata abu-abu lokal untuk setiap blok
        local_gray = local_means.mean(axis=2, keepdims=True) # Shape: (grid_rows, grid_cols, 1)

        # 4. Hitung Gain Lokal dan terapkan Bounded Constraint
        local_gains = local_gray / (local_means + self._eps) # Shape: (grid_rows, grid_cols, 3)
        
        # Batasi gain merah (index 2 pada channel BGR) secara lokal
        local_gains[:, :, 2] = np.minimum(local_gains[:, :, 2], self._max_red_gain)

        # 5. Upsample Gain Map ke ukuran asli menggunakan batuan cv2.resize (Bilinear)
        # Ini adalah kunci agar transisi koreksi warna antar grid tidak patah/kotak-kotak
        gain_map_resized = cv2.resize(local_gains, (w, h), interpolation=cv2.INTER_LINEAR)

        # 6. Aplikasikan gain map yang sudah halus ke gambar asli
        corrected = np.clip(img_f * gain_map_resized * 255.0, 0.0, 255.0).astype(np.uint8)
        return corrected                               # Kandidat 1                               # Kandidat 1

    # Langkah B - CLAHE di LAB space
    def _step_b_detail_enhancement(self, cand1: np.ndarray) -> np.ndarray:
        # LAB space
        lab = cv2.cvtColor(cand1, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Terapkan CLAHE pada L dengan clip limit yang sedikit lebih agresif untuk detail
        l_enhanced = self._clahe.apply(l)
        
        # Berikan sedikit Unsharp Masking untuk menaikkan nilai UIQM (aspek sharpness)
        cand2_lab = cv2.merge((l_enhanced, a, b))
        cand2_bgr = cv2.cvtColor(cand2_lab, cv2.COLOR_LAB2BGR)
        
        # Unsharp masking ringan
        blurred = cv2.GaussianBlur(cand2_bgr, (5, 5), 0)
        cand2 = cv2.addWeighted(cand2_bgr, 1.5, blurred, -0.5, 0)
        return cand2                                  # Kandidat 2

    # Langkah C - Lightweight Single-Level Fusion

    def _weight_saliency(self, bgr: np.ndarray) -> np.ndarray:
        """
        Normalized Saliency Weight Map berbasis jarak Euclidean kuadrat di LAB.

        Hasil dinormalisasi ke [0, 1] via min-max agar setara skala
        dengan brightness weight (juga [0, 1]).
        """
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        global_mean = lab.mean(axis=(0, 1))            # (3,) [μL, μA, μB]

        diff = lab - global_mean                       # broadcast (H,W,3)
        saliency = (diff ** 2).sum(axis=2)             # (H, W)

        # Normalisasi min-max ke [0, 1]
        s_min = saliency.min()
        s_max = saliency.max()
        saliency = (saliency - s_min) / (s_max - s_min + self._eps)

        return saliency

    def _weight_brightness(self, bgr: np.ndarray) -> np.ndarray:
        """
        Brightness weight map berbasis distribusi Gaussian.

        """
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray_norm = gray / 255.0                       # [0, 1]

        brightness = np.exp(self._bright_coeff * (gray_norm - 0.5) ** 2)
        return brightness                              # (H, W)

    def _fuse(
        self,
        cand1: np.ndarray,
        cand2: np.ndarray,
    ) -> np.ndarray:
        """
        Pixel-wise weighted fusion tanpa piramida Laplacian.

        """
        # --- Weight maps Kandidat 1 ---
        sal1 = self._weight_saliency(cand1)
        br1  = self._weight_brightness(cand1)
        raw1 = sal1 * br1                              # (H, W) multiplicative

        # --- Weight maps Kandidat 2 ---
        sal2 = self._weight_saliency(cand2)
        br2  = self._weight_brightness(cand2)
        raw2 = sal2 * br2                              # (H, W) multiplicative

        # --- Normalisasi: W1 + W2 = 1 per piksel ---
        total = raw1 + raw2 + self._eps
        w1 = (raw1 / total)[:, :, np.newaxis]         # (H, W, 1) untuk broadcast
        w2 = (raw2 / total)[:, :, np.newaxis]

        # --- Pixel-wise blend ---
        c1_f = cand1.astype(np.float32)
        c2_f = cand2.astype(np.float32)

        fused = c1_f * w1 + c2_f * w2                 # (H, W, 3)
        return np.clip(fused, 0.0, 255.0).astype(np.uint8)

    # Public API
    def enhance(self, image: np.ndarray) -> np.ndarray:
        """
        Jalankan pipeline fusi adaptif ringan pada citra bawah laut.

        """
        if (
            image is None
            or not isinstance(image, np.ndarray)
            or image.ndim != 3
            or image.shape[2] != 3
        ):
            raise ValueError(
                "Input harus berupa citra BGR 3-saluran uint8 (ndim=3, ch=3)."
            )

        cand1 = self._step_a_color_correction(image)
        cand2 = self._step_b_detail_enhancement(cand1)
        final = self._fuse(cand1, cand2)

        return final