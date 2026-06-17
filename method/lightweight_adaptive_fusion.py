"""
LightweightAdaptiveFusion

Pipeline tiga langkah:
  A. Fast Color Correction   Bounded Gray World
  B. Detail Enhancement      CLAHE pada Luminance (YCbCr)
  C. Lightweight Fusion      Saliency + Brightness weight maps

Seluruh operasi berbasis numpy vectorization — zero pixel-level loop.

"""

from __future__ import annotations

import cv2
import numpy as np


class LightweightAdaptiveFusion:
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

    # Langkah A – Fast Color Correction (Bounded Gray World)
    def _step_a_color_correction(self, image: np.ndarray) -> np.ndarray:
        """
        Koreksi warna cepat via Bounded Gray World.

        """
        img_f = image.astype(np.float32)               # (H, W, 3) float32

        # Mean per saluran: axis=(0,1) → shape (3,) untuk B, G, R
        ch_mean = img_f.mean(axis=(0, 1))              # [μB, μG, μR]
        mean_gray = ch_mean.mean()                     # skalar

        gain = mean_gray / (ch_mean + self._eps)       # [gB, gG, gR]

        # Bounded: batasi gain merah (index 2 = R pada BGR)
        gain[2] = min(gain[2], self._max_red_gain)

        # Broadcast langsung: (H, W, 3) * (3,)
        corrected = np.clip(img_f * gain, 0.0, 255.0).astype(np.uint8)
        return corrected                               # Kandidat 1

    # Langkah B – Detail Enhancement via CLAHE pada Luminance
    def _step_b_detail_enhancement(self, cand1: np.ndarray) -> np.ndarray:
        """
        Perkuat detail tekstur dengan CLAHE pada saluran Y (YCbCr).

        """
        ycbcr = cv2.cvtColor(cand1, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycbcr)

        y_enhanced = self._clahe.apply(y)

        merged = cv2.merge((y_enhanced, cr, cb))
        cand2 = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)
        return cand2                                   # Kandidat 2

    # Langkah C – Lightweight Single-Level Fusion

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