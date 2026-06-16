"""
FastUnderwaterEnhancer
======================
Implementasi algoritma dari paper:
  "Fast Underwater Image Enhancement for Real-Time Applications"

Pipeline terdiri dari 3 fase berurutan:
  1. Color Enhancement    CLAHE per-channel (B, G, R)
  2. Dehazing & Contrast  CLAHE pada saluran Luminance (YCbCr)
  3. White Balancing      Histogram Stretching (persentil 2–98)

Dependencies: opencv-python, numpy
"""

import cv2
import numpy as np


class FastUnderwaterEnhancer:
    """
    Enhancer citra bawah air berbasis tiga fase pemrosesan.

    """

    def __init__(
        self,
        clip_limit: float = 1.0,
        tile_grid: tuple[int, int] = (8, 8),
        low_pct: float = 2.0,
        high_pct: float = 98.0,
    ) -> None:
        self._clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid,
        )
        self._low_pct = low_pct
        self._high_pct = high_pct

    # Fase 1 – Color Enhancement
    def _color_enhancement(self, bgr: np.ndarray) -> np.ndarray:
        """
        Terapkan CLAHE secara independen pada setiap saluran warna BGR.

        """
        b, g, r = cv2.split(bgr)

        b_eq = self._clahe.apply(b)
        g_eq = self._clahe.apply(g)
        r_eq = self._clahe.apply(r)

        return cv2.merge((b_eq, g_eq, r_eq))

    # Fase 2 – Dehazing & Contrast
    def _dehazing_contrast(self, bgr: np.ndarray) -> np.ndarray:
        """
        Perkuat kontras luminance di ruang warna YCbCr.

        """
        ycbcr = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycbcr)

        y_eq = self._clahe.apply(y)

        ycbcr_eq = cv2.merge((y_eq, cr, cb))
        return cv2.cvtColor(ycbcr_eq, cv2.COLOR_YCrCb2BGR)

    # Fase 3 – White Balancing
    def _white_balancing(self, bgr: np.ndarray) -> np.ndarray:
        """
        Histogram stretching berbasis persentil untuk white balancing.

        """
        img_f = bgr.astype(np.float32)

        p_low = np.percentile(img_f, self._low_pct)
        p_high = np.percentile(img_f, self._high_pct)

        # Kliping + normalisasi Min-Max
        img_clipped = np.clip(img_f, p_low, p_high)
        img_stretched = (img_clipped - p_low) / (p_high - p_low + 1e-6) * 255.0

        return img_stretched.astype(np.uint8)

    # Public API
    def enhance(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                "Input harus berupa citra BGR 3-saluran (ndim=3, channels=3)."
            )

        stage1 = self._color_enhancement(image)
        stage2 = self._dehazing_contrast(stage1)
        stage3 = self._white_balancing(stage2)

        return stage3