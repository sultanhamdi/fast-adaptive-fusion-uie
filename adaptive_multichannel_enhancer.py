"""
AdaptiveMultichannelEnhancer
Implementasi algoritma dari paper:
  "Underwater Image Enhancement Based on Multichannel Adaptive Compensation"

"""

import cv2
import numpy as np
from scipy.ndimage import uniform_filter
from typing import Tuple


class AdaptiveMultichannelEnhancer:
    """
    Underwater Image Enhancer menggunakan Multichannel Adaptive Compensation.
    """

    def __init__(
        self,
        grid_rows: int = 8,
        grid_cols: int = 8,
        attenuation: float = 0.5,
        entropy_block_size: int = 16,
        clahe_clip: float = 2.0,
        clahe_tile: Tuple[int, int] = (8, 8),
        pyramid_levels: int = 5,
        brightness_sigma: float = 0.25,
        brightness_target: float = 0.5,
    ):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.attenuation = attenuation          # α
        self.entropy_block_size = entropy_block_size
        self.pyramid_levels = pyramid_levels
        self.brightness_sigma = brightness_sigma
        self.brightness_target = brightness_target

        # Inisialisasi CLAHE\
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip,
            tileGridSize=clahe_tile
        )

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """
        Pipeline utama enhancement.
        """
        assert image.ndim == 3 and image.shape[2] == 3, \
            "Input harus citra BGR 3-channel (H x W x 3)"

        # --- Step 1: GACC ---
        gacc_img = self._gacc(image)

        # --- Step 2: Multichannel Generation ---
        ch1 = self._legw(gacc_img)          # Channel 1: LEGW
        ch2 = self._apply_clahe(ch1)        # Channel 2: CLAHE on ch1

        # --- Step 3: Weight Maps ---
        w1_sal, w1_bri = self._compute_weights(ch1)
        w2_sal, w2_bri = self._compute_weights(ch2)

        # Gabungkan saliency + brightness (element-wise product), lalu normalize
        raw_w1 = w1_sal * w1_bri
        raw_w2 = w2_sal * w2_bri
        total  = raw_w1 + raw_w2 + 1e-8     # epsilon untuk hindari div-by-zero
        norm_w1 = raw_w1 / total
        norm_w2 = raw_w2 / total

        # --- Step 4: Pyramid Fusion ---
        result = self._pyramid_fusion(ch1, ch2, norm_w1, norm_w2)

        return result

    # STEP 1 — Gridded Adaptive Channel Compensation (GACC)

    def _gacc(self, image: np.ndarray) -> np.ndarray:
        img_f = image.astype(np.float32)
        H, W, _ = img_f.shape
        output  = img_f.copy()

        row_edges = np.linspace(0, H, self.grid_rows + 1, dtype=int)
        col_edges = np.linspace(0, W, self.grid_cols + 1, dtype=int)

        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                r0, r1 = row_edges[i], row_edges[i + 1]
                c0, c1 = col_edges[j], col_edges[j + 1]

                patch = img_f[r0:r1, c0:c1, :]
                
                # Normalisasi mean ke rentang 0-1 untuk perhitungan bobot sesuai paper
                mu_norm = patch.mean(axis=(0, 1)) / 255.0  

                v_max_idx = int(np.argmax(mu_norm))
                v_max_norm = mu_norm[v_max_idx]

                for c in range(3):
                    if c != v_max_idx:
                        # Rumus adaptif (Paper Eq 2): omega * (Vmax - mean) * (1 - mean) * V_i
                        factor = self.attenuation * (v_max_norm - mu_norm[c]) * (1.0 - mu_norm[c])
                        output[r0:r1, c0:c1, c] = patch[:, :, c] + (factor * patch[:, :, c])

        return np.clip(output, 0, 255).astype(np.uint8)

    # STEP 2a — Channel 1: Local Entropy-Constrained Gray World (LEGW)

    def _legw(self, image: np.ndarray) -> np.ndarray:
        """
        Local Entropy-Constrained Gray World White Balance.
        """
        img_f = image.astype(np.float32) / 255.0   # normalize ke [0,1]
        H, W, _ = img_f.shape
        lam = 0.3   # faktor regulasi entropi (λ)

        # --- 1. Rata-rata reflektansi global β ---
        beta = img_f.mean()                          # scalar

        # --- 2. Entropy lokal per channel ---
        local_entropy = np.zeros_like(img_f)         # (H, W, 3)
        for c in range(3):
            local_entropy[:, :, c] = self._local_shannon_entropy(
                img_f[:, :, c],
                block_size=self.entropy_block_size
            )

        # --- 3. Gain awal Gray World: gain_c = β / μ_c ---
        mu_c = img_f.mean(axis=(0, 1))
        mu_c = np.maximum(mu_c, 1e-6)
        gain_base = beta / mu_c

        # --- 4. Gain yang dikoreksi entropi ---
        mean_entropy = local_entropy.mean(axis=(0, 1))
        mean_entropy  = np.maximum(mean_entropy, 1e-6)
        rel_entropy_diff = (local_entropy - mean_entropy[np.newaxis, np.newaxis, :]) \
                           / mean_entropy[np.newaxis, np.newaxis, :]
        gain_map = gain_base[np.newaxis, np.newaxis, :] * (1.0 + lam * rel_entropy_diff)

        # --- 5. Terapkan gain ---
        result = np.clip(img_f * gain_map, 0.0, 1.0)
        result = (result * 255).astype(np.uint8)
        return result

    def _local_shannon_entropy(
        self, channel: np.ndarray, block_size: int
    ) -> np.ndarray:
        """
        Hitung entropi Shannon lokal untuk satu channel grayscale.

        """
        n_bins  = 32
        bins    = np.linspace(0, 1, n_bins + 1)
        entropy_map = np.zeros_like(channel, dtype=np.float32)

        for b in range(n_bins):
            in_bin = ((channel >= bins[b]) & (channel < bins[b + 1])).astype(np.float32)

            count_b  = uniform_filter(in_bin,     size=block_size, mode='reflect')
            count_tot = uniform_filter(
                np.ones_like(channel), size=block_size, mode='reflect'
            )
            p_b = count_b / (count_tot + 1e-8)

            safe_p = np.where(p_b > 1e-8, p_b, 1.0)
            entropy_map -= np.where(p_b > 1e-8, p_b * np.log2(safe_p), 0.0)

        return entropy_map

    # STEP 2b — Channel 2: CLAHE

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Terapkan CLAHE (Contrast Limited Adaptive Histogram Equalization)
        pada Channel 1 (hasil LEGW) dengan memproses di ruang LAB.

        Hanya channel L yang di-equalize agar tidak mengubah warna.
        """
        lab   = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq  = self.clahe.apply(l)
        lab_eq = cv2.merge([l_eq, a, b])
        result = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
        return result

    # STEP 3 — Weight Maps

    def _compute_weights(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Hitung Saliency Weight dan Brightness Weight untuk satu channel citra.
        """
        sal = self._saliency_weight(image)
        bri = self._brightness_weight(image)
        return sal, bri

    def _saliency_weight(self, image: np.ndarray) -> np.ndarray:
        """
        Saliency Weight berbasis jarak Lab.

        """
        lab   = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        mu_L  = lab[:, :, 0].mean()
        mu_a  = lab[:, :, 1].mean()
        mu_b  = lab[:, :, 2].mean()
        w_sal = (
            (lab[:, :, 0] - mu_L) ** 2 +
            (lab[:, :, 1] - mu_a) ** 2 +
            (lab[:, :, 2] - mu_b) ** 2
        )
        return w_sal.astype(np.float32)

    def _brightness_weight(self, image: np.ndarray) -> np.ndarray:
        """
        Brightness Weight menggunakan distribusi Gaussian eksponensial.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        t      = self.brightness_target
        sigma  = self.brightness_sigma
        w_bri  = np.exp(-((gray - t) ** 2) / (2.0 * sigma ** 2))
        return w_bri.astype(np.float32)

    # STEP 4 — Laplacian–Gaussian Pyramid Fusion

    def _pyramid_fusion(
        self,
        ch1: np.ndarray,
        ch2: np.ndarray,
        w1:  np.ndarray,
        w2:  np.ndarray,
    ) -> np.ndarray:
        """
        Fusi dua citra menggunakan piramida Laplacian–Gaussian.
        """
        N = self.pyramid_levels

        img1 = ch1.astype(np.float32) / 255.0
        img2 = ch2.astype(np.float32) / 255.0

        # --- 1. Gaussian Pyramid untuk weight maps ---
        gp_w1 = self._gaussian_pyramid(w1, N)
        gp_w2 = self._gaussian_pyramid(w2, N)

        # --- 2. Laplacian Pyramid untuk citra ---
        lp_1  = self._laplacian_pyramid(img1, N)
        lp_2  = self._laplacian_pyramid(img2, N)

        # --- 3. Fusi tiap level piramida ---
        fused_pyramid = []
        for k in range(N + 1):
            w1_k = gp_w1[k][:, :, np.newaxis]
            w2_k = gp_w2[k][:, :, np.newaxis]

            ls_k = w1_k * lp_1[k] + w2_k * lp_2[k]
            fused_pyramid.append(ls_k)

        # --- 4. Collapse (rekonstruksi) piramida ---
        result = self._collapse_pyramid(fused_pyramid)
        result = np.clip(result, 0.0, 1.0)
        result = (result * 255).astype(np.uint8)
        return result

    def _gaussian_pyramid(
        self, img: np.ndarray, levels: int
    ) -> list:
        """
        Bangun Gaussian Pyramid sebanyak `levels` tingkat.
        """
        pyramid = [img.copy()]
        current = img.copy()
        for _ in range(levels):
            current = cv2.pyrDown(current)
            pyramid.append(current)
        return pyramid

    def _laplacian_pyramid(
        self, img: np.ndarray, levels: int
    ) -> list:
        """
        Bangun Laplacian Pyramid sebanyak `levels` tingkat.
        """
        gp    = self._gaussian_pyramid(img, levels)
        lp    = []

        for k in range(levels):
            h, w  = gp[k].shape[:2]
            up    = cv2.pyrUp(gp[k + 1], dstsize=(w, h))

            up    = up[:h, :w]
            lap_k = gp[k] - up
            lp.append(lap_k)

        lp.append(gp[levels])
        return lp

    def _collapse_pyramid(self, pyramid: list) -> np.ndarray:
        """
        Rekonstruksi citra dari piramida Laplacian (collapse).
        """
        N      = len(pyramid) - 1
        result = pyramid[N].copy()

        for k in range(N - 1, -1, -1):
            h, w   = pyramid[k].shape[:2]
            result = cv2.pyrUp(result, dstsize=(w, h))
            result = result[:h, :w]        # koreksi ukuran pyrUp
            result = result + pyramid[k]   # tambahkan detail band

        return result