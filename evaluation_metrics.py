from __future__ import annotations

import time
import functools
from typing import Callable, Any

import cv2
import numpy as np
from scipy.ndimage import convolve

_C1 = 0.0282
_C2 = 0.2953
_C3 = 3.5753

def _to_float32(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float32)

def _bgr_to_lab(image: np.ndarray) -> np.ndarray:
    # Konversi ke float [0, 1] agar OpenCV menghasilkan skala CIELab standar
    # (L: 0-100, a: -127 hingga 127, b: -127 hingga 127)
    img_float = image.astype(np.float32) / 255.0
    return cv2.cvtColor(img_float, cv2.COLOR_BGR2LAB)

def _uicm(rg: np.ndarray, yb: np.ndarray) -> float:
    mu_rg, sigma_rg = rg.mean(), rg.std()
    mu_yb, sigma_yb = yb.mean(), yb.std()

    uicm = (
        -0.0268 * np.sqrt(mu_rg ** 2 + mu_yb ** 2)
        + 0.1586 * np.sqrt(sigma_rg ** 2 + sigma_yb ** 2)
    )
    return float(uicm)

def _uism(gray: np.ndarray) -> float:
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_map = np.hypot(sobel_x, sobel_y)

    h, w = edge_map.shape
    block = 8

    h_trim = (h // block) * block
    w_trim = (w // block) * block
    edge_map = edge_map[:h_trim, :w_trim]

    k1 = h_trim // block
    k2 = w_trim // block

    blocks = edge_map.reshape(k1, block, k2, block)
    block_max = blocks.max(axis=(1, 3))
    block_min = blocks.min(axis=(1, 3))

    eps = 1e-8
    eme = (2.0 / (k1 * k2)) * np.sum(
        np.log((block_max + eps) / (block_min + eps))
    )
    return float(eme)

def _uiconm(gray: np.ndarray) -> float:
    h, w = gray.shape
    block = 8

    h_trim = (h // block) * block
    w_trim = (w // block) * block
    img_trim = gray[:h_trim, :w_trim]

    k1 = h_trim // block
    k2 = w_trim // block

    blocks = img_trim.reshape(k1, block, k2, block)
    block_max = blocks.max(axis=(1, 3))
    block_min = blocks.min(axis=(1, 3))

    eps = 1e-8
    amee = (1.0 / (k1 * k2)) * np.sum(
        np.abs(np.log((block_max + eps) / (block_min + eps)))
    )
    return float(amee)

def compute_uiqm(image: np.ndarray) -> dict[str, float]:
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input harus berupa citra BGR 3-saluran uint8.")

    img_f = _to_float32(image)
    b, g, r = img_f[:, :, 0], img_f[:, :, 1], img_f[:, :, 2]

    rg = r - g
    yb = 0.5 * (r + g) - b

    gray = 0.299 * r + 0.587 * g + 0.114 * b

    uicm  = _uicm(rg, yb)
    uism  = _uism(gray)
    uiconm = _uiconm(gray)

    uiqm = _C1 * uicm + _C2 * uism + _C3 * uiconm

    return {
        "uiqm"   : round(uiqm, 6),
        "uicm"   : round(uicm, 6),
        "uism"   : round(uism, 6),
        "uiconm" : round(uiconm, 6),
    }

def compute_uciqe(image: np.ndarray) -> dict[str, float]:
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input harus berupa citra BGR 3-saluran uint8.")

    lab = _bgr_to_lab(image)
    L = lab[:, :, 0]
    a = lab[:, :, 1]
    b_ch = lab[:, :, 2]

    chroma = np.hypot(a, b_ch)
    sigma_c = float(chroma.std())
    con_l = float(np.percentile(L, 98) - np.percentile(L, 2))

    # Cegah ledakan nilai saturasi dengan menetapkan batas bawah luminance 1%
    L_safe = np.maximum(L, 1.0) 
    saturation = chroma / L_safe
    mu_s = float(saturation.mean())

    uciqe = 0.4680 * sigma_c + 0.2745 * con_l + 0.2576 * mu_s

    return {
        "uciqe"           : round(uciqe, 6),
        "sigma_chroma"    : round(sigma_c, 6),
        "contrast_luma"   : round(con_l, 6),
        "mean_saturation" : round(mu_s, 6),
    }

# ═══════════════════════════════════════════════════════════════
# Reference-Based Metrics (memerlukan reference image)
# ═══════════════════════════════════════════════════════════════

def compute_psnr(
    enhanced: np.ndarray, reference: np.ndarray
) -> dict[str, float]:
    if enhanced.shape != reference.shape:
        raise ValueError("Ukuran citra enhanced dan reference harus sama.")

    mse = float(np.mean(
        (enhanced.astype(np.float64) - reference.astype(np.float64)) ** 2
    ))
    if mse < 1e-10:
        return {"psnr": float("inf"), "mse": 0.0}

    psnr = 10.0 * np.log10(255.0 ** 2 / mse)
    return {
        "psnr" : round(psnr, 4),
        "mse"  : round(mse, 4),
    }


def compute_ssim(
    enhanced: np.ndarray,
    reference: np.ndarray,
    win_size: int = 11,
) -> dict[str, float]:
    if enhanced.shape != reference.shape:
        raise ValueError("Ukuran citra enhanced dan reference harus sama.")

    C1 = (0.01 * 255.0) ** 2
    C2 = (0.03 * 255.0) ** 2
    sigma_g = 1.5

    img1 = enhanced.astype(np.float64)
    img2 = reference.astype(np.float64)

    ssim_channels: list[float] = []
    for c in range(img1.shape[2]):
        c1 = img1[:, :, c]
        c2 = img2[:, :, c]

        mu1 = cv2.GaussianBlur(c1, (win_size, win_size), sigma_g)
        mu2 = cv2.GaussianBlur(c2, (win_size, win_size), sigma_g)

        mu1_sq  = mu1 * mu1
        mu2_sq  = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.GaussianBlur(c1 * c1, (win_size, win_size), sigma_g) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(c2 * c2, (win_size, win_size), sigma_g) - mu2_sq
        sigma12   = cv2.GaussianBlur(c1 * c2, (win_size, win_size), sigma_g) - mu1_mu2

        num   = (2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)
        denom = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

        ssim_map = num / denom
        ssim_channels.append(float(ssim_map.mean()))

    return {"ssim": round(float(np.mean(ssim_channels)), 6)}


# ═══════════════════════════════════════════════════════════════
# No-Reference Naturalness Metric (NIQE-like approximation)
# ═══════════════════════════════════════════════════════════════

def compute_niqe_approx(image: np.ndarray) -> dict[str, float]:
    from scipy.stats import kurtosis as _kurt, skew as _skew

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)

    ks = 7
    sigma_g = ks / 6.0
    mu = cv2.GaussianBlur(gray, (ks, ks), sigma_g)
    var_local = cv2.GaussianBlur(gray ** 2, (ks, ks), sigma_g) - mu ** 2
    sigma_local = np.sqrt(np.maximum(var_local, 0.0)) + 1.0
    mscn = (gray - mu) / sigma_local

    flat = mscn.flatten()
    v = float(flat.var())
    s = float(_skew(flat))
    k = float(_kurt(flat, fisher=True))

    deviation = float(np.sqrt((v - 1.0) ** 2 + s ** 2 + (k / 3.0) ** 2))

    return {
        "niqe_approx"   : round(deviation, 6),
        "mscn_variance"  : round(v, 6),
        "mscn_skewness"  : round(s, 6),
        "mscn_kurtosis"  : round(k, 6),
    }


def timing_wrapper(func: Callable[..., Any]) -> Callable[..., tuple[Any, dict]]:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> tuple[Any, dict]:
        t_start = time.perf_counter()
        result  = func(*args, **kwargs)
        t_end   = time.perf_counter()

        elapsed_ms = (t_end - t_start) * 1_000.0
        fps        = 1_000.0 / (elapsed_ms + 1e-9)

        timing_info = {
            "elapsed_ms" : round(elapsed_ms, 4),
            "fps"        : round(fps, 2),
            "func"       : func.__name__,
        }
        return result, timing_info

    return wrapper

def benchmark(
    func: Callable[..., Any],
    image: np.ndarray,
    n_runs: int = 50,
) -> dict[str, float]:
    times: list[float] = []

    _ = func(image)

    for _ in range(n_runs):
        t0 = time.perf_counter()
        func(image)
        times.append((time.perf_counter() - t0) * 1_000.0)

    arr = np.array(times)
    mean_ms = float(arr.mean())

    return {
        "mean_ms"  : round(mean_ms, 4),
        "std_ms"   : round(float(arr.std()), 4),
        "min_ms"   : round(float(arr.min()), 4),
        "max_ms"   : round(float(arr.max()), 4),
        "mean_fps" : round(1_000.0 / (mean_ms + 1e-9), 2),
        "n_runs"   : n_runs,
    }
