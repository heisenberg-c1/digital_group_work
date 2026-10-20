"""退化合成器：为报告实验生成「干净图 + 退化图」严格配对的数据。

五种典型场景与课程要求的对应关系::

    motion    运动模糊   → 复原实验（维纳 K 扫描）
    defocus   散焦模糊   → 复原实验
    moire     摩尔纹     → 去网纹实验（理想 vs Butterworth）
    shadow    光照不均   → 光照归一实验
    cast      偏色       → 白平衡实验

GT（ground truth）就是传入的原图本身，保证逐像素配对。
"""

from __future__ import annotations

import cv2
import numpy as np

from docscan.core import restore
from docscan.core.types import Image, to_float

SCENARIOS = ("motion", "defocus", "moire", "shadow", "cast")


def _as_gray(img: Image) -> np.ndarray:
    arr = to_float(img)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    return arr.astype(np.float32)


def add_motion_blur(img: Image, *, length: int = 11, angle_deg: float = 0.0,
                    noise_sigma: float = 0.005, seed: int = 0) -> Image:
    """运动模糊 + 轻噪声。"""
    psf = restore.motion_psf(max(31, length * 2 | 1), length, angle_deg)
    arr = _as_gray(img).astype(np.float64)
    h = restore.freq_kernel(psf, arr.shape)
    out = np.fft.ifft2(np.fft.fft2(arr) * h).real
    rng = np.random.default_rng(seed)
    out = out + rng.normal(0.0, noise_sigma, out.shape)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def add_defocus(img: Image, *, radius: int = 5, noise_sigma: float = 0.005,
                seed: int = 1) -> Image:
    """散焦模糊 + 轻噪声。"""
    psf = restore.disk_psf(max(31, radius * 4 | 1), radius)
    arr = _as_gray(img).astype(np.float64)
    h = restore.freq_kernel(psf, arr.shape)
    out = np.fft.ifft2(np.fft.fft2(arr) * h).real
    rng = np.random.default_rng(seed)
    out = out + rng.normal(0.0, noise_sigma, out.shape)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def add_moire(img: Image, *, wavelength: int = 24, angle_deg: float = 30.0,
              amplitude: float = 0.18) -> Image:
    """叠加正弦摩尔纹（不越界合成，避免谐波失真）。"""
    arr = _as_gray(img)
    h, w = arr.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    theta = np.deg2rad(angle_deg)
    phase = 2.0 * np.pi * (xx * np.cos(theta) + yy * np.sin(theta)) / wavelength
    # 先把内容压进安全区间再叠加，保证 max(内容)+幅度 ≤ 1
    scaled = arr * (1.0 - amplitude)
    return np.clip(scaled + amplitude * np.sin(phase), 0.0, 1.0).astype(np.float32)


def add_shadow(img: Image, *, lo: float = 0.4, angle_deg: float = 0.0) -> Image:
    """乘性线性光照不均（模拟侧光拍摄）。"""
    arr = _as_gray(img)
    h, w = arr.shape
    theta = np.deg2rad(angle_deg)
    ramp = (np.cos(theta) * np.linspace(-0.5, 0.5, w)[None, :]
            + np.sin(theta) * np.linspace(-0.5, 0.5, h)[:, None])
    field = lo + (1.0 - lo) * (ramp - ramp.min()) / max(float(np.ptp(ramp)), 1e-9)
    return np.clip(arr * field, 0.0, 1.0).astype(np.float32)


def add_color_cast(img: Image, *, gains: tuple[float, float, float] = (1.25, 1.0, 0.75)) -> Image:
    """BGR 偏色（模拟白平衡失误的暖光/冷光场景）。仅对彩色图有意义。"""
    arr = to_float(img)
    if arr.ndim != 3:
        return arr
    return np.clip(arr * np.array(gains, dtype=np.float32), 0.0, 1.0).astype(np.float32)


def synthesize(img: Image, kind: str, **kwargs) -> tuple[Image, Image]:
    """按场景名生成 (退化图, GT) 配对。

    GT 与退化图严格同域：色彩域场景（cast）返回彩色 GT，其余返回灰度 GT。
    """
    gt = to_float(img).copy() if kind == "cast" else _as_gray(img)
    table = {
        "motion": add_motion_blur,
        "defocus": add_defocus,
        "moire": add_moire,
        "shadow": add_shadow,
        "cast": add_color_cast,
    }
    if kind not in table:
        raise ValueError(f"未知场景 {kind!r}，可选：{SCENARIOS}")
    return table[kind](img, **kwargs), gt
