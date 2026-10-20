"""空间域增强：伽马（LUT）、对比度拉伸、直方图均衡、CLAHE。"""

from __future__ import annotations

import cv2
import numpy as np

from docscan.core.colorspace import apply_on_luma
from docscan.core.types import Image, to_float, to_uint8


def gamma_lut(gamma: float = 1.0) -> np.ndarray:
    """构建伽马校正查找表（uint8，256 项）。

    变换::

        out = 255 · (in / 255)^(1/γ)

    γ > 1 提亮暗部，γ < 1 压暗。γ = 1 时为严格恒等映射。

    Raises:
        ValueError: gamma 非正。
    """
    if gamma <= 0:
        raise ValueError(f"gamma 必须为正数，收到 {gamma}")
    lut = np.rint(255.0 * (np.arange(256, dtype=np.float64) / 255.0) ** (1.0 / gamma))
    return np.clip(lut, 0, 255).astype(np.uint8)


def apply_gamma(img: Image, gamma: float = 1.0) -> Image:
    """伽马校正。uint8 走 LUT（快），float 直接做幂运算。"""
    if img.dtype == np.uint8:
        lut = gamma_lut(gamma)
        if img.ndim == 2:
            return cv2.LUT(img, lut)
        return cv2.merge([cv2.LUT(img[..., i], lut) for i in range(img.shape[2])])
    arr = to_float(img)
    return np.clip(arr ** (1.0 / gamma), 0.0, 1.0).astype(np.float32)


def contrast_stretch(
    img: Image, *, low_pct: float = 1.0, high_pct: float = 99.0
) -> Image:
    """百分位对比度拉伸：把 [p_low, p_high] 线性映射到 [0, 1]。

    常数图（lo == hi）不除零，直接返回原值域内的常数。
    """
    arr = to_float(img)
    lo = float(np.percentile(arr, low_pct))
    hi = float(np.percentile(arr, high_pct))
    if hi - lo < 1e-8:
        return arr.astype(np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def hist_equalize(img: Image) -> Image:
    """直方图均衡，只在亮度通道执行（避免彩色图偏色）。"""
    arr = to_uint8(to_float(img)) if img.dtype != np.uint8 else img
    if arr.ndim == 2:
        return cv2.equalizeHist(arr)
    out = apply_on_luma(
        to_float(arr),
        lambda y: cv2.equalizeHist(np.rint(y).astype(np.uint8)).astype(np.float64),
    )
    return out


def clahe(img: Image, *, clip_limit: float = 2.0, grid_size: int = 8) -> Image:
    """限制对比度自适应直方图均衡（CLAHE），同样只在亮度通道执行。

    Args:
        clip_limit: 削波阈值，越小越接近恒等（→0 时输出≈输入）。
        grid_size: 网格划分数。
    """
    arr = to_uint8(to_float(img)) if img.dtype != np.uint8 else img
    engine = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))

    def _apply(y: np.ndarray) -> np.ndarray:
        return engine.apply(np.rint(y).astype(np.uint8)).astype(np.float64)

    if arr.ndim == 2:
        return engine.apply(arr)
    return apply_on_luma(to_float(arr), _apply)
