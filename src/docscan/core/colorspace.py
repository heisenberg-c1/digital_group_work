"""色彩空间封装：BGR ↔ YCbCr（BT.601），以及「只动亮度」的高阶函数。

设计动机：文档增强大多只想改亮度（去阴影、均衡、锐化），直接在 BGR 上做
会引入偏色。``apply_on_luma`` 把「转 YCbCr → 改 Y → 转回」收进一个高阶函数，
调用方拿到的是「给我一个 Y→Y 的函数」这样干净的接口。
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from docscan.core.types import Image, to_float

# BT.601 全量程（full-range）系数，Y/Cb/Cr 均在 0..255 尺度，Cb/Cr 以 128 为中心
_M_RGB2YCC = np.array(
    [
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312],
    ],
    dtype=np.float64,
)
_OFFSET = np.array([0.0, 128.0, 128.0], dtype=np.float64)


def _bgr_to_rgb(arr: Image) -> Image:
    return arr[..., ::-1] if arr.ndim == 3 else arr


def bgr_to_ycbcr(bgr: Image) -> tuple[Image, Image, Image]:
    """BGR float32[0,1] → (Y, Cb, Cr)，均为 float64、0..255 尺度。"""
    rgb = _bgr_to_rgb(to_float(bgr)).reshape(-1, 3).astype(np.float64) * 255.0
    ycc = rgb @ _M_RGB2YCC.T + _OFFSET
    target_shape = (*bgr.shape[:2], 3) if bgr.ndim == 3 else bgr.shape
    ycc = ycc.reshape(target_shape)
    y, cb, cr = ycc[..., 0], ycc[..., 1], ycc[..., 2]
    return y, cb, cr


def ycbcr_to_bgr(y: Image, cb: Image, cr: Image) -> Image:
    """(Y, Cb, Cr)（0..255 尺度）→ BGR float32[0,1]。"""
    ycc = np.stack([y, cb, cr], axis=-1).astype(np.float64)
    rgb = (ycc - _OFFSET) @ np.linalg.inv(_M_RGB2YCC).T / 255.0
    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb[..., ::-1].astype(np.float32)


def apply_on_luma(bgr: Image, func: Callable[[Image], Image]) -> Image:
    """只对亮度通道应用 ``func``，色度保持不变。

    Args:
        bgr: BGR float32[0,1] 或 uint8。
        func: Y(0..255 float64) → Y 的映射，输出会被裁剪回 0..255。

    Returns:
        BGR float32[0,1]。
    """
    y, cb, cr = bgr_to_ycbcr(bgr)
    y2 = np.clip(np.asarray(func(y), dtype=np.float64), 0.0, 255.0)
    return ycbcr_to_bgr(y2, cb, cr)
