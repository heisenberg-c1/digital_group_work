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


# ---------------------------------------------------------------------------
# HSV：与 YCbCr 并列的第二套「亮度/色度分离」通道
#
# YCbCr 的优点是线性、可逆、与显示链路一致；HSV 的优点是把「颜色类别（H）」
# 与「纯度（S）」「明度（V）」彻底解耦，判断「纸面是否发黄」「是否是彩色桌面」
# 这类语义问题时比 YCbCr 直观。两套都提供，才能对同一幅图回答不同问题。
#
# 这里用 float 域自行实现（不走 cv2 的 8-bit HSV）：OpenCV 的 H 只有 0..179，
# 且 8-bit 往返会引入 >1 级的量化误差，无法满足「色度逐元素不变」的断言。
# ---------------------------------------------------------------------------

_EPS = 1e-12


def bgr_to_hsv(bgr: Image) -> tuple[Image, Image, Image]:
    """BGR float32[0,1] → (H, S, V)。

    Returns:
        H: 色相，度，``[0, 360)``；S: 饱和度 ``[0, 1]``；V: 明度 ``[0, 255]``
        （与 ``bgr_to_ycbcr`` 的 0..255 尺度保持一致，便于共用 ``apply_on_*``）。
    """
    arr = to_float(bgr)
    if arr.ndim != 3:
        raise ValueError("bgr_to_hsv 需要三通道 BGR 图像")
    rgb = arr[..., ::-1].astype(np.float64)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    maxc = rgb.max(axis=-1)
    minc = rgb.min(axis=-1)
    delta = maxc - minc
    v = maxc
    s = np.where(maxc > _EPS, delta / np.maximum(maxc, _EPS), 0.0)

    h = np.zeros_like(maxc)
    nz = delta > _EPS
    is_r = nz & (maxc == r)
    is_g = nz & (maxc == g) & ~is_r
    is_b = nz & (maxc == b) & ~is_r & ~is_g
    h[is_r] = ((g[is_r] - b[is_r]) / delta[is_r]) % 6.0
    h[is_g] = (b[is_g] - r[is_g]) / delta[is_g] + 2.0
    h[is_b] = (r[is_b] - g[is_b]) / delta[is_b] + 4.0
    h *= 60.0
    return h, np.clip(s, 0.0, 1.0), np.clip(v, 0.0, 1.0) * 255.0


def hsv_to_bgr(h: Image, s: Image, v: Image) -> Image:
    """(H 度, S 0..1, V 0..255) → BGR float32[0,1]。"""
    vv = np.clip(np.asarray(v, dtype=np.float64), 0.0, 255.0) / 255.0
    ss = np.clip(np.asarray(s, dtype=np.float64), 0.0, 1.0)
    hh = np.mod(np.asarray(h, dtype=np.float64), 360.0) / 60.0

    c = vv * ss
    x = c * (1.0 - np.abs(np.mod(hh, 2.0) - 1.0))
    m = vv - c
    idx = np.floor(hh).astype(np.int64) % 6

    z = np.zeros_like(c)
    r = np.select([idx == 0, idx == 1, idx == 2, idx == 3, idx == 4], [c, x, z, z, x], c)
    g = np.select([idx == 0, idx == 1, idx == 2, idx == 3, idx == 4], [x, c, c, x, z], z)
    b = np.select([idx == 0, idx == 1, idx == 2, idx == 3, idx == 4], [z, z, x, c, c], x)

    rgb = np.stack([r + m, g + m, b + m], axis=-1)
    return np.clip(rgb[..., ::-1], 0.0, 1.0).astype(np.float32)


def apply_on_value(bgr: Image, func: Callable[[Image], Image]) -> Image:
    """只对明度通道 V 应用 ``func``，色相与饱和度保持不变。

    Args:
        bgr: BGR float32[0,1] 或 uint8。
        func: V(0..255 float64) → V 的映射，输出会被裁剪回 0..255。

    Returns:
        BGR float32[0,1]。
    """
    h, s, v = bgr_to_hsv(bgr)
    v2 = np.clip(np.asarray(func(v), dtype=np.float64), 0.0, 255.0)
    return hsv_to_bgr(h, s, v2)
