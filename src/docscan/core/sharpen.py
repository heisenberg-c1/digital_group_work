"""锐化：非锐化掩模（USM）。"""

from __future__ import annotations

import cv2
import numpy as np

from docscan.core.types import Image, to_float, to_uint8


def unsharp_mask(
    img: Image, *, radius: float = 2.0, amount: float = 1.0, threshold: int = 0
) -> Image:
    """非锐化掩模::

        out = I + amount · (I − G_σ(I))

    只放大比 ``threshold``（0-255 尺度）更大的局部差异，避免放大平坦区噪点。

    Args:
        img: BGR 或灰度。
        radius: 高斯模糊 σ（像素）。
        amount: 锐化强度，0 = 严格恒等。
        threshold: 差异低于该值（灰度级）的像素不锐化。

    Returns:
        与输入同 dtype 的图像。
    """
    if radius <= 0:
        raise ValueError(f"radius 必须为正，收到 {radius}")
    if amount < 0:
        raise ValueError(f"amount 不能为负，收到 {amount}")

    k = int(2 * round(3 * radius) + 1) | 1
    if img.dtype == np.uint8:
        arr = img.astype(np.float64)
    else:
        arr = to_float(img).astype(np.float64) * 255.0

    channels = [arr] if arr.ndim == 2 else [arr[..., i] for i in range(arr.shape[2])]
    out_channels = []
    for ch in channels:
        blur = cv2.GaussianBlur(ch, (k, k), radius)
        diff = ch - blur  # 0..255 尺度
        if threshold > 0:
            mask = np.abs(diff) >= threshold
            diff = diff * mask
        out_channels.append(ch + amount * diff)

    out = out_channels[0] if arr.ndim == 2 else np.stack(out_channels, axis=-1)
    if img.dtype == np.uint8:
        return np.rint(np.clip(out, 0, 255)).astype(np.uint8)
    return np.clip(out / 255.0, 0.0, 1.0).astype(np.float32)


def ensure_uint8(img: Image) -> Image:
    """便捷转换（供 pipeline 使用）。"""
    return to_uint8(img)
