"""噪声抑制：中值滤波与双边滤波。

文档场景的噪声有两类：传感器噪点（高斯/泊松，颗粒状）和压缩块效应。
二值化会把前者放大成孤立黑点，所以去噪应当放在二值化之前、锐化之前
（锐化会放大噪声，顺序颠倒等于白做）。

两种算子各有取舍：

* 中值滤波：对椒盐噪声（孤立黑白点）近乎最优，边缘保持好，但会磨掉
  细笔画（3×3 已足够，5×5 起明显糊字）；
* 双边滤波：按灰度差加权，保边能力更强，适合高斯噪声，代价是慢 5~10 倍。

本模块保持输入 dtype（uint8 进 uint8 出，float 进 float 出），
以便在 uint8 边界层与 float 算法层之间自由插入。
"""

from __future__ import annotations

import cv2
import numpy as np

from docscan.core.types import Image, to_float


def _odd(ksize: int) -> int:
    """把窗口边长规整为 ≥3 的奇数（cv2 的要求）。"""
    return max(3, int(ksize) | 1)


def median(img: Image, *, ksize: int = 3) -> Image:
    """中值滤波，保持输入 dtype。

    Args:
        img: BGR 或灰度图（uint8 / float32[0,1]）。
        ksize: 窗口边长，自动规整为奇数。

    Returns:
        与输入同 dtype、同形状的图像。
    """
    k = _odd(ksize)
    if img.dtype == np.uint8:
        return cv2.medianBlur(img, k)
    arr = to_float(img)
    u8 = np.rint(arr * 255.0).astype(np.uint8)
    out = cv2.medianBlur(u8, k)
    return (out.astype(np.float32) / 255.0).astype(np.float32)


def bilateral(
    img: Image,
    *,
    d: int = 5,
    sigma_color: float = 0.05,
    sigma_space: float = 5.0,
) -> Image:
    """双边滤波，保持输入 dtype。

    Args:
        img: BGR 或灰度图（uint8 / float32[0,1]）。
        d: 邻域直径（≤0 时由 sigma_space 推导）。
        sigma_color: 色彩（灰度差）σ。uint8 域按 255 尺度解释，
            float 域按 [0,1] 尺度解释——调用方传 0..1 即可。
        sigma_space: 空间 σ（像素）。

    Returns:
        与输入同 dtype、同形状的图像。
    """
    if img.dtype == np.uint8:
        return cv2.bilateralFilter(img, d, sigma_color * 255.0, sigma_space)
    arr = to_float(img)
    out = cv2.bilateralFilter(arr.astype(np.float32), d, float(sigma_color), sigma_space)
    return np.clip(out, 0.0, 1.0).astype(np.float32)
