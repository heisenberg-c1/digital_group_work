"""二值化：Otsu、自适应阈值、形态学清理。

约定：输入可以是 uint8 或 float32[0,1]；输出为 uint8 且取值只有 {0, 255}
（二值图是「掩膜」，不套用 float32[0,1] 的数值约定，故显式返回 uint8）。
"""

from __future__ import annotations

import cv2
import numpy as np

from docscan.core.types import Image, to_uint8


def to_gray(img: Image) -> Image:
    """三通道 BGR → 单通道灰度；已是灰度则原样返回。"""
    arr = to_uint8(img)
    if arr.ndim == 2:
        return arr
    return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)


def otsu(gray: Image, *, invert: bool = False) -> Image:
    """全局 Otsu 阈值二值化。

    Args:
        gray: 灰度图。
        invert: 为 True 时输出反相（黑底白字），默认白底黑字。

    Returns:
        uint8 {0, 255}。
    """
    g = to_gray(gray)
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, out = cv2.threshold(g, 0, 255, flag + cv2.THRESH_OTSU)
    return out


def adaptive(
    gray: Image,
    *,
    block_size: int = 31,
    c: float = 10.0,
    method: str = "gaussian",
    invert: bool = False,
) -> Image:
    """局部自适应阈值二值化。

    阈值 ``T(x,y) = mean_local(x,y) - C``（gaussian 版用高斯加权均值），
    适合光照不均的手机照片：全局阈值会在阴影边界留下黑块，局部阈值不会。

    Args:
        gray: 灰度图。
        block_size: 邻域边长，必须为正奇数且 > 1。
        c: 偏置常数，越大越「敢把像素判成黑」（保留笔画但易吃噪点）。
        method: ``"mean"`` 或 ``"gaussian"``。
        invert: 输出反相。

    Raises:
        ValueError: ``block_size`` 不是大于 1 的奇数，或 ``method`` 非法。
    """
    if block_size < 3 or block_size % 2 == 0:
        raise ValueError(f"block_size 必须是大于 1 的奇数，收到 {block_size}")
    if method not in ("mean", "gaussian"):
        raise ValueError(f"method 只能是 'mean' 或 'gaussian'，收到 {method!r}")

    g = to_gray(gray)
    flag = cv2.ADAPTIVE_THRESH_MEAN_C if method == "mean" else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    bin_flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    return cv2.adaptiveThreshold(g, 255, flag, bin_flag, block_size, c)


def clean_small_components(binary: Image, *, min_area: int = 8) -> Image:
    """按连通域面积过滤孤立噪点，保留面积 ≥ ``min_area`` 的前景块。

    以「墨迹=前景」处理：内部先反相，使文字为 255，再过滤。

    Args:
        binary: uint8 二值图（白底黑字或黑底白字皆可，按少数派自动判断）。
        min_area: 最小保留面积（像素数）。

    Returns:
        uint8 {0, 255}，与输入极性一致。
    """
    arr = to_uint8(binary)
    if arr.ndim == 3:
        arr = to_gray(arr)

    # 前景 = 较暗的一类（文字）。若前景像素反而更多，说明是黑底白字，先反相。
    fg = (arr == 0).astype(np.uint8)
    inverted = fg.mean() > 0.5
    work = (255 - arr) if inverted else arr

    num, labels, stats, _ = cv2.connectedComponentsWithStats(work, connectivity=8)
    out = np.zeros_like(work)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255

    return (255 - out) if inverted else out
