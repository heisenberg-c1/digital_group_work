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


def sauvola(
    gray: Image,
    *,
    window: int = 25,
    k: float = 0.2,
    r: float = 128.0,
) -> Image:
    """Sauvola 局部二值化（积分图加速），公式与 skimage 一致。

    阈值::

        T(x,y) = m(x,y) · [1 + k · (σ(x,y)/R − 1)]

    相比自适应均值法，Sauvola 用局部标准差 σ 动态修正偏置：在接近纯白的
    背景区（σ≈0）阈值贴近均值本身，不会把纸面纹理判成文字；在笔画附近
    （σ 大）阈值下压，保住细笔画。R 是动态范围归一因子（8-bit 惯例取 128）。

    Args:
        gray: 灰度图。
        window: 邻域边长（奇数，≥3）。
        k: 敏感度参数，文献常用 0.2~0.5。
        r: 动态范围参数，与 skimage 默认一致取 128 以便对拍。

    Returns:
        uint8 {0, 255}，白底黑字。

    Raises:
        ValueError: window 不是奇数或 < 3。
    """
    if window < 3 or window % 2 == 0:
        raise ValueError(f"window 必须是 ≥3 的奇数，收到 {window}")

    f = to_gray(gray).astype(np.float64)
    h, w = f.shape
    half = window // 2

    # reflect 填充后做积分图：边界窗口大小恒定，且与 skimage 的
    # uniform_filter(mode='reflect') 语义一致，保证 oracle 对拍成立
    padded = np.pad(f, half, mode="reflect")
    integral = cv2.integral(padded)          # (h+2half+1, w+2half+1)
    integral_sq = cv2.integral(padded * padded)

    ys = np.arange(h)
    xs = np.arange(w)
    y0 = ys                       # 填充后窗口不再需要裁边
    y1 = ys + window
    x0 = xs
    x1 = xs + window

    s = (
        integral[np.ix_(y1, x1)]
        - integral[np.ix_(y0, x1)]
        - integral[np.ix_(y1, x0)]
        + integral[np.ix_(y0, x0)]
    )
    s2 = (
        integral_sq[np.ix_(y1, x1)]
        - integral_sq[np.ix_(y0, x1)]
        - integral_sq[np.ix_(y1, x0)]
        + integral_sq[np.ix_(y0, x0)]
    )
    n = float(window * window)
    mean = s / n
    var = np.maximum(s2 / n - mean * mean, 0.0)
    std = np.sqrt(var)

    threshold = mean * (1.0 + k * (std / r - 1.0))
    return ((f > threshold) * 255).astype(np.uint8)


def clean_small_components(binary: Image, *, min_area: int = 8) -> Image:
    """按连通域面积过滤孤立噪点，保留面积 ≥ ``min_area`` 的前景块。

    以「墨迹=前景」处理：内部统一为黑字白底 → 反相使墨迹为 255 →
    过滤连通域 → 按输入极性还原。

    Args:
        binary: uint8 二值图（白底黑字或黑底白字皆可，自动判断极性）。
        min_area: 最小保留面积（像素数）。

    Returns:
        uint8 {0, 255}，与输入极性一致。
    """
    arr = to_uint8(binary)
    if arr.ndim == 3:
        arr = to_gray(arr)

    # 极性统一：暗像素占多数说明是「白字黑底」，先反相成标准黑字白底
    inverted_input = (arr < 128).mean() > 0.5
    if inverted_input:
        arr = 255 - arr

    ink = 255 - arr  # 墨迹 = 255
    num, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    kept = np.zeros_like(ink)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            kept[labels == i] = 255

    out = 255 - kept  # 回到白底黑字
    if inverted_input:
        out = 255 - out
    return out
