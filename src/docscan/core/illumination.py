"""光照归一：背景估计去阴影、Retinex。

算法在 float32[0,1] 域工作。

去阴影原理：纸面是大面积亮区，文字是高频细节。用「形态学闭运算 + 大核高斯」
估计光照场（闭运算把暗色笔画填掉、保留纸面亮度），再用原图除以光照场，
把亮度变化「约掉」—— 除法对乘性光照是精确逆操作，对加性阴影是良好近似。

Retinex（SSR/MSR）：log(I) - log(I * G) 的多尺度加权和，把「反射率 × 光照」
分解成反射率估计，输出经 min-max 归一保证值域。
"""

from __future__ import annotations

import cv2
import numpy as np

from docscan.core.types import Image, to_float, to_uint8


def _channels(arr: Image) -> list[np.ndarray]:
    return [arr] if arr.ndim == 2 else [arr[..., i] for i in range(arr.shape[2])]


def _merge(channels: list[np.ndarray], like: Image) -> Image:
    if like.ndim == 2:
        return channels[0]
    return np.stack(channels, axis=-1)


def _odd_kernel(sigma: float) -> int:
    """由 sigma 推导奇数核边长（±3σ 覆盖 99.7% 能量）。"""
    return int(2 * round(max(sigma, 1.0)) + 1) | 1


def estimate_background(img: Image, *, sigma: float = 30.0) -> Image:
    """估计光照场（低频亮度分布），输出与输入同形状的 float32[0,1]。

    步骤：每通道先做大核闭运算（把暗色笔画填成纸面亮度），
    再做大核高斯平滑（消除闭运算留下的块状痕迹）。
    """
    arr = to_float(img)
    k = _odd_kernel(sigma)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    ks = _odd_kernel(sigma / 2.0)
    bgs = []
    for ch in _channels(arr):
        closed = cv2.morphologyEx(
            to_uint8(ch), cv2.MORPH_CLOSE, kernel, iterations=1
        )
        bgs.append(to_float(cv2.GaussianBlur(closed, (ks, ks), max(sigma / 2.0, 1.0))))
    return _merge(bgs, arr)


def remove_shading(
    img: Image, *, sigma: float = 30.0, percentile: float = 97.0, strength: float = 1.0
) -> Image:
    """背景估计相除去阴影。

    公式::

        out = clip( (I / BG) / percentile(I / BG, p) , 0, 1)

    除法把乘性光照归一；再用高分位数把纸面亮度拉回 ~1.0，
    避免整图偏暗或偏亮。

    Args:
        img: BGR 或灰度图。
        sigma: 光照场平滑尺度，越大保留的光照起伏越少。
        percentile: 归一化分位数，纸面占比极高时可调低到 90。
        strength: 0=原图，1=完全校正；供 UI 做前后对比。

    Returns:
        float32[0,1] 图像（uint8 输入也返回 float，便于与后续模块衔接）。
    """
    arr = to_float(img)
    bg = estimate_background(arr, sigma=sigma)
    ratio = arr / np.maximum(bg, 1e-4)
    ref = max(float(np.percentile(ratio, percentile)), 1e-4)
    corrected = np.clip(ratio / ref, 0.0, 1.0)
    return np.clip(arr * (1 - strength) + corrected * strength, 0.0, 1.0)


def retinex(
    img: Image,
    *,
    sigmas: tuple[float, ...] = (15.0, 80.0, 250.0),
    weights: tuple[float, ...] | None = None,
) -> Image:
    """多尺度 Retinex（MSR）。

    公式::

        R = Σ w_i · [ log(I + ε) − log(I * G_i + ε) ] / Σ w_i

    多尺度兼顾动态范围压缩（小 σ）与色彩保真（大 σ）。

    Args:
        sigmas: 各尺度高斯标准差。
        weights: 对应权重，默认等权。

    Returns:
        float32[0,1]，逐通道 min-max 归一，无 NaN、无负值。
    """
    arr = to_float(img)
    if weights is None:
        weights = tuple([1.0] * len(sigmas))
    if len(weights) != len(sigmas):
        raise ValueError("sigmas 与 weights 长度不一致")

    eps = 1.0 / 255.0
    log_i = np.log(arr + eps)
    acc = np.zeros_like(arr)
    for w, sigma in zip(weights, sigmas, strict=True):
        blur = cv2.GaussianBlur(arr, (_odd_kernel(sigma), _odd_kernel(sigma)), sigma)
        acc = acc + w * (log_i - np.log(blur + eps))
    acc /= sum(weights)

    out = np.empty_like(acc)
    for i, ch in enumerate(_channels(acc)):
        lo, hi = float(ch.min()), float(ch.max())
        out_ch = (ch - lo) / (hi - lo) if hi - lo > 1e-8 else np.zeros_like(ch)
        if arr.ndim == 2:
            out = out_ch
        else:
            out[..., i] = out_ch
    return np.clip(out, 0.0, 1.0).astype(np.float32)
