"""白平衡：灰世界假设、白块（白点）估计。"""

from __future__ import annotations

import numpy as np

from docscan.core.types import Image, to_float


def gray_world(bgr: Image, *, strength: float = 1.0) -> Image:
    """灰世界白平衡：假设场景平均反射是灰色，把三通道均值拉平。

    out_c = I_c · (1 + strength·(m_ref/m_c − 1))，m_ref 取三通道均值的均值。

    已白平衡的图三通道均值相等 → 缩放因子为 1 → 恒等。
    """
    arr = to_float(bgr)
    if arr.ndim != 3:
        return arr
    means = arr.reshape(-1, 3).mean(axis=0)
    ref = float(means.mean())
    scales = np.array([ref / max(m, 1e-6) for m in means], dtype=np.float32)
    blended = 1.0 + strength * (scales - 1.0)
    return np.clip(arr * blended, 0.0, 1.0).astype(np.float32)


def white_patch(bgr: Image, *, percentile: float = 99.5, strength: float = 1.0) -> Image:
    """白块白平衡：把每通道最亮 p% 的均值当作「白」，据此拉平三通道。

    比灰世界更符合「纸是白的」这一文档先验，不易被大面积深色文字带偏。
    """
    arr = to_float(bgr)
    if arr.ndim != 3:
        return arr
    levels = []
    for c in range(3):
        ch = arr[..., c].reshape(-1)
        thr = np.percentile(ch, percentile)
        bright = ch[ch >= thr]
        levels.append(float(bright.mean()) if bright.size else 1.0)
    ref = max(max(levels), 1e-6)
    scales = np.array([ref / max(lv, 1e-6) for lv in levels], dtype=np.float32)
    blended = 1.0 + strength * (scales - 1.0)
    return np.clip(arr * blended, 0.0, 1.0).astype(np.float32)
