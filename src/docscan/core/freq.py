"""频域去网纹：幅度谱、周期峰检测、理想 / Butterworth 陷波器。

摩尔纹的来源：拍屏幕 / 扫描印刷品时，传感器网格与显示网格干涉，
产生规律的正弦条纹，能量集中在幅度谱的一对对称亮点上。
陷波器的思路：在频域把这对点挖掉，再逆变换回去。

关键设计：
* 峰检测在 log 幅度谱上做（动态范围大），迭代取局部极大并抑制邻域；
* Butterworth 陷波器 H = 1/(1+(D0/D)^(2n))，过渡带平缓，空域振铃远小于
  理想陷波（矩形截断）——这是报告要求④的核心对比实验；
* 掩膜按 (u,v) 与 (−u,−v) 成对挖除，保证实数输出。
"""

from __future__ import annotations

import cv2
import numpy as np

from docscan.core.types import Image, to_float


def fft2_shifted(gray: Image) -> np.ndarray:
    """灰度图 → 中心化的二维傅里叶谱（复数，DC 在画布中心）。"""
    arr = to_float(gray)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    return np.fft.fftshift(np.fft.fft2(arr.astype(np.float64)))


def magnitude_spectrum(gray: Image) -> np.ndarray:
    """对数幅度谱（已中心化、归一到 [0,1]），供显示与找峰。"""
    f = fft2_shifted(gray)
    mag = np.log1p(np.abs(f))
    peak = mag.max()
    return mag / peak if peak > 0 else mag


def find_periodic_peaks(
    spec_log: np.ndarray,
    *,
    top_n: int = 6,
    min_distance: int | None = None,
    dc_radius: int | None = None,
    threshold_rel: float = 0.2,
    min_prominence: float = 0.5,
    max_period_px: float = 64.0,
) -> list[tuple[int, int]]:
    """在（已中心化的）对数幅度谱上找周期峰。

    做法：屏蔽 DC 邻域（把真实内容的高能低频排除）→ 迭代取全局最大 →
    抑制其邻域 → 重复。返回峰相对 DC 的偏移 ``(du, dv)``，
    约定 ``u`` 沿 x（列）方向、``v`` 沿 y（行）方向。

    DC 屏蔽半径是最容易踩的坑：摩尔纹的峰半径约为 ``N/λ``（N 为画布边长，
    λ 为条纹波长），短周期网纹（λ ≤ 30 px @1024）只有 30~40 bin，
    一旦 DC 半径大于它，真实峰会被当成「低频内容」屏蔽掉，``argmax``
    只能取到次强的伪峰——指标显示「能量下降 45 dB」，视觉上条纹却纹丝不动。
    因此半径按**待检最大周期**推算 ``0.5·N/λ_max``，而不是取一个固定比例：
    固定比例在长周期一端会再次踩坑（1024² 取 1/64 = 16 bin 时，λ=64 的
    轴向峰恰好落在半径 16 上被一并挖掉，陷波后 SSIM 0.7486 反而低于不处理
    的 0.7583）。另做显著性校验兜住平坦谱上的伪峰。

    Args:
        spec_log: magnitude_spectrum 的输出。
        top_n: 最多返回几个峰。
        min_distance: 峰间最小间距（像素），默认为短边的 1/32。
        dc_radius: DC 屏蔽半径，显式给定则覆盖 max_period_px 的推算。
        threshold_rel: 相对阈值（相对屏蔽 DC 后的最大值）。
        min_prominence: 峰相对「背景中位数」的最小超出比例，0 表示不校验。
        max_period_px: 期望能检出的最大条纹周期（像素），默认 64。

    Returns:
        [(du, dv), ...]，按能量从高到低。

    Raises:
        ValueError: max_period_px 非正。
    """
    spec = np.asarray(spec_log, dtype=np.float64)
    h, w = spec.shape
    cy, cx = h // 2, w // 2
    if max_period_px <= 0:
        raise ValueError(f"max_period_px 必须为正，收到 {max_period_px}")
    min_distance = min_distance if min_distance is not None else max(8, min(h, w) // 32)
    if dc_radius is None:
        # 峰半径 = N/λ；取一半留出安全边界，保证 λ ≤ max_period_px 的峰不被吞
        dc_radius = max(6, int(0.5 * min(h, w) / float(max_period_px)))

    work = spec.copy()
    yy, xx = np.ogrid[:h, :w]
    dc_mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= dc_radius**2
    work[dc_mask] = 0.0

    peak_max = float(work.max())
    positive = work[work > 0]
    background = float(np.median(positive)) if positive.size else 0.0
    # 阈值同时受「相对最大值」与「背景中位数」约束，避免在平坦谱上凭空取峰
    thresh = min(
        peak_max,
        max(peak_max * threshold_rel, background * (1.0 + min_prominence)),
    )

    peaks: list[tuple[int, int]] = []
    for _ in range(top_n):
        idx = int(np.argmax(work))
        y, x = divmod(idx, w)
        if work[y, x] < thresh:
            break
        peaks.append((x - cx, y - cy))
        y0, y1 = max(0, y - min_distance), min(h, y + min_distance + 1)
        x0, x1 = max(0, x - min_distance), min(w, x + min_distance + 1)
        work[y0:y1, x0:x1] = 0.0
    return peaks


def _notch_centers(peaks: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """每个峰与它的 DC 对称点一起挖（实数信号频谱共轭对称）。"""
    centers: list[tuple[int, int]] = []
    for du, dv in peaks:
        centers.append((du, dv))
        if (du, dv) != (-du, -dv):
            centers.append((-du, -dv))
    return centers


def ideal_notch_mask(
    shape: tuple[int, int], peaks: list[tuple[int, int]], radius: int
) -> np.ndarray:
    """理想陷波掩膜：峰点邻域置 0，其余为 1。

    频域矩形截断 → 空域 sinc 波纹（振铃强），仅作对比基准。
    """
    h, w = shape
    cy, cx = h // 2, w // 2
    mask = np.ones((h, w), dtype=np.float64)
    yy, xx = np.ogrid[:h, :w]
    for du, dv in _notch_centers(peaks):
        mask[(yy - (cy + dv)) ** 2 + (xx - (cx + du)) ** 2 <= radius**2] = 0.0
    return mask


def butterworth_notch_mask(
    shape: tuple[int, int], peaks: list[tuple[int, int]], radius: int, order: int = 4
) -> np.ndarray:
    """Butterworth 陷波掩膜。

    公式::

        H(u,v) = Π_k 1 / (1 + [D0 / D_k(u,v)]^(2n))

    其中 D_k 是到第 k 个陷波点的距离，D0 为截止半径。
    阶数 n 越大响应越接近理想陷波（单调），n 小则过渡平缓、振铃弱。
    """
    if order < 1:
        raise ValueError(f"order 必须 ≥ 1，收到 {order}")
    h, w = shape
    cy, cx = h // 2, w // 2
    mask = np.ones((h, w), dtype=np.float64)
    yy, xx = np.ogrid[:h, :w]
    for du, dv in _notch_centers(peaks):
        d = np.sqrt((yy - (cy + dv)) ** 2 + (xx - (cx + du)) ** 2)
        mask *= 1.0 / (1.0 + (radius / np.maximum(d, 1e-9)) ** (2 * order))
    return np.clip(mask, 0.0, 1.0)


def _notch_plane(
    plane: np.ndarray, peaks: list[tuple[int, int]], kind: str, radius: int, order: int
) -> np.ndarray:
    """对单通道 float64 平面做陷波，返回同尺寸 float64（未裁剪）。"""
    h, w = plane.shape
    spec = np.fft.fftshift(np.fft.fft2(plane))
    if kind == "ideal":
        mask = ideal_notch_mask((h, w), peaks, radius)
    else:
        mask = butterworth_notch_mask((h, w), peaks, radius, order)
    return np.fft.ifft2(np.fft.ifftshift(spec * mask)).real


def notch_filter(
    gray: Image,
    peaks: list[tuple[int, int]],
    *,
    kind: str = "butterworth",
    radius: int | None = None,
    order: int = 4,
) -> Image:
    """对图像做陷波滤波，返回 float32[0,1]。

    彩色输入**只在亮度通道（YCbCr 的 Y）上陷波**，色度保持不变——
    网纹是全局周期性的，灰度处理足够；若整图转灰度再返回，后续的
    白平衡、彩色导出会直接退化成灰度件（彩色扫描件需求随之失效）。

    Args:
        gray: 灰度或彩色图（彩色图内部走 YCbCr 亮度通道）。
        peaks: find_periodic_peaks 的输出，或 UI 手动选点。
        kind: ``"butterworth"`` / ``"ideal"``。
        radius: 陷波半径（bin 数），默认为短边的 1/100。
        order: Butterworth 阶数。
    """
    if kind not in ("butterworth", "ideal"):
        raise ValueError(f"kind 只能是 'butterworth' 或 'ideal'，收到 {kind!r}")
    arr = to_float(gray)
    h, w = arr.shape[:2]
    radius = radius if radius is not None else max(2, min(h, w) // 100)

    if arr.ndim == 3:
        from docscan.core.colorspace import apply_on_luma

        def _on_luma(y: np.ndarray) -> np.ndarray:
            plane = np.clip(np.asarray(y, dtype=np.float64), 0.0, 255.0) / 255.0
            return _notch_plane(plane, peaks, kind, radius, order) * 255.0

        return apply_on_luma(arr, _on_luma)

    out = _notch_plane(arr.astype(np.float64), peaks, kind, radius, order)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def peak_energy_db(gray: Image, peak: tuple[int, int]) -> float:
    """谱上某峰点的能量（dB），用于量化「陷波挖掉了多少」。"""
    spec = fft2_shifted(gray)
    h, w = spec.shape
    cy, cx = h // 2, w // 2
    du, dv = peak
    return float(20.0 * np.log10(abs(spec[cy + dv, cx + du]) + 1e-12))


def energy_drop_db(before: Image, after: Image, peak: tuple[int, int]) -> float:
    """陷波前后峰点能量差（dB，负值 = 能量下降）。"""
    return peak_energy_db(after, peak) - peak_energy_db(before, peak)
