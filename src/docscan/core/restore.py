"""退化建模与复原：PSF 合成、逆滤波、维纳滤波、倒谱估计 PSF。

退化模型（线性空间不变）::

    g(x,y) = f(x,y) * h(x,y) + n(x,y)
    G(u,v) = F(u,v) · H(u,v) + N(u,v)

频域实现要点：小尺寸 PSF 必须先「中心化嵌入」大画布（把 PSF 中心移到 (0,0)
再做 FFT），否则卷积相位是错的——这是复现不上「教科书公式」的最常见原因。
"""

from __future__ import annotations

import cv2
import numpy as np

from docscan.core.types import Image, to_float


def motion_psf(size: int, length: int, angle_deg: float = 0.0) -> np.ndarray:
    """运动模糊 PSF：长度 ``length``、方向 ``angle_deg`` 的线段核。

    Args:
        size: 核边长（奇数）。
        length: 运动模糊长度（像素）。
        angle_deg: 运动方向，0 = 水平，顺时针为正（图像坐标系 y 向下）。

    Returns:
        (size, size) float64，和为 1。
    """
    if size % 2 == 0 or size < 3:
        raise ValueError(f"size 必须是奇数且 ≥ 3，收到 {size}")
    if length < 1:
        raise ValueError(f"length 必须 ≥ 1，收到 {length}")
    if length > size:
        raise ValueError(f"length={length} 超出核尺寸 {size}")

    psf = np.zeros((size, size), dtype=np.float64)
    c = size // 2
    half = (length - 1) / 2.0
    theta = np.deg2rad(angle_deg)
    dx, dy = np.cos(theta), np.sin(theta)
    # 以亚像素步进画线，长度的误差 ≤ 0.5px
    steps = max(int(length * 8), 2)
    for t in np.linspace(-half, half, steps):
        x = c + t * dx
        y = c + t * dy
        psf[round(y), round(x)] += 1.0
    return psf / psf.sum()


def disk_psf(size: int, radius: int) -> np.ndarray:
    """散焦（out-of-focus）PSF：半径 ``radius`` 的圆盘核，和为 1。"""
    if radius < 1 or radius > size // 2:
        raise ValueError(f"radius 必须在 1..{size // 2}，收到 {radius}")
    psf = np.zeros((size, size), dtype=np.float64)
    cv2.circle(psf, (size // 2, size // 2), radius, 1.0, thickness=-1, lineType=cv2.LINE_8)
    return psf / psf.sum()


def freq_kernel(psf: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """把 PSF 嵌入大画布并中心化到 (0,0)，返回其 FFT（ optical transfer function）。"""
    kh, kw = psf.shape
    h, w = shape
    if kh > h or kw > w:
        raise ValueError(f"PSF 尺寸 {psf.shape} 大于画布 {shape}")
    canvas = np.zeros((h, w), dtype=np.float64)
    canvas[:kh, :kw] = psf
    canvas = np.roll(canvas, (-(kh // 2), -(kw // 2)), axis=(0, 1))
    return np.fft.fft2(canvas)


def convolve_blur(img: Image, psf: np.ndarray, *, noise_sigma: float = 0.0) -> Image:
    """用给定 PSF 退化图像（模拟 g = f * h + n），输出 float32[0,1]。

    噪声用固定高斯模型（np.random 默认源），便于实验复现时外部传 seed。
    """
    arr = to_float(img)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    arr = arr.astype(np.float64)
    h = freq_kernel(psf, arr.shape)
    out = np.fft.ifft2(np.fft.fft2(arr) * h).real
    if noise_sigma > 0:
        rng = np.random.default_rng()
        out = out + rng.normal(0.0, noise_sigma, out.shape)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def inverse_filter(
    img: Image, psf: np.ndarray, *, eps: float = 1e-3
) -> Image:
    """逆滤波（带 ε 正则）::

        F = G / H,  其中 |H| ≥ eps；否则 F = 0（截断噪声放大区）

    eps 越小复原越「激进」（噪声放大越重），越大越保守。

    Raises:
        ValueError: eps 非正。
    """
    if eps <= 0:
        raise ValueError(f"eps 必须为正，收到 {eps}")
    arr = _to_gray64(img)
    G = np.fft.fft2(arr)
    H = freq_kernel(psf, arr.shape)
    F = np.zeros_like(G)
    strong = np.abs(H) >= eps
    F[strong] = G[strong] / H[strong]
    out = np.fft.ifft2(F).real
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def wiener(img: Image, psf: np.ndarray, *, K: float = 0.01) -> Image:
    """维纳滤波（最小均方误差意义下的最优线性复原）::

        F = G · H* / (|H|² + K)

    K = 0 退化为纯逆滤波；K 越大越平滑（越接近「不处理」）。
    K 应近似等于噪信比 N/S。

    Raises:
        ValueError: K 为负。
    """
    if K < 0:
        raise ValueError(f"K 不能为负，收到 {K}")
    arr = _to_gray64(img)
    G = np.fft.fft2(arr)
    H = freq_kernel(psf, arr.shape)
    H2 = np.abs(H) ** 2
    # K=0 且 |H|→0 时分母为 0，加机器级小量防炸（恒等 PSF 情形 H=1 不受影响）
    denom = H2 + K
    denom = np.where(denom == 0, np.finfo(np.float64).tiny, denom)
    F = np.conj(H) * G / denom
    out = np.fft.ifft2(F).real
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def estimate_psf_cepstrum(
    img: Image, *, min_length: int = 3, search_radius: int | None = None
) -> tuple[float, int]:
    """倒谱法估计运动模糊的方向与长度。

    原理：运动模糊的频谱沿运动方向有规律的暗纹（H 的零点带），其倒谱在
    运动方向上距离 ``L`` 处呈现一对负峰。找「最深的负峰」即得模糊向量。

    Returns:
        (angle_deg, length)。angle 为运动方向，已折叠到 [-90°, 90°)
        （运动模糊对 180° 方向不敏感：0° 与 180° 的 PSF 相同）；
        length 为模糊长度（像素）。

    Raises:
        RuntimeError: 未找到显著负峰（如图像本身没有运动模糊）。
    """
    arr = _to_gray64(img)
    h, w = arr.shape
    cy, cx = h // 2, w // 2
    log_spec = np.log(np.abs(np.fft.fft2(arr)) + 1e-12)
    cep = np.fft.fftshift(np.fft.ifft2(log_spec).real)

    if search_radius is None:
        search_radius = min(h, w) // 4
    work = cep.copy()
    # 排除中心区（倒谱的直流与图像自身结构）
    r0 = max(3, min_length // 2)
    work[cy - r0: cy + r0 + 1, cx - r0: cx + r0 + 1] = 0.0
    # 只在圆环内搜索
    yy, xx = np.ogrid[:h, :w]
    far = (yy - cy) ** 2 + (xx - cx) ** 2 > search_radius**2
    work[far] = 0.0

    idx = int(np.argmin(work))
    vy, vx = np.unravel_index(idx, work.shape)
    if work[vy, vx] >= 0:
        raise RuntimeError("倒谱中未找到显著负峰，图像可能没有运动模糊")

    dy, dx = float(vy - cy), float(vx - cx)
    # 0° 与 180° 的线段 PSF 相同 → 折叠方向歧义
    angle = float(np.degrees(np.arctan2(dy, dx)))
    if angle >= 90.0:
        angle -= 180.0
    elif angle < -90.0:
        angle += 180.0
    length = max(round(float(np.hypot(dy, dx))), min_length)
    return angle, length


def _to_gray64(img: Image) -> np.ndarray:
    arr = to_float(img)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    return arr.astype(np.float64)
