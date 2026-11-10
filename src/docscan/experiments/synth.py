"""退化合成器：为报告实验生成「干净图 + 退化图」严格配对的数据。

五种典型场景与课程要求的对应关系::

    motion    运动模糊   → 复原实验（维纳 K 扫描）
    defocus   散焦模糊   → 复原实验
    moire     摩尔纹     → 去网纹实验（理想 vs Butterworth）
    shadow    光照不均   → 光照归一实验
    cast      偏色       → 白平衡实验

GT（ground truth）就是传入的原图本身，保证逐像素配对。
"""

from __future__ import annotations

import cv2
import numpy as np

from docscan.core import restore
from docscan.core.types import Image, to_float

SCENARIOS = ("perspective", "shadow", "moire", "motion", "defocus", "cast")


def _as_gray(img: Image) -> np.ndarray:
    arr = to_float(img)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    return arr.astype(np.float32)


def add_perspective(
    img: Image, *, jitter: float = 0.10, desk: float = 0.35, seed: int = 0
) -> Image:
    """透视畸变：把页面四角向内收，四周用深色桌面填充，模拟手持斜拍。

    没有这一场景，几何校正就只能靠单元测试里的合成四边形自证，
    端到端表格里根本看不出「透视校正是否有效」。
    """
    arr = _as_gray(img)
    h, w = arr.shape[:2]
    rng = np.random.default_rng(seed)
    dx, dy = jitter * w, jitter * h
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = np.float32(
        [
            [rng.uniform(0.5, 1.0) * dx, rng.uniform(0.5, 1.0) * dy],
            [w - 1 - rng.uniform(0.3, 0.9) * dx, rng.uniform(0.2, 0.7) * dy],
            [w - 1 - rng.uniform(0.2, 0.6) * dx, h - 1 - rng.uniform(0.5, 1.0) * dy],
            [rng.uniform(0.5, 1.0) * dx, h - 1 - rng.uniform(0.3, 0.8) * dy],
        ]
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(
        arr, matrix, (w, h), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=desk,
    )
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def add_motion_blur(img: Image, *, length: int = 11, angle_deg: float = 0.0,
                    noise_sigma: float = 0.005, seed: int = 0) -> Image:
    """运动模糊 + 轻噪声。"""
    psf = restore.motion_psf(max(31, length * 2 | 1), length, angle_deg)
    arr = _as_gray(img).astype(np.float64)
    h = restore.freq_kernel(psf, arr.shape)
    out = np.fft.ifft2(np.fft.fft2(arr) * h).real
    rng = np.random.default_rng(seed)
    out = out + rng.normal(0.0, noise_sigma, out.shape)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def add_defocus(img: Image, *, radius: int = 5, noise_sigma: float = 0.005,
                seed: int = 1) -> Image:
    """散焦模糊 + 轻噪声。"""
    psf = restore.disk_psf(max(31, radius * 4 | 1), radius)
    arr = _as_gray(img).astype(np.float64)
    h = restore.freq_kernel(psf, arr.shape)
    out = np.fft.ifft2(np.fft.fft2(arr) * h).real
    rng = np.random.default_rng(seed)
    out = out + rng.normal(0.0, noise_sigma, out.shape)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def add_moire(
    img: Image,
    *,
    wavelength: int = 24,
    angle_deg: float = 30.0,
    amplitude: float = 0.18,
    return_base: bool = False,
) -> Image | tuple[Image, Image]:
    """叠加正弦摩尔纹（不越界合成，避免谐波失真）。

    为了让 ``content + stripe`` 不越过 [0,1]，先把内容整体压到 ``1-amplitude``。
    这带来一个必须显式处理的后果：**退化图整体比原图暗**，若仍拿原图当 GT，
    SSIM/振铃指标会带一层系统性的亮度偏差（实测 SSIM 被压到 0.41、振铃被
    残差淹没到无法区分两种陷波器）。因此 ``return_base=True`` 时额外返回
    「压暗后、无条纹」的同一内容作为 GT —— 这正是理想去网纹器该恢复的东西。
    """
    arr = _as_gray(img)
    h, w = arr.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    theta = np.deg2rad(angle_deg)
    phase = 2.0 * np.pi * (xx * np.cos(theta) + yy * np.sin(theta)) / wavelength
    # 先把内容压进安全区间再叠加，保证 max(内容)+幅度 ≤ 1
    scaled = arr * (1.0 - amplitude)
    out = np.clip(scaled + amplitude * np.sin(phase), 0.0, 1.0).astype(np.float32)
    if return_base:
        return out, scaled.astype(np.float32)
    return out


def add_shadow(img: Image, *, lo: float = 0.4, angle_deg: float = 0.0) -> Image:
    """乘性线性光照不均（模拟侧光拍摄）。"""
    arr = _as_gray(img)
    h, w = arr.shape
    theta = np.deg2rad(angle_deg)
    ramp = (np.cos(theta) * np.linspace(-0.5, 0.5, w)[None, :]
            + np.sin(theta) * np.linspace(-0.5, 0.5, h)[:, None])
    field = lo + (1.0 - lo) * (ramp - ramp.min()) / max(float(np.ptp(ramp)), 1e-9)
    return np.clip(arr * field, 0.0, 1.0).astype(np.float32)


def add_color_cast(
    img: Image, *, gains: tuple[float, float, float] = (1.0, 0.78, 0.62)
) -> Image:
    """BGR 偏色（模拟暖光/冷光下的白平衡失误）。仅对彩色图有意义。

    增益刻意全部 ≤ 1（B 保持、G/R 被吸收）：若用 >1 的增益，亮部会直接
    削顶到 1.0，信息被不可逆地破坏，之后任何白平衡都救不回来——实验会
    显示「校正后反而更差」，那是合成器的问题，不是算法的问题。
    """
    arr = to_float(img)
    if arr.ndim != 3:
        return arr
    return np.clip(arr * np.array(gains, dtype=np.float32), 0.0, 1.0).astype(np.float32)


def synthesize(img: Image, kind: str, **kwargs) -> tuple[Image, Image]:
    """按场景名生成 (退化图, GT) 配对。

    GT 与退化图严格同域、同亮度：色彩域场景（cast）返回彩色 GT，
    moire 返回「压暗后无条纹」的内容（见 ``add_moire``），其余返回灰度 GT。
    """
    gt = to_float(img).copy() if kind == "cast" else _as_gray(img)
    table = {
        "perspective": add_perspective,
        "motion": add_motion_blur,
        "defocus": add_defocus,
        "moire": add_moire,
        "shadow": add_shadow,
        "cast": add_color_cast,
    }
    if kind not in table:
        raise ValueError(f"未知场景 {kind!r}，可选：{SCENARIOS}")
    if kind == "moire":
        degraded, base = add_moire(img, return_base=True, **kwargs)
        return degraded, base
    return table[kind](img, **kwargs), gt
