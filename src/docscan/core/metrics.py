"""评测指标：PSNR、SSIM、清晰度、振铃。

约定：输入 uint8 或 float32[0,1]，内部统一到 float [0,1] 域计算。
测试策略：SSIM/PSNR 与 scikit-image 对拍（oracle），core/ 本身不依赖 skimage。
"""

from __future__ import annotations

import cv2
import numpy as np

from docscan.core.types import Image, to_float


def _as_gray_pair(ref: Image, test: Image) -> tuple[np.ndarray, np.ndarray]:
    a = to_float(np.asarray(ref))
    b = to_float(np.asarray(test))
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    if b.ndim == 3:
        b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    if a.shape != b.shape:
        raise ValueError(f"两幅图尺寸不一致：{a.shape} vs {b.shape}")
    return a.astype(np.float64), b.astype(np.float64)


def psnr(ref: Image, test: Image) -> float:
    """峰值信噪比（dB），完全相同返回 inf。

    公式::

        PSNR = 10·log10(1 / MSE)    （值域 [0,1]，MAX=1）
    """
    a, b = _as_gray_pair(ref, test)
    mse = float(((a - b) ** 2).mean())
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def _gaussian_kernel(ksize: int = 11, sigma: float = 1.5) -> np.ndarray:
    kx = cv2.getGaussianKernel(ksize, sigma)
    return (kx @ kx.T).astype(np.float64)


_SSIM_KERNEL = _gaussian_kernel()


def ssim(ref: Image, test: Image) -> float:
    """结构相似度（高斯窗 11×11，σ=1.5，与 skimage 默认参数一致）。

    公式（局部统计量经高斯加权）::

        SSIM = [(2μxμy + C1)(2σxy + C2)] / [(μx²+μy²+C1)(σx²+σy²+C2)]
        C1 = 0.01², C2 = 0.03²（值域 [0,1]）

    Returns:
        全图均值，范围 [-1, 1]，相同图恒为 1.0。
    """
    x, y = _as_gray_pair(ref, test)
    c1, c2 = (0.01) ** 2, (0.03) ** 2

    def filt(img: np.ndarray) -> np.ndarray:
        return cv2.filter2D(img, -1, _SSIM_KERNEL, borderType=cv2.BORDER_REFLECT)

    mu_x, mu_y = filt(x), filt(y)
    mu_xx, mu_yy, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_x = filt(x * x) - mu_xx
    sigma_y = filt(y * y) - mu_yy
    sigma_xy = filt(x * y) - mu_xy

    num = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    den = (mu_xx + mu_yy + c1) * (sigma_x + sigma_y + c2)
    local = num / den
    return float(local.mean())


def cer(reference: str, hypothesis: str) -> float:
    """字符错误率（Character Error Rate）::

        CER = 编辑距离（Levenshtein） / 参考文本长度

    比较前去除所有空白字符（OCR 输出的换行/空格不属于识别错误）。

    边界约定：参考为空时，识别也为空 → 0.0；识别非空 → 1.0（全错）。
    """
    ref = "".join(reference.split())
    hyp = "".join(hypothesis.split())
    if not ref:
        return 0.0 if not hyp else 1.0

    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i]
        for j, hc in enumerate(hyp, 1):
            cost = 0 if rc == hc else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1] / len(ref)


def sharpness(gray: Image) -> float:
    """清晰度：拉普拉斯响应方差（Tenengrad 变体）。

    模糊图的边缘响应弱 → 方差小；清晰图方差大。
    """
    arr = to_float(np.asarray(gray))
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    arr64 = arr.astype(np.float64)
    lap = cv2.Laplacian(arr64, cv2.CV_64F, ksize=3)
    return float(lap.var())


def ringing_index(
    gray: Image, reference: Image | None = None, *, band_width: int = 3
) -> float:
    """振铃指标：边缘环带内被滤波引入的伪影 / 波纹能量。

    两种用法：

    1. ``ringing_index(filtered, reference=clean)`` —— 环带内
       ``|filtered − clean|`` 的均值，直接度量「滤波在边缘附近引入了什么」。
       理想陷波频域矩形截断 → 空域 sinc 波纹强 → 指标高；
       Butterworth 过渡带平缓 → 指标低。这是对比两种陷波器的核心论据。
    2. ``ringing_index(img)``（无参考）—— 环带内 Sobel 梯度能量，
       作为图像自身「边缘处波纹丰富度」的粗略描述。

    环带定义：Canny 强边缘膨胀 ``band_width`` 像素后去掉边缘本身。
    """
    arr = to_float(np.asarray(gray))
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    u8 = np.rint(np.clip(arr, 0, 1) * 255).astype(np.uint8)
    edges = cv2.Canny(u8, 80, 160)
    if edges.sum() == 0:
        return 0.0
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * band_width + 1, 2 * band_width + 1)
    )
    ring = cv2.dilate(edges, kernel) - edges

    if reference is not None:
        ref = to_float(np.asarray(reference))
        if ref.ndim == 3:
            ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        return float(np.abs(arr - ref)[ring > 0].mean())

    grad = cv2.Sobel(arr.astype(np.float64), cv2.CV_64F, 1, 0, ksize=3) ** 2
    grad += cv2.Sobel(arr.astype(np.float64), cv2.CV_64F, 0, 1, ksize=3) ** 2
    grad = np.sqrt(grad)
    n = max(int((edges > 0).sum()), 1)
    return float(grad[ring > 0].sum() / n)
