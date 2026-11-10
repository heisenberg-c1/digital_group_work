"""噪声抑制测试：中值 / 双边，以及「保 dtype」这条管线约定。"""

from __future__ import annotations

import cv2
import numpy as np

from conftest import make_page
from docscan.core import denoise


def _salt_pepper(gray: np.ndarray, ratio: float = 0.05, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = gray.copy()
    mask = rng.random(gray.shape) < ratio
    out[mask] = rng.choice([0, 255], size=int(mask.sum())).astype(np.uint8)
    return out


class TestMedian:
    def test_removes_salt_pepper(self):
        clean = cv2.cvtColor(make_page(200, 260), cv2.COLOR_BGR2GRAY)
        noisy = _salt_pepper(clean, 0.05)
        out = denoise.median(noisy, ksize=3)
        # 与干净图的差应显著小于噪声图与干净图的差
        assert float(np.abs(out.astype(np.int16) - clean).mean()) < float(
            np.abs(noisy.astype(np.int16) - clean).mean()
        ) * 0.4

    def test_even_ksize_is_normalized_to_odd(self):
        img = make_page(64, 64)
        out = denoise.median(img, ksize=4)  # cv2 要求奇数，内部应规整为 5
        assert out.shape == img.shape

    def test_float_input_returns_float(self):
        img = make_page(64, 64).astype(np.float32) / 255.0
        out = denoise.median(img, ksize=3)
        assert out.dtype == np.float32
        assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0

    def test_uint8_input_returns_uint8(self):
        img = make_page(64, 64)
        assert denoise.median(img).dtype == np.uint8


class TestBilateral:
    def test_preserves_edges_better_than_median(self):
        """双边按灰度差加权，同一平滑强度下边缘保持更好（梯度更大）。"""
        img = cv2.cvtColor(make_page(160, 200), cv2.COLOR_BGR2GRAY)
        noisy = _salt_pepper(img, 0.03, seed=1)
        bil = denoise.bilateral(noisy, sigma_color=0.08)
        med = denoise.median(noisy, ksize=5)
        grad = lambda a: float(cv2.Laplacian(a.astype(np.float64), cv2.CV_64F).var())  # noqa: E731
        assert grad(bil) > grad(med)

    def test_keeps_shape_and_dtype(self):
        img = make_page(80, 120)
        out = denoise.bilateral(img)
        assert out.shape == img.shape and out.dtype == np.uint8

    def test_float_range_clipped(self):
        img = make_page(64, 64).astype(np.float32) / 255.0
        out = denoise.bilateral(img, sigma_color=0.2)
        assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
