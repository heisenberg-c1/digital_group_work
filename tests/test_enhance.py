"""空间域增强测试：伽马、拉伸、均衡、CLAHE。"""

from __future__ import annotations

import numpy as np
import pytest

from docscan.core import enhance


class TestGamma:
    def test_gamma_one_strict_identity(self):
        rng = np.random.default_rng(1)
        img = rng.integers(0, 256, size=(60, 80, 3), dtype=np.uint8)
        out = enhance.apply_gamma(img, gamma=1.0)
        np.testing.assert_array_equal(out, img)

    def test_lut_gamma_one_identity(self):
        lut = enhance.gamma_lut(1.0)
        np.testing.assert_array_equal(lut, np.arange(256, dtype=np.uint8))

    def test_gamma_brightens_dark(self):
        img = np.full((10, 10), 50, dtype=np.uint8)
        out = enhance.apply_gamma(img, gamma=2.0)
        assert out.mean() > img.mean()

    def test_invalid_gamma_raises(self):
        with pytest.raises(ValueError):
            enhance.gamma_lut(0.0)
        with pytest.raises(ValueError):
            enhance.gamma_lut(-1.0)

    def test_float_input(self):
        img = np.full((20, 20), 0.25, np.float32)
        out = enhance.apply_gamma(img, gamma=2.0)
        assert out.dtype == np.float32
        assert out.mean() > 0.25


class TestContrastStretch:
    def test_stretches_range(self):
        rng = np.random.default_rng(2)
        img = (rng.uniform(0.3, 0.6, size=(50, 50))).astype(np.float32)
        out = enhance.contrast_stretch(img, low_pct=1, high_pct=99)
        assert out.min() < img.min() + 1e-4
        assert out.max() > img.max() - 1e-4

    def test_constant_image_no_div_zero(self):
        img = np.full((30, 30), 0.5, np.float32)
        out = enhance.contrast_stretch(img)
        np.testing.assert_allclose(out, img, atol=1e-6)


class TestEqualize:
    def test_full_range_after_equalize(self):
        rng = np.random.default_rng(3)
        img = (rng.uniform(0.3, 0.6, size=(80, 80)) * 255).astype(np.uint8)
        out = enhance.hist_equalize(img)
        assert out.dtype == np.uint8
        assert out.max() > 200 and out.min() < 60

    def test_color_no_color_shift_direction(self):
        """彩色均衡只在 Y 上做，BGR 三通道同值像素输出仍近似同值。"""
        img = np.full((60, 60, 3), 100, dtype=np.uint8)
        img[..., 0] = 100
        out = enhance.hist_equalize(img)
        # 常数图：Y 恒定 → 输出仍为灰色（三通道近似相等）
        per_pixel_spread = out.max(axis=2).astype(int) - out.min(axis=2).astype(int)
        assert int(per_pixel_spread.max()) <= 2


class TestClahe:
    def test_enhances_local_contrast(self):
        rng = np.random.default_rng(5)
        base = rng.uniform(0.45, 0.55, size=(160, 160))
        img = (base * 255).astype(np.uint8)
        out = enhance.clahe(img, clip_limit=3.0, grid_size=8)
        assert float(out.std()) > float(img.std())

    def test_clip_limit_bounds_behavior(self):
        """clipLimit 越小削波越强：极小值下输出被大幅重映射（非恒等）。

        OpenCV 的 CLAHE 无论 clip 大小都做 tile 内直方图均衡，
        clip 只限制放大倍率——这与 skimage 的语义一致，故不设恒等断言。
        """
        rng = np.random.default_rng(4)
        img = (rng.uniform(0.35, 0.65, size=(120, 120)) * 255).astype(np.uint8)
        strong = enhance.clahe(img, clip_limit=0.01, grid_size=8)
        assert strong.shape == img.shape
        assert float(strong.std()) > 20.0  # 低对比输入被拉开

    def test_color_output_in_range(self):
        img = np.full((50, 50, 3), 120, dtype=np.uint8)
        img[10:20, 10:20] = 90
        out = enhance.clahe(img, clip_limit=2.0)
        assert out.shape == img.shape
        assert out.min() >= 0 and out.max() <= 255
