"""光照归一测试：去阴影与 Retinex。"""

from __future__ import annotations

import numpy as np
import pytest

from docscan.core import illumination


def _gradient_light(shape, lo=0.45, hi=1.0):
    """水平方向线性光照场。"""
    h, w = shape
    ramp = np.linspace(lo, hi, w, dtype=np.float32)
    return np.tile(ramp, (h, 1))


class TestRemoveShading:
    def test_block_mean_std_drops_to_fifth(self):
        """已知梯度光照下的均匀页面：校正后块均值 std 降到原来的 1/5 以下。"""
        page = np.full((600, 800), 0.88, dtype=np.float32)
        light = _gradient_light(page.shape)
        shaded = np.clip(page * light, 0, 1)
        out = illumination.remove_shading(shaded, sigma=40.0)

        def block_std(img, n=8):
            means = []
            hs, ws = img.shape[0] // n, img.shape[1] // n
            for i in range(n):
                for j in range(n):
                    block = img[i * hs:(i + 1) * hs, j * ws:(j + 1) * ws]
                    means.append(float(block.mean()))
            return float(np.std(means))

        before, after = block_std(shaded), block_std(out)
        assert after < before / 5.0, f"before={before:.4f} after={after:.4f}"

    def test_text_survives_correction(self):
        """文字（低值区）在除法后仍然比纸面暗，没有被抹掉。"""
        page = np.full((400, 600), 0.9, dtype=np.float32)
        page[180:220, 100:500] = 0.15  # 一条黑字
        light = _gradient_light(page.shape)
        out = illumination.remove_shading(np.clip(page * light, 0, 1), sigma=40.0)
        assert out[200, 300] < out[100, 300] - 0.3  # 字内 vs 纸面

    def test_strength_zero_is_identity(self):
        rng = np.random.default_rng(0)
        img = rng.uniform(0.2, 0.9, size=(100, 120)).astype(np.float32)
        out = illumination.remove_shading(img, strength=0.0)
        np.testing.assert_allclose(out, img, atol=1e-6)

    def test_output_range(self):
        img = np.clip(np.full((80, 80), 0.7) * _gradient_light((80, 80)), 0, 1)
        out = illumination.remove_shading(img)
        assert out.min() >= 0.0 and out.max() <= 1.0
        assert not np.isnan(out).any()

    def test_color_image(self):
        """彩色输入逐通道处理，输出仍是三通道。"""
        img = np.stack(
            [
                np.full((100, 100), 0.8, np.float32),
                np.full((100, 100), 0.7, np.float32),
                np.full((100, 100), 0.6, np.float32),
            ],
            axis=-1,
        ) * _gradient_light((100, 100))[..., None]
        out = illumination.remove_shading(np.clip(img, 0, 1))
        assert out.ndim == 3 and out.shape == img.shape


class TestRetinex:
    def test_no_nan_no_negative_in_range(self):
        rng = np.random.default_rng(1)
        img = rng.uniform(0.0, 1.0, size=(120, 140, 3)).astype(np.float32)
        out = illumination.retinex(img)
        assert not np.isnan(out).any()
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_constant_image_is_stable(self):
        """常数图：Retinex 的高频项为零，min-max 归一不除零、输出全 0。"""
        img = np.full((64, 64), 0.5, dtype=np.float32)
        out = illumination.retinex(img)
        assert out.shape == img.shape
        assert float(out.max()) <= 1.0

    def test_recovers_text_from_shadow(self):
        """阴影里的文字，Retinex 后纸面-文字对比度恢复。"""
        page = np.full((200, 300), 0.85, dtype=np.float32)
        page[80:120, 60:240] = 0.2
        light = _gradient_light(page.shape, lo=0.35)
        out = illumination.retinex(np.clip(page * light, 0, 1))
        paper, text = float(out[40, 150]), float(out[100, 150])
        assert paper - text > 0.3, f"paper={paper:.3f} text={text:.3f}"

    def test_mismatched_weights_raise(self):
        img = np.full((32, 32), 0.5, np.float32)
        with pytest.raises(ValueError):
            illumination.retinex(img, weights=(1.0, 1.0))
