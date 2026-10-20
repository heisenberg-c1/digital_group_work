"""色彩空间与白平衡测试。"""

from __future__ import annotations

import numpy as np

from docscan.core import color, colorspace


class TestYCbCr:
    def test_roundtrip_error_leq_one_gray_level(self):
        rng = np.random.default_rng(7)
        bgr = rng.uniform(0, 1, size=(60, 80, 3)).astype(np.float32)
        y, cb, cr = colorspace.bgr_to_ycbcr(bgr)
        back = colorspace.ycbcr_to_bgr(y, cb, cr)
        diff = np.abs(back - bgr) * 255.0
        assert float(diff.max()) <= 1.0, f"max roundtrip err={diff.max():.3f}"

    def test_gray_image_chroma_centered(self):
        """灰色图 Cb/Cr 应≈128。"""
        img = np.full((40, 40, 3), 0.5, np.float32)
        y, cb, cr = colorspace.bgr_to_ycbcr(img)
        assert abs(float(cb.mean()) - 128.0) < 0.5
        assert abs(float(cr.mean()) - 128.0) < 0.5
        np.testing.assert_allclose(y, 127.5, atol=0.5)

    def test_apply_on_luma_keeps_chroma(self):
        """只改 Y 且不越 gamut 时，色度与输入逐元素一致（±1 灰度级）。

        注意：若映射把像素推出 gamut（裁剪发生），色度必然改变——
        这是色彩空间的数学性质而非实现缺陷，故用温和的 gamut 内变换验证。
        """
        rng = np.random.default_rng(11)
        bgr = rng.uniform(0.2, 0.8, size=(50, 70, 3)).astype(np.float32)
        out = colorspace.apply_on_luma(bgr, lambda y: 0.5 * y + 64.0)
        y0, cb0, cr0 = colorspace.bgr_to_ycbcr(bgr)
        y1, cb1, cr1 = colorspace.bgr_to_ycbcr(out)
        assert np.abs(cb1 - cb0).max() <= 1.0
        assert np.abs(cr1 - cr0).max() <= 1.0
        # 亮度确实被映射：y1 ≈ 0.5·y0 + 64（往返误差 ≤ 1 灰度级）
        np.testing.assert_allclose(y1, 0.5 * y0 + 64.0, atol=1.0)

    def test_apply_identity_is_identity(self):
        rng = np.random.default_rng(3)
        bgr = rng.uniform(0, 1, size=(40, 40, 3)).astype(np.float32)
        out = colorspace.apply_on_luma(bgr, lambda y: y)
        assert float(np.abs(out - bgr).max()) * 255.0 <= 1.0


class TestGrayWorld:
    def test_balanced_image_is_identity(self):
        """已白平衡（三通道均值相等）的图 → 恒等（±1 灰度级）。"""
        rng = np.random.default_rng(5)
        base = rng.uniform(0.2, 0.8, size=(80, 80)).astype(np.float32)
        img = np.stack([base, base, base], axis=-1)
        out = color.gray_world(img)
        assert float(np.abs(out - img).max()) * 255.0 <= 1.0

    def test_color_cast_corrected(self):
        """偏蓝图（B 通道强）校正后三通道均值接近。"""
        rng = np.random.default_rng(9)
        base = rng.uniform(0.3, 0.7, size=(100, 100)).astype(np.float32)
        img = np.stack([base * 1.3, base, base * 0.7], axis=-1)  # BGR：蓝强红弱
        img = np.clip(img, 0, 1)
        out = color.gray_world(img)
        means = out.reshape(-1, 3).mean(axis=0)
        assert float(means.max() - means.min()) < 0.05

    def test_strength_zero_identity(self):
        rng = np.random.default_rng(2)
        img = rng.uniform(0, 1, size=(40, 40, 3)).astype(np.float32)
        out = color.gray_world(img, strength=0.0)
        np.testing.assert_allclose(out, img, atol=1e-6)


class TestWhitePatch:
    def test_balanced_is_identity(self):
        rng = np.random.default_rng(4)
        base = rng.uniform(0.1, 0.9, size=(80, 80)).astype(np.float32)
        img = np.stack([base, base, base], axis=-1)
        out = color.white_patch(img)
        assert float(np.abs(out - img).max()) * 255.0 <= 1.0

    def test_cast_corrected(self):
        rng = np.random.default_rng(6)
        base = rng.uniform(0.3, 0.9, size=(100, 100)).astype(np.float32)
        img = np.clip(np.stack([base * 1.2, base, base * 0.8], axis=-1), 0, 1)
        out = color.white_patch(img)
        highs = [float(out[..., c].max()) for c in range(3)]
        assert max(highs) - min(highs) < 0.15
