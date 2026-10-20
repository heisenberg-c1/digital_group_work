"""指标测试：CER 边界与行为。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from docscan.core import metrics


class TestCER:
    def test_empty_reference(self):
        assert metrics.cer("", "") == 0.0
        assert metrics.cer("", "abc") == 1.0  # 参考空、识别非空 → 全错

    def test_identical_is_zero(self):
        assert metrics.cer("数字图像处理", "数字图像处理") == 0.0

    def test_all_substitution(self):
        """等长全替换：CER = 1。"""
        assert metrics.cer("abcdef", "uvwxyz") == pytest.approx(1.0)

    def test_all_insertion(self):
        """纯插入：编辑距离 = 插入数。"""
        assert metrics.cer("abc", "xxabc") == pytest.approx(2 / 3)
        assert metrics.cer("abc", "aabbcc") == pytest.approx(1.0)

    def test_whitespace_ignored(self):
        """OCR 的换行与空格不算错误。"""
        assert metrics.cer("数字 图像", "数字\n图像") == 0.0
        assert metrics.cer("hello world", "helloworld") == 0.0

    def test_partial_deletion(self):
        assert metrics.cer("abcdef", "abef") == pytest.approx(2 / 6)

    def test_substitution_cheaper_than_delete_insert(self):
        assert metrics.cer("abcd", "abxd") == pytest.approx(1 / 4)


class TestPSNRSSIMExtras:
    def test_psnr_known_noise(self):
        """高斯噪声解析值验证（σ=0.02 → 34 dB）。"""
        rng = np.random.default_rng(8)
        img = np.full((300, 300), 0.5, np.float32)
        noisy = np.clip(img + rng.normal(0, 0.02, img.shape), 0, 1).astype(np.float32)
        assert abs(metrics.psnr(img, noisy) - (-20 * math.log10(0.02))) < 0.05

    def test_ssim_constant_pair_finite(self):
        a = np.full((40, 40), 0.3, np.float32)
        b = np.full((40, 40), 0.3, np.float32)
        v = metrics.ssim(a, b)
        assert np.isfinite(v)
        assert v == pytest.approx(1.0, abs=1e-6)

    def test_size_mismatch_raises(self):
        import numpy as np

        with pytest.raises(ValueError):
            metrics.psnr(np.zeros((10, 10)), np.zeros((20, 20)))
