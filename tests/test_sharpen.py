"""USM 锐化与 OCR 降级测试。"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from conftest import make_page
from docscan.core import metrics, sharpen


class TestUnsharpMask:
    def test_amount_zero_strict_identity(self):
        img = make_page(120, 160)
        out = sharpen.unsharp_mask(img, amount=0.0)
        np.testing.assert_array_equal(out, img)

    def test_sharpens_blurred(self):
        """模糊图锐化后清晰度指标上升。"""
        img = make_page(200, 280)
        blurred = cv2.GaussianBlur(img, (7, 7), 2.0)
        out = sharpen.unsharp_mask(blurred, radius=2.0, amount=1.5)
        assert metrics.sharpness(out) > metrics.sharpness(blurred)

    def test_output_range_and_dtype(self):
        img = make_page(80, 80)
        out = sharpen.unsharp_mask(img, amount=2.0)
        assert out.dtype == np.uint8
        assert out.min() >= 0 and out.max() <= 255

    def test_invalid_params_raise(self):
        img = make_page(40, 40)
        with pytest.raises(ValueError):
            sharpen.unsharp_mask(img, radius=0.0)
        with pytest.raises(ValueError):
            sharpen.unsharp_mask(img, amount=-1.0)

    def test_threshold_suppresses_noise(self):
        """高阈值下平坦噪点区不被放大。"""
        rng = np.random.default_rng(2)
        flat = np.full((100, 100), 128, np.uint8)
        noisy = np.clip(flat + rng.normal(0, 3, flat.shape), 0, 255).astype(np.uint8)
        out = sharpen.unsharp_mask(noisy, amount=2.0, threshold=10)
        assert float(out.std()) < float(noisy.std()) + 5.0


class TestOcrEngineDegradation:
    def test_unavailable_returns_none_not_crash(self):
        """RapidOCR 未安装时：available()=False、recognize()=None，不崩。"""
        from docscan.ocr.engine import OcrEngine

        engine = OcrEngine()
        if engine.available():
            pytest.skip("本机已安装 RapidOCR，降级路径不可测")
        img = make_page(100, 140)
        assert engine.recognize(img) is None
        assert engine.available() is False  # 结果被缓存，不重复 import

    def test_engine_singleton(self):
        from docscan.ocr.engine import get_engine

        assert get_engine() is get_engine()
