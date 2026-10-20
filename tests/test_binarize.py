"""Sauvola 二值化测试：与 skimage oracle 对拍。"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_page
from docscan.core import binarize


class TestSauvola:
    def test_matches_skimage_oracle(self):
        """与 skimage.filters.threshold_sauvola 逐像素对拍（差异 ≤ 0.1%）。"""
        skimage_filters = pytest.importorskip("skimage.filters")
        rng = np.random.default_rng(3)
        img = (rng.uniform(0.3, 0.95, size=(120, 160)) * 255).astype(np.uint8)
        img[40:60, 30:130] = 30  # 一块「文字」

        mine = binarize.sauvola(img, window=25, k=0.2)
        t = skimage_filters.threshold_sauvola(img, window_size=25, k=0.2)
        oracle = ((img.astype(np.float64) > t) * 255).astype(np.uint8)

        mismatch = float((mine != oracle).mean())
        assert mismatch <= 0.001, f"不匹配比例 {mismatch:.4%}"

    def test_document_binary_has_ink_and_paper(self):
        img = make_page(200, 280)
        out = binarize.sauvola(img, window=25)
        assert set(np.unique(out)).issubset({0, 255})
        ink = float((out == 0).mean())
        assert 0.01 < ink < 0.5, f"墨迹占比 {ink:.2%}"

    def test_even_window_raises(self):
        with pytest.raises(ValueError):
            binarize.sauvola(np.zeros((50, 50), np.uint8), window=24)
        with pytest.raises(ValueError):
            binarize.sauvola(np.zeros((50, 50), np.uint8), window=1)

    def test_adaptive_even_block_raises(self):
        with pytest.raises(ValueError):
            binarize.adaptive(np.zeros((50, 50), np.uint8), block_size=30)

    def test_clean_small_components_removes_specks(self):
        """100 个单像素噪点全部被清除。"""
        img = np.full((300, 300), 255, np.uint8)
        rng = np.random.default_rng(5)
        for _ in range(100):
            img[rng.integers(0, 300), rng.integers(0, 300)] = 0
        out = binarize.clean_small_components(img, min_area=8)
        assert int((out == 0).sum()) == 0

    def test_clean_keeps_real_ink(self):
        img = np.full((200, 200), 255, np.uint8)
        img[80:90, 40:160] = 0  # 一条 10×120 的笔画
        out = binarize.clean_small_components(img, min_area=8)
        assert int((out == 0).sum()) == 10 * 120


class TestOtsu:
    def test_bimodal_threshold_between_peaks(self):
        img = np.zeros((100, 100), np.uint8)
        img[:50] = 40
        img[50:] = 200
        out = binarize.otsu(img)
        assert set(np.unique(out)) == {0, 255}
        assert (out[:50] == 0).all() and (out[50:] == 255).all()
