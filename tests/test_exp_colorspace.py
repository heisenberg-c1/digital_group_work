"""色彩空间实验的指标口径测试。

重点守住两处容易算错的细节：色相是圆周量（359° 与 1° 只差 2°），
以及「只改 V」必须精确保持色相——否则实验结论本身不成立。
"""

from __future__ import annotations

import numpy as np

from docscan.core import colorspace
from docscan.experiments.exp_colorspace import (
    _curve,
    _hue_stats,
    _per_channel,
)


def _bgr_from_hsv(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    return colorspace.hsv_to_bgr(h, s, v)


class TestHueStats:
    def test_hue_wraps_around_zero(self):
        """359° → 0° 是 1° 漂移，不是 359°。"""
        ref = _bgr_from_hsv(np.array([[359.0]]), np.ones((1, 1)), np.full((1, 1), 255.0))
        out = _bgr_from_hsv(np.array([[0.0]]), np.ones((1, 1)), np.full((1, 1), 255.0))
        drift, _ = _hue_stats(ref, out)
        assert drift < 2.0, f"圆周距离算错：{drift:.2f}°"

    def test_gray_pixels_are_ignored(self):
        """灰色纸面饱和度低于阈值，不参与色相统计（否则噪声会污染结论）。"""
        gray = np.full((1, 4, 3), 0.5, np.float32)
        brighter = np.full((1, 4, 3), 0.8, np.float32)
        drift, sat = _hue_stats(gray, brighter)
        assert drift == 0.0 and sat == 0.0


class TestBrightnessPaths:
    def test_value_path_keeps_hue_exactly(self):
        """V 是乘性缩放，色相与饱和度理论上逐像素不变。"""
        hsv_h = np.array([[0.0, 60.0, 120.0, 200.0, 300.0]])
        hsv_s = np.full((1, 5), 0.7)
        hsv_v = np.full((1, 5), 120.0)
        ref = _bgr_from_hsv(hsv_h, hsv_s, hsv_v)

        out = colorspace.apply_on_value(ref, _curve)
        drift, sat = _hue_stats(ref, out)
        assert drift < 0.01, f"仅改 V 不应漂移色相，实测 {drift:.3f}°"
        assert abs(sat) < 0.01

    def test_per_channel_shifts_hue_more_than_value_path(self):
        """对照组（逐通道）会真的改变颜色，这正是要求②要说明的问题。"""
        hsv_h = np.array([[10.0, 90.0, 150.0, 220.0, 310.0]])
        hsv_s = np.full((1, 5), 0.85)
        hsv_v = np.full((1, 5), 110.0)
        ref = _bgr_from_hsv(hsv_h, hsv_s, hsv_v)

        value_out = colorspace.apply_on_value(ref, _curve)
        bgr_out = _per_channel(ref)
        drift_value, _ = _hue_stats(ref, value_out)
        drift_bgr, _ = _hue_stats(ref, bgr_out)
        assert drift_bgr > drift_value + 0.1

    def test_curve_is_monotonic_and_in_range(self):
        x = np.linspace(0, 255, 32)
        y = _curve(x)
        assert y.min() >= 0.0 and y.max() <= 255.0
        assert np.all(np.diff(y) >= 0.0)
