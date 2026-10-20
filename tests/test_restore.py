"""退化复原测试：PSF、逆滤波、维纳、倒谱估计。"""

from __future__ import annotations

import contextlib

import cv2
import numpy as np
import pytest

from conftest import make_page
from docscan.core import metrics, restore
from docscan.experiments import synth

SIZE = 128


def _page() -> np.ndarray:
    return cv2.cvtColor(make_page(SIZE, SIZE).astype(np.float32) / 255.0, cv2.COLOR_BGR2GRAY)


class TestPSF:
    def test_motion_psf_normalized_and_size(self):
        psf = restore.motion_psf(31, 15, 30.0)
        assert psf.shape == (31, 31)
        assert psf.sum() == pytest.approx(1.0, abs=1e-9)

    def test_motion_psf_geometry(self):
        """30° 方向的 PSF：能量主轴应与 x 轴约成 30°（角度歧义按 180° 折叠）。"""
        psf = restore.motion_psf(31, 15, 30.0)
        ys, xs = np.nonzero(psf)
        w = psf[ys, xs]
        dx, dy = xs - 15.0, ys - 15.0
        cov = np.cov(np.vstack([dx, dy]), aweights=w)
        evals, evecs = np.linalg.eigh(cov)
        main = evecs[:, int(np.argmax(evals))]
        angle = np.degrees(np.arctan2(main[1], main[0]))
        diff = abs((angle - 30.0 + 90.0) % 180.0 - 90.0)  # 折叠到 90° 内
        assert diff < 5.0, f"主轴角度 {angle:.1f}"

    def test_motion_psf_zero_angle_is_horizontal(self):
        psf = restore.motion_psf(21, 11, 0.0)
        ys, xs = np.nonzero(psf)
        assert (ys == 10).all()  # 同一行
        assert len(set(xs)) == 11  # 11 列

    def test_disk_psf_normalized(self):
        psf = restore.disk_psf(21, 5)
        assert psf.sum() == pytest.approx(1.0, abs=1e-9)
        assert np.count_nonzero(psf) == pytest.approx(np.pi * 25, rel=0.15)

    def test_invalid_params_raise(self):
        with pytest.raises(ValueError):
            restore.motion_psf(30, 10)  # 偶数
        with pytest.raises(ValueError):
            restore.motion_psf(21, 30)  # 长度越界
        with pytest.raises(ValueError):
            restore.disk_psf(21, 11)  # 半径越界


class TestWiener:
    def test_identity_psf_is_identity(self):
        """恒等 PSF（δ 核）+ 极小 K → 输出≈输入（atol 1e-6）。"""
        img = _page()
        delta = np.zeros((3, 3))
        delta[1, 1] = 1.0
        out = restore.wiener(img, delta, K=1e-9)
        assert float(np.abs(out - img).max()) <= 1e-6

    def test_known_psf_improves_psnr_over_3db(self):
        """已知 PSF：维纳复原比退化图 PSNR 提升 > 3 dB。"""
        img = _page()
        blurred = synth.add_motion_blur(img, length=9, angle_deg=25.0, noise_sigma=0.01, seed=42)
        psf = restore.motion_psf(31, 9, 25.0)
        out = restore.wiener(blurred, psf, K=0.005)
        gain = metrics.psnr(img, out) - metrics.psnr(img, blurred)
        assert gain > 3.0, f"PSNR 提升 {gain:.2f} dB"

    def test_K_scan_is_concave(self):
        """K 扫描呈「先升后降」：最优 K 在中间（M6 报告曲线的依据）。"""
        img = _page()
        blurred = synth.add_motion_blur(img, length=7, noise_sigma=0.01, seed=1)
        psf = restore.motion_psf(31, 7, 0.0)
        scores = {
            K: metrics.psnr(img, restore.wiener(blurred, psf, K=K))
            for K in (1e-4, 0.01, 0.1, 1.0)
        }
        peak = max(scores, key=scores.get)
        assert peak not in (1e-4, 1.0), scores  # 峰值在扫描区间内部
        assert scores[0.01] - max(scores[1e-4], scores[1.0]) > 1.0, scores

    def test_negative_K_raises(self):
        with pytest.raises(ValueError):
            restore.wiener(_page(), np.ones((3, 3)) / 9, K=-1.0)


class TestInverseFilter:
    def test_noiseless_blur_restored(self):
        """无噪声退化 + 逆滤波 → PSNR > 30 dB。"""
        img = _page()
        psf = restore.motion_psf(31, 7, 0.0)
        blurred = restore.convolve_blur(img, psf)  # 无噪声
        out = restore.inverse_filter(blurred, psf, eps=1e-6)
        assert metrics.psnr(img, out) > 30.0

    def test_large_eps_is_conservative(self):
        """ε 很大时截断大部分频带，输出应比小 ε 更接近退化图（更保守）。"""
        img = _page()
        psf = restore.disk_psf(21, 4)
        blurred = restore.convolve_blur(img, psf, noise_sigma=0.01)
        d_small = float(np.abs(restore.inverse_filter(blurred, psf, eps=1e-4) - blurred).mean())
        d_large = float(np.abs(restore.inverse_filter(blurred, psf, eps=0.5) - blurred).mean())
        assert d_large < d_small

    def test_inverse_equals_wiener_when_K_zero(self):
        """K=0 的维纳与逆滤波（无截断时）一致。"""
        img = _page()
        psf = restore.motion_psf(31, 5, 0.0)
        blurred = restore.convolve_blur(img, psf)  # 无噪声：|H|≥eps 区不触发
        a = restore.wiener(blurred, psf, K=0.0)
        b = restore.inverse_filter(blurred, psf, eps=1e-12)
        assert float(np.abs(a - b).max()) < 1e-6


class TestCepstrumEstimate:
    @staticmethod
    def _irregular_page() -> np.ndarray:
        """无周期行结构的测试页（文字行的垂直周期峰会干扰倒谱搜索）。"""
        rng = np.random.default_rng(7)
        img = np.full((SIZE, SIZE), 0.92, np.float32)
        for _ in range(24):
            x, y = rng.integers(5, SIZE - 40), rng.integers(5, SIZE - 20)
            w, h = rng.integers(15, 45), rng.integers(4, 9)
            img[y:y + h, x:x + w] = 0.1
        return img

    @pytest.mark.parametrize("angle", [-40.0, -15.0, 0.0, 30.0, 65.0])
    def test_angle_within_10_deg(self, angle):
        """已知运动角度 → 倒谱估计误差 < 10°（按 180° 方向歧义折叠）。"""
        img = self._irregular_page()
        blurred = synth.add_motion_blur(img, length=15, angle_deg=angle, noise_sigma=0.0, seed=3)
        est_angle, est_len = restore.estimate_psf_cepstrum(blurred, min_length=5)
        diff = abs((est_angle - angle + 90.0) % 180.0 - 90.0)
        assert diff < 10.0, f"angle={angle} est={est_angle:.1f}"
        assert 9 <= est_len <= 22, f"length est={est_len}"

    def test_no_blur_raises_or_degrades(self):
        img = self._irregular_page()
        # 无运动模糊时，倒谱找不到显著负峰 → 报错（RuntimeError）是合理行为
        with contextlib.suppress(RuntimeError):
            restore.estimate_psf_cepstrum(img)


class TestSynth:
    def test_all_scenarios_pair_with_gt(self):
        for kind in synth.SCENARIOS:
            # cast 是彩色域操作，用彩色页；其余场景合成器内部转灰度
            img = make_page(SIZE, SIZE).astype(np.float32) / 255.0 if kind == "cast" else _page()
            degraded, gt = synth.synthesize(img, kind)
            assert degraded.shape == gt.shape, kind
            assert degraded.dtype == np.float32, kind
            assert not np.isnan(degraded).any(), kind
            assert float(np.abs(degraded - gt).mean()) > 0.001, kind  # 确实退化了

    def test_unknown_scenario_raises(self):
        with pytest.raises(ValueError):
            synth.synthesize(_page(), "earthquake")

    def test_moire_no_harmonic_clipping(self):
        """摩尔纹合成不得越界削顶（否则频谱出现谐波假峰）。"""
        img = np.full((256, 256), 0.95, np.float32)
        degraded, _ = synth.synthesize(img, "moire", amplitude=0.3)
        # 若发生削顶，直方图会在 1.0 处堆积
        hist = np.histogram(degraded, bins=50, range=(0, 1))[0]
        assert hist[-1] / degraded.size < 0.05
