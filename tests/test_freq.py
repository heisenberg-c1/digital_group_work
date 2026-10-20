"""频域去网纹测试：谱、找峰、陷波器、振铃对比。"""

from __future__ import annotations

import itertools

import cv2
import numpy as np
import pytest

from conftest import add_stripes, make_page
from docscan.core import freq, metrics

SIZE = 256  # 2 的幂，条纹波长 32 → f0 恰好落在第 8 个 bin


def _striped_doc():
    """灰度文档 + 已知频率的水平条纹。

    注意：页面先压到 0.72 再叠加 0.25 条纹，确保合成不越界 [0,1]——
    越界裁剪会产生谐波失真，让频谱出现假峰（实验教训，见实验记录）。
    """
    page = make_page(SIZE, SIZE).astype(np.float32) / 255.0
    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    base = (gray * 0.72).astype(np.float32)
    return base, add_stripes(base, wavelength=32, angle_deg=0.0, amplitude=0.25)


class TestSpectrum:
    def test_constant_image_only_dc(self):
        """常量图：除 DC 外频谱能量应为 ~0（数值噪声级别）。"""
        img = np.full((SIZE, SIZE), 0.5, np.float32)
        spec = freq.magnitude_spectrum(img)
        h, w = spec.shape
        cy, cx = h // 2, w // 2
        spec[cy, cx] = 0.0
        assert float(spec.max()) < 1e-6

    def test_spectrum_normalized(self):
        _, striped = _striped_doc()
        spec = freq.magnitude_spectrum(striped)
        assert spec.min() >= 0.0 and spec.max() <= 1.0


class TestFindPeaks:
    def test_known_frequency_within_one_bin(self):
        """λ=16、角度 0 → 峰应在 (±16, 0)，误差 < 1 bin。

        用无文字行的空白页：文字行本身有强周期性，会盖过条纹峰——
        这正是「先合成、后真实」的原因。波长 16 保证 f0=16 bin
        落在默认 DC 屏蔽半径之外。
        """
        blank = np.full((SIZE, SIZE), 0.6, np.float32)
        striped = add_stripes(blank, wavelength=16, angle_deg=0.0, amplitude=0.3)
        spec = freq.magnitude_spectrum(striped)
        peaks = freq.find_periodic_peaks(spec, top_n=2)
        found = {(round(du), round(dv)) for du, dv in peaks}
        assert any(abs(du - 16) <= 1 and abs(dv) <= 1 for du, dv in found), found
        assert any(abs(du + 16) <= 1 and abs(dv) <= 1 for du, dv in found), found

    def test_diagonal_stripes(self):
        page = np.full((SIZE, SIZE), 0.8, np.float32)
        striped = add_stripes(page, wavelength=32, angle_deg=45.0, amplitude=0.4)
        spec = freq.magnitude_spectrum(striped)
        peaks = freq.find_periodic_peaks(spec, top_n=4)
        # 45° 条纹 → 峰沿 (1,1)/√2 方向：du≈dv
        found = [(du, dv) for du, dv in peaks if abs(abs(du) - abs(dv)) <= 2 and abs(du) > 2]
        assert found, peaks

    def test_respects_dc_exclusion(self):
        """DC 半径内不允许出峰。"""
        _, striped = _striped_doc()
        spec = freq.magnitude_spectrum(striped)
        peaks = freq.find_periodic_peaks(spec, dc_radius=20)
        assert all(abs(du) > 20 or abs(dv) > 20 or (du**2 + dv**2) > 400 for du, dv in peaks)


class TestNotchMasks:
    def test_mask_symmetry(self):
        """H(u,v) == H(-u,-v)（实数信号的共轭对称性）。"""
        h = w = 128
        mask = freq.ideal_notch_mask((h, w), [(10, 3)], 4)
        bw = freq.butterworth_notch_mask((h, w), [(10, 3)], 4, order=2)
        for m in (mask, bw):
            cy, cx = h // 2, w // 2
            assert m[cy, cx] > 0.9  # DC 远离陷波点，响应接近 1（BW 为两响应乘积）
            # 手工取对称点对比
            assert m[cy + 3, cx + 10] == m[cy - 3, cx - 10]

    def test_ideal_mask_binary(self):
        mask = freq.ideal_notch_mask((64, 64), [(5, 0)], 3)
        assert set(np.unique(mask)) == {0.0, 1.0}

    def test_butterworth_monotonic_toward_ideal(self):
        """阶数增大时，陷波中心响应单调下降 → 趋近理想陷波。"""
        shape = (64, 64)
        responses = []
        for order in (1, 2, 4, 8, 16):
            m = freq.butterworth_notch_mask(shape, [(8, 0)], 3, order=order)
            responses.append(m[32, 40])  # 峰点处
        assert all(a > b for a, b in itertools.pairwise(responses)), responses

    def test_invalid_params_raise(self):
        with pytest.raises(ValueError):
            freq.butterworth_notch_mask((64, 64), [(3, 0)], 3, order=0)
        with pytest.raises(ValueError):
            freq.notch_filter(np.zeros((32, 32), np.float32), [(3, 0)], kind="wrong")


class TestNotchFilter:
    def test_energy_drop_over_20db_and_text_preserved(self):
        """核心断言：f0 能量下降 > 20 dB 且文字区域 SSIM > 0.90。

        峰用手动指定（模拟 UI 手动选点路径）：真实文档上文字行的
        周期性能量与网纹混在一起，自动找峰只用于无结构背景。
        """
        clean, striped = _striped_doc()
        peaks = [(8, 0), (-8, 0)]

        out = freq.notch_filter(striped, peaks, kind="butterworth", radius=3, order=4)
        drop = freq.energy_drop_db(striped, out, peaks[0])
        assert drop < -20.0, f"能量仅下降 {-drop:.1f} dB"

        s = metrics.ssim(clean, out)
        assert s > 0.90, f"SSIM={s:.3f}"

    def test_auto_peaks_on_blank_background(self):
        """自动找峰 + 陷波在无结构背景上闭环工作。"""
        blank = np.full((SIZE, SIZE), 0.6, np.float32)
        striped = add_stripes(blank, wavelength=16, amplitude=0.3)
        spec = freq.magnitude_spectrum(striped)
        peaks = freq.find_periodic_peaks(spec, top_n=2)
        assert len(peaks) >= 2
        out = freq.notch_filter(striped, peaks, kind="butterworth", radius=3, order=4)
        assert freq.energy_drop_db(striped, out, peaks[0]) < -20.0

    def test_ideal_also_works(self):
        _, striped = _striped_doc()
        out = freq.notch_filter(striped, [(8, 0), (-8, 0)], kind="ideal", radius=3)
        assert freq.energy_drop_db(striped, out, (8, 0)) < -20.0

    def test_butterworth_rings_less_than_ideal(self):
        """报告核心论据：Butterworth 的振铃指标 < 理想陷波。

        用「滤波输出相对干净参考图、在文字边缘环带内的平均绝对差」
        度量振铃——理想陷波频域矩形截断 → 空域 sinc 波纹 → 差值大。
        """
        base, striped = _striped_doc()
        peaks = [(8, 0), (-8, 0)]
        r_ideal = metrics.ringing_index(
            freq.notch_filter(striped, peaks, kind="ideal", radius=3), base
        )
        r_bw = metrics.ringing_index(
            freq.notch_filter(striped, peaks, kind="butterworth", radius=3, order=4), base
        )
        assert r_bw < r_ideal, f"ideal={r_ideal:.4f} butterworth={r_bw:.4f}"
        assert (r_ideal - r_bw) / r_ideal > 0.03  # 区分度至少 3%


class TestMetrics:
    def test_psnr_identical_is_inf(self):
        img = make_page(64, 64)
        assert metrics.psnr(img, img) == float("inf")

    def test_psnr_matches_analytic_value(self):
        """高斯噪声 σ=0.01 → PSNR = -20·log10(0.01) = 40 dB（差 < 0.05）。"""
        rng = np.random.default_rng(0)
        img = np.full((200, 200), 0.5, np.float32)
        noisy = np.clip(img + rng.normal(0, 0.01, img.shape), 0, 1).astype(np.float32)
        expected = -20.0 * np.log10(0.01)
        assert abs(metrics.psnr(img, noisy) - expected) < 0.05

    def test_ssim_identical_is_one(self):
        img = make_page(64, 64)
        assert metrics.ssim(img, img) == pytest.approx(1.0, abs=1e-9)

    def test_ssim_matches_skimage_oracle(self):
        """与 scikit-image 对拍：同一高斯窗参数下差异 < 0.01。"""
        skimage = pytest.importorskip("skimage.metrics")
        rng = np.random.default_rng(1)
        base = rng.uniform(0.3, 0.7, size=(128, 128)).astype(np.float32)
        other = np.clip(base + rng.normal(0, 0.05, base.shape), 0, 1).astype(np.float32)
        mine = metrics.ssim(base, other)
        oracle = skimage.structural_similarity(
            base, other, data_range=1.0, gaussian_weights=True, sigma=1.5, use_sample_covariance=False
        )
        assert abs(mine - oracle) < 0.01, f"mine={mine:.4f} oracle={oracle:.4f}"

    def test_ssim_constant_images_no_nan(self):
        a = np.full((32, 32), 0.4, np.float32)
        b = np.full((32, 32), 0.6, np.float32)
        v = metrics.ssim(a, b)
        assert np.isfinite(v)

    def test_sharpness_orders_images(self):
        sharp = make_page(200, 280)
        blurred = freq.notch_filter(sharp, [], kind="butterworth", radius=200)  # 高斯平滑替代
        # 直接用 cv2 高斯模糊做对照
        import cv2

        blurred = cv2.GaussianBlur(sharp.astype(np.float32) / 255.0, (9, 9), 3.0)
        assert metrics.sharpness(sharp) > metrics.sharpness(blurred)
