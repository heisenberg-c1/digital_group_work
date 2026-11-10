"""Runner 编排测试：顺序执行、阶段跳过、失败降级、确定性。"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from conftest import make_page
from docscan.core.types import to_uint8
from docscan.pipeline.config import PipelineConfig
from docscan.pipeline.runner import Runner
from docscan.pipeline.stages import Stage, build_default_stages


def _noop_stage(name: str, calls: list) -> Stage:
    def func(img, params):
        calls.append(name)
        return img, "ok", {}

    return Stage(name=name, label=name, func=func)


class TestRunner:
    def test_runs_stages_in_order(self):
        calls: list[str] = []
        runner = Runner([_noop_stage("a", calls), _noop_stage("b", calls)])
        img = make_page(200, 300)
        result = runner.run(img)
        assert calls == ["a", "b"]
        assert result.order == ["a", "b"]
        assert result.final is not None
        assert result.total_ms >= 0

    def test_disabled_stage_not_called(self):
        """关闭的阶段函数必须完全不被调用（mock 断言）。"""
        calls: list[str] = []
        s1 = _noop_stage("a", calls)
        s2 = _noop_stage("b", calls)
        runner = Runner([s1, s2])
        runner.set_enabled("b", False)
        result = runner.run(make_page(100, 100))
        assert calls == ["a"]
        assert result.skipped == ["b"]
        assert "b" not in result.stage_outputs

    def test_stage_failure_degrades_not_crashes(self):
        """单阶段抛异常：记录告警，后续阶段继续执行。"""
        calls: list[str] = []

        def bad(img, params):
            raise ZeroDivisionError("除零")

        runner = Runner(
            [
                Stage("bad", "坏阶段", bad),
                _noop_stage("next", calls),
            ]
        )
        result = runner.run(make_page(100, 100))
        assert any("坏阶段" in w for w in result.warnings)
        assert calls == ["next"]
        assert result.final is not None

    def test_progress_callback_receives_each_stage(self):
        seen: list[str] = []
        runner = Runner([_noop_stage("a", []), _noop_stage("b", [])])
        runner.run(make_page(100, 100), progress_cb=lambda name, _sr: seen.append(name))
        assert seen == ["a", "b"]

    def test_deterministic_output(self):
        """相同输入两次运行，最终输出逐字节一致（黄金用例的基础）。"""
        runner = Runner(build_default_stages())
        img = make_page(400, 560)
        r1 = runner.run(img)
        r2 = runner.run(img)
        np.testing.assert_array_equal(to_uint8(r1.final), to_uint8(r2.final))

    def test_default_chain_end_to_end(self, tilted_page):
        """默认链（几何校正 + 光照归一 + 二值化）在斜拍文档上产出二值图。"""
        img, _ = tilted_page
        runner = Runner(build_default_stages())
        result = runner.run(img)
        assert result.order == ["warp", "illumination", "binarize"]
        final = to_uint8(result.final)
        assert final.ndim == 2
        assert set(np.unique(final)).issubset({0, 255})
        # 校正成功时置信度指标被记录
        assert result.stage_outputs["warp"].metrics["confidence"] > 0.6

    def test_unknown_stage_raises(self):
        runner = Runner(build_default_stages())
        with pytest.raises(KeyError):
            runner.set_enabled("not_exist", True)


class TestStageParams:
    def test_params_passed_to_func(self):
        received: dict = {}

        def func(img, params):
            received.update(params)
            return img, "", {}

        stage = Stage("s", "s", func, params={"block_size": 15})
        Runner([stage]).run(make_page(80, 80))
        assert received["block_size"] == 15

    def test_reset_params_restores_defaults(self):
        stages = {s.name: s for s in build_default_stages()}
        stages["binarize"].params["block_size"] = 99
        stages["binarize"].reset_params()
        assert stages["binarize"].params["block_size"] == 31


class TestStageBehaviours:
    """针对审计缺陷的回归测试：这些路径原先没有测试覆盖。"""

    def test_denoise_stage_exists_and_disabled_by_default(self):
        stages = {s.name: s for s in build_default_stages()}
        assert "denoise" in stages
        assert stages["denoise"].enabled is False
        # 顺序：增强之后、去网纹之前（先去噪再陷波/复原）
        names = [s.name for s in build_default_stages()]
        assert names.index("enhance") < names.index("denoise") < names.index("freq")

    def test_enhance_none_applies_only_gamma(self):
        from docscan.pipeline.stages import stage_enhance

        img = make_page(120, 160)
        out, note, _ = stage_enhance(img, {"method": "none", "gamma": 1.0})
        assert note == "不增强"
        np.testing.assert_array_equal(out, img)

    def test_warp_accepts_grayscale(self):
        """灰度输入不应让几何校正阶段直接抛异常（否则灰度链静默跳过校正）。"""
        from docscan.pipeline.stages import stage_warp

        gray = make_page(400, 560)[..., 0]
        out, note, metrics_out = stage_warp(gray, {"min_confidence": 0.6})
        assert out.ndim == 2
        assert "confidence" in metrics_out
        assert "需要三通道" not in note

    def test_restore_defocus_and_inverse(self):
        from docscan.pipeline.stages import stage_restore

        img = make_page(240, 320)
        _, note, m = stage_restore(
            img, {"psf_type": "defocus", "defocus_radius": 5, "filter": "wiener", "K": 0.01}
        )
        assert "散焦" in note and m["psf_radius"] == 5.0

        _, note2, _ = stage_restore(
            img,
            {
                "psf_type": "motion", "psf_mode": "manual", "length": 9,
                "angle_deg": 0.0, "filter": "inverse", "eps": 1e-3,
            },
        )
        assert "逆滤波" in note2 and "手动 PSF" in note2

    def test_restore_reports_failure_instead_of_raising(self):
        """倒谱估不出 PSF 时返回原图 + 明确说明，而不是把异常抛给 Runner。"""
        from docscan.pipeline.stages import stage_restore

        flat = np.full((200, 200, 3), 200, np.uint8)
        out, note, _ = stage_restore(flat, {"psf_type": "motion", "psf_mode": "est"})
        assert out is flat
        assert "失败" in note or "倒谱" in note

    def test_freq_stage_flags_ineffective_notch(self):
        """陷波无效（伪峰）时必须置 notch_ok=0 并在 note 里提示。"""
        from docscan.pipeline.stages import stage_freq_denoise

        img = np.full((200, 200, 3), 200, np.uint8)  # 平坦图：非 DC 处无能量
        _, note, m = stage_freq_denoise(img, {"manual_peaks": [(20, 20)], "radius": 4})
        assert m["notch_ok"] == 0.0
        assert "手动选点" in note


class TestPipelineConfig:
    def test_roundtrip_json(self, tmp_path):
        stages = build_default_stages()
        by_name = {s.name: s for s in stages}
        by_name["warp"].enabled = False
        by_name["binarize"].params["block_size"] = 41
        cfg = PipelineConfig.from_stages(stages)
        p = cfg.save(tmp_path / "preset.json")

        loaded = PipelineConfig.load(p)
        fresh = build_default_stages()
        loaded.apply_to(fresh)
        fresh_by_name = {s.name: s for s in fresh}
        assert fresh_by_name["warp"].enabled is False
        assert fresh_by_name["binarize"].params["block_size"] == 41
        assert fresh_by_name["binarize"].enabled is True

    def test_apply_ignores_unknown_keys(self):
        cfg = PipelineConfig(enabled={"nope": True}, params={"warp": {"nope": 1}})
        fresh = build_default_stages()
        cfg.apply_to(fresh)  # 不应抛异常
        assert fresh[0].enabled is True


class TestEnhanceStageMethods:
    """D5 回归：空间域方法必须真的接入，且未知方法不得静默回退。"""

    @pytest.mark.parametrize("method", ["clahe", "histeq", "stretch", "none"])
    def test_each_method_produces_its_own_output(self, method):
        from docscan.pipeline.stages import stage_enhance

        img = make_page(160, 200)
        out, note, _ = stage_enhance(img, {"method": method, "gamma": 1.0})
        assert note and out.shape == img.shape
        if method == "none":
            np.testing.assert_array_equal(out, img)
        else:
            assert not np.array_equal(out, img), f"{method} 未改变图像，等于空转"

    def test_he_alias_maps_to_histeq(self):
        """组员版实验脚本用 ``he``，必须落到直方图均衡而不是静默走 CLAHE。"""
        from docscan.pipeline.stages import stage_enhance

        img = make_page(160, 200)
        by_alias, note_alias, _ = stage_enhance(img, {"method": "he", "gamma": 1.0})
        by_name, note_name, _ = stage_enhance(img, {"method": "histeq", "gamma": 1.0})
        assert note_alias == note_name == "直方图均衡"
        np.testing.assert_array_equal(by_alias, by_name)

    def test_unknown_method_raises(self):
        from docscan.pipeline.stages import stage_enhance

        with pytest.raises(ValueError, match="未知增强方法"):
            stage_enhance(make_page(120, 160), {"method": "bogus"})

    def test_stretch_percentiles_are_honoured(self):
        from docscan.pipeline.stages import stage_enhance

        # 低对比渐变：分位不同则拉伸后的斜率不同（整页纯白/纯黑看不出来）
        ramp = (0.40 + 0.20 * np.linspace(0, 1, 160)[None, :]).astype(np.float32)
        img = np.repeat(ramp, 120, axis=0)
        narrow, note_narrow, _ = stage_enhance(
            img, {"method": "stretch", "low_pct": 5.0, "high_pct": 95.0}
        )
        wide, _, _ = stage_enhance(
            img, {"method": "stretch", "low_pct": 0.0, "high_pct": 100.0}
        )
        assert "5%~95%" in note_narrow
        assert not np.array_equal(narrow, wide)


class TestRestoreStagePsfSelection:
    """D6 回归：散焦必须用圆盘 PSF，且未知枚举值不得静默回退。"""

    @staticmethod
    def _defocused():
        from docscan.core import restore

        page = make_page(240, 320)
        gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        return gray, cv2.filter2D(gray, -1, restore.disk_psf(31, 5))

    def test_disk_psf_beats_motion_psf_on_defocus(self):
        from docscan.core import metrics
        from docscan.pipeline.stages import stage_restore

        gt, degraded = self._defocused()
        disk, _, m = stage_restore(
            degraded, {"psf_type": "defocus", "defocus_radius": 5, "K": 0.003}
        )
        motion, _, _ = stage_restore(
            degraded,
            {"psf_type": "motion", "psf_mode": "manual", "length": 11,
             "angle_deg": 0.0, "K": 0.003},
        )
        ssim_disk = metrics.ssim(gt, disk)
        ssim_motion = metrics.ssim(gt, motion)
        assert m["psf_radius"] == 5.0
        assert ssim_disk > ssim_motion, (
            f"圆盘 {ssim_disk:.4f} 应优于线段 {ssim_motion:.4f}"
        )

    def test_disk_alias_accepted(self):
        from docscan.pipeline.stages import stage_restore

        _, degraded = self._defocused()
        _, note, m = stage_restore(degraded, {"psf_type": "disk", "defocus_radius": 5})
        assert "散焦" in note and m["psf_radius"] == 5.0

    @pytest.mark.parametrize(
        ("params", "match"),
        [
            ({"psf_type": "bogus"}, "未知 PSF 类型"),
            ({"psf_type": "motion", "psf_mode": "bogus"}, "未知 PSF 来源"),
            ({"psf_type": "motion", "filter": "bogus"}, "未知复原滤波器"),
        ],
    )
    def test_unknown_enum_raises(self, params, match):
        from docscan.pipeline.stages import stage_restore

        with pytest.raises(ValueError, match=match):
            stage_restore(make_page(120, 160), params)
