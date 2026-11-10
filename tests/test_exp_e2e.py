"""端到端实验的场景解析与策略表测试。

只测纯逻辑（不跑整条链）：``_resolve_scene`` 是「静默降级」缺陷 D2 的守卫点。
"""

from __future__ import annotations

import pytest

from docscan.experiments.exp_e2e import STAGE_POLICY, _resolve_scene


class TestSceneResolution:
    @pytest.mark.parametrize("scene", ["normal", "perspective", "shadow", "moire",
                                       "motion", "defocus", "cast"])
    def test_all_seven_scenes_resolve(self, scene):
        assert _resolve_scene(scene, synthetic=True) == scene

    def test_suffix_is_stripped(self):
        """真实照片常带后缀（defocus_01.jpg），只取第一段作为场景。"""
        assert _resolve_scene("defocus_01", synthetic=True) == "defocus"

    def test_case_insensitive(self):
        assert _resolve_scene("DEFOCUS", synthetic=True) == "defocus"

    def test_unknown_scene_raises_on_synthetic(self):
        with pytest.raises(ValueError, match="未知场景"):
            _resolve_scene("blur", synthetic=True)

    def test_unknown_scene_falls_back_on_real_photo(self, caplog):
        """真实照片没有场景标注，允许回退，但必须留下 warning。"""
        assert _resolve_scene("blur", synthetic=False) == "normal"
        assert "未识别" in caplog.text


class TestStagePolicy:
    def test_defocus_and_cast_have_real_degradation_path(self):
        """D2/D4 回归：这两个场景必须各自启用复原/白平衡，不能退化成 normal。"""
        assert STAGE_POLICY["defocus"]["restore"] is True
        assert STAGE_POLICY["cast"]["white_balance"] is True
        assert STAGE_POLICY["cast"] != STAGE_POLICY["normal"]

    def test_motion_does_not_enable_sharpen(self):
        """motion 场景开 USM 会让 SSIM 低于不处理，默认必须关闭。"""
        assert STAGE_POLICY["motion"].get("sharpen", False) is False
