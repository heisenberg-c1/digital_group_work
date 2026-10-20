"""Runner 编排测试：顺序执行、阶段跳过、失败降级、确定性。"""

from __future__ import annotations

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
        """默认链（几何校正 + 二值化）在斜拍文档上产出二值图。"""
        img, _ = tilted_page
        runner = Runner(build_default_stages())
        result = runner.run(img)
        assert result.order == ["warp", "binarize"]
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
        stage = build_default_stages()[1]  # binarize
        stage.params["block_size"] = 99
        stage.reset_params()
        assert stage.params["block_size"] == 31


class TestPipelineConfig:
    def test_roundtrip_json(self, tmp_path):
        stages = build_default_stages()
        stages[0].enabled = False
        stages[1].params["block_size"] = 41
        cfg = PipelineConfig.from_stages(stages)
        p = cfg.save(tmp_path / "preset.json")

        loaded = PipelineConfig.load(p)
        fresh = build_default_stages()
        loaded.apply_to(fresh)
        assert fresh[0].enabled is False
        assert fresh[1].params["block_size"] == 41
        assert fresh[1].enabled is True

    def test_apply_ignores_unknown_keys(self):
        cfg = PipelineConfig(enabled={"nope": True}, params={"warp": {"nope": 1}})
        fresh = build_default_stages()
        cfg.apply_to(fresh)  # 不应抛异常
        assert fresh[0].enabled is True
