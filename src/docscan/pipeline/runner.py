"""Runner：顺序执行处理链，收集每个阶段的输出与指标。

线程模型：``Runner.run`` 是纯同步函数（可在任意线程调用）；
UI 层用 worker 线程跑它，通过队列把 ``StageResult`` 逐个回传，
UI 线程只消费队列 —— worker 绝不直接触碰 tkinter。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from docscan.core.types import Image, StageResult, to_float
from docscan.pipeline.stages import Stage


@dataclass
class RunResult:
    """一次完整处理的结果。"""

    final: Image
    stage_outputs: dict[str, StageResult] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    total_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def notes_text(self) -> str:
        """把全部阶段决策拼成多行文本，供日志面板显示。"""
        lines = []
        for name in self.order:
            r = self.stage_outputs[name]
            lines.append(f"[{name}] {r.notes} ({r.elapsed_ms:.0f} ms)")
        for name in self.skipped:
            lines.append(f"[{name}] 已跳过")
        for w in self.warnings:
            lines.append(f"[警告] {w}")
        return "\n".join(lines)


class Runner:
    """处理链执行器。

    Args:
        stages: 有序阶段列表。
        to_float_domain: 为 True 时算法在 float32[0,1] 上执行。
            当前二值化要求 uint8，故默认 False；M2 引入光照归一后切换。
    """

    def __init__(self, stages: list[Stage], *, to_float_domain: bool = False) -> None:
        self.stages = stages
        self.to_float_domain = to_float_domain

    def run(
        self,
        image: Image,
        progress_cb: Callable[[str, StageResult], None] | None = None,
    ) -> RunResult:
        """顺序执行所有启用阶段。

        Args:
            image: BGR 输入图。
            progress_cb: 每完成一个阶段回调一次 ``(stage_name, StageResult)``，
                worker 线程用它把结果放进队列。

        Returns:
            RunResult，含每个阶段输出、最终图与告警。
        """
        started = time.perf_counter()
        result = RunResult(final=image)
        current = image

        if self.to_float_domain:
            current = to_float(current)

        for stage in self.stages:
            if not stage.enabled:
                result.skipped.append(stage.name)
                continue
            t0 = time.perf_counter()
            try:
                out, notes, metrics = stage.func(current, dict(stage.params))
            except Exception as exc:
                result.warnings.append(f"阶段 {stage.label} 执行失败：{exc}")
                continue
            sr = StageResult(
                image=out,
                notes=notes or "完成",
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                metrics=metrics,
            )
            result.stage_outputs[stage.name] = sr
            result.order.append(stage.name)
            current = out
            if progress_cb is not None:
                progress_cb(stage.name, sr)

        result.final = current
        result.total_ms = (time.perf_counter() - started) * 1000.0
        return result

    def set_enabled(self, name: str, enabled: bool) -> None:
        """开关某个阶段。"""
        for stage in self.stages:
            if stage.name == name:
                stage.enabled = enabled
                return
        raise KeyError(f"未知阶段：{name}")

    def get_stage(self, name: str) -> Stage:
        for stage in self.stages:
            if stage.name == name:
                return stage
        raise KeyError(f"未知阶段：{name}")
