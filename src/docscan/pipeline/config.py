"""处理链配置：阶段开关、参数、输出模式的持久化载体。

M1 只做内存中的默认配置；M3 起增加 JSON 读写（保存/加载预设）。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from docscan.pipeline.stages import Stage


@dataclass
class PipelineConfig:
    """一次运行的完整参数集。"""

    enabled: dict[str, bool] = field(default_factory=dict)
    params: dict[str, dict] = field(default_factory=dict)
    # 输出模式：color / gray / bw
    output_mode: str = "bw"

    def apply_to(self, stages: list[Stage]) -> None:
        """把配置写到阶段对象上。"""
        for stage in stages:
            if stage.name in self.enabled:
                stage.enabled = bool(self.enabled[stage.name])
            if stage.name in self.params:
                for key, value in self.params[stage.name].items():
                    if key in stage.params:
                        stage.params[key] = value

    @classmethod
    def from_stages(cls, stages: list[Stage]) -> PipelineConfig:
        """从阶段对象反向收集配置。"""
        return cls(
            enabled={s.name: s.enabled for s in stages},
            params={s.name: dict(s.params) for s in stages},
        )

    def save(self, path: str | Path) -> Path:
        """保存为 JSON 预设。"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> PipelineConfig:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)
