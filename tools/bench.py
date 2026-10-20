"""基准测试：冷启动耗时 + 12MP 端到端耗时。

用法::

    python tools/bench.py [--repeat 3]

目标（评分表）：冷启动 < 1s，12MP 端到端 < 3s。
冷启动计时的是 `import docscan`（不含 tkinter 窗口构建，与打包后的
可执行文件冷启有差异，打包版另测）。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from docscan.experiments.utils import render_text_page  # noqa: E402
from docscan.pipeline.runner import Runner  # noqa: E402
from docscan.pipeline.stages import build_default_stages  # noqa: E402


def bench_cold_start(repeat: int) -> float:
    """子进程 import 计时（避开本进程已加载模块的干扰）。"""
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        subprocess.run(
            [sys.executable, "-c", "import docscan, docscan.pipeline.stages"],
            check=True,
            cwd=str(REPO),
        )
        times.append((time.perf_counter() - t0) * 1000)
    return min(times)


def make_12mp_page() -> np.ndarray:
    """渲染 12MP 级别的文字页（4000×3000）。"""
    page, _ = render_text_page(width=4000, height=3000, font_px=120, margin=280)
    return page


def bench_12mp(repeat: int) -> dict[str, float]:
    page = make_12mp_page()
    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    img = gray.astype(np.float32) / 255.0

    totals, warp_ms = [], []
    for _ in range(repeat):
        stages = build_default_stages()
        runner = Runner(stages)
        t0 = time.perf_counter()
        result = runner.run(img)
        total = (time.perf_counter() - t0) * 1000
        totals.append(total)
        if "warp" in result.stage_outputs:
            warp_ms.append(result.stage_outputs["warp"].elapsed_ms)
    return {
        "total_ms": min(totals),
        "warp_ms": min(warp_ms) if warp_ms else 0.0,
        "mp": page.shape[0] * page.shape[1] / 1e6,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="docscan 基准测试")
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args(argv)

    cold = bench_cold_start(args.repeat)
    print(f"冷启动 import（子进程，取最优）: {cold:.0f} ms  [目标 < 1000 ms]")

    r = bench_12mp(args.repeat)
    print(f"12MP({r['mp']:.1f}MP) 端到端（取最优）: {r['total_ms']:.0f} ms  [目标 < 3000 ms]")
    print(f"  其中几何校正检测（缩略图）: {r['warp_ms']:.0f} ms")
    ok = cold < 1000 and r["total_ms"] < 3000
    print("结论:", "达标 ✓" if ok else "未达标 ✗（建议启用预览模式策略，见架构文档 §4）")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
