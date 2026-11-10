"""命令行入口。

GUI:  ``python -m docscan``
CLI:  ``python -m docscan --cli samples/demo/a.jpg -o out.png --mode bw``

CLI 与 GUI **共用同一条处理链**（``pipeline.Runner`` + ``build_default_stages``），
差别只在于：CLI 用参数决定阶段开关，GUI 用勾选框。这样「批处理」和「界面」
不会给出两个不同质量的结果——早期版本 CLI 只做「校正 + 二值化」，光照归一、
去网纹、复原全部缺失，等于同一个功能的两套实现。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

from docscan import __version__
from docscan.core import binarize
from docscan.core.types import Image
from docscan.io.exporter import export_pdf, save_image
from docscan.io.loader import load_any
from docscan.pipeline.config import PipelineConfig
from docscan.pipeline.runner import Runner
from docscan.pipeline.stages import Stage, build_default_stages

log = logging.getLogger("docscan")

# --enable 允许打开的附加阶段（warp/illumination/binarize 由专门参数控制）
_EXTRA_STAGES = ("white_balance", "enhance", "denoise", "freq", "restore", "sharpen")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="docscan",
        description="手机文档扫描与可读性增强系统",
    )
    parser.add_argument("inputs", nargs="*", help="输入图像（jpg/png/bmp/tif/pdf）")
    parser.add_argument("--cli", action="store_true", help="无界面批处理模式")
    parser.add_argument("-o", "--output", help="输出路径（单输入时）或输出目录")
    parser.add_argument(
        "--mode",
        choices=("color", "gray", "bw"),
        default="bw",
        help="输出模式：彩色 / 灰度 / 黑白（默认 bw）",
    )
    parser.add_argument("--no-warp", action="store_true", help="跳过几何校正")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.6,
        help="角点置信度阈值，低于此值退化为整图（默认 0.6）",
    )
    parser.add_argument("--block-size", type=int, default=31, help="自适应阈值邻域边长（奇数）")
    parser.add_argument("--c", type=float, default=10.0, help="自适应阈值偏置常数")
    parser.add_argument(
        "--enable",
        default="",
        help=f"额外启用的阶段，逗号分隔（{'/'.join(_EXTRA_STAGES)}）",
    )
    parser.add_argument("--preset", help="从 JSON 预设加载阶段开关与参数")
    parser.add_argument("-v", "--verbose", action="store_true", help="输出详细日志")
    parser.add_argument("--version", action="version", version=f"docscan {__version__}")
    return parser


def _build_stages(args: argparse.Namespace) -> list[Stage]:
    """把命令行参数翻译成阶段配置（与 GUI 用同一套 Stage 对象）。"""
    stages = build_default_stages()
    by_name = {s.name: s for s in stages}

    by_name["warp"].enabled = not args.no_warp
    by_name["warp"].params["min_confidence"] = args.min_confidence
    by_name["binarize"].params["block_size"] = args.block_size
    by_name["binarize"].params["c"] = args.c

    for name in [x.strip() for x in args.enable.split(",") if x.strip()]:
        if name not in by_name:
            raise ValueError(f"未知阶段：{name}（可选：{', '.join(by_name)}）")
        by_name[name].enabled = True

    if args.preset:
        PipelineConfig.load(args.preset).apply_to(stages)
        # 预设之后仍按命令行收敛：输出档位与是否校正由用户显式指定优先
        by_name["warp"].enabled = not args.no_warp

    by_name["binarize"].enabled = args.mode == "bw"
    return stages


def _finalize(img: Image, mode: str) -> Image:
    """按输出模式收尾：彩色保三通道、灰度转单通道、黑白确保是二值图。"""
    if mode == "color":
        return img if img.ndim == 3 else cv2.cvtColor(binarize.to_gray(img), cv2.COLOR_GRAY2BGR)
    gray = binarize.to_gray(img)
    if mode == "gray":
        return gray
    # 二值化阶段正常已产出 {0,255}；预设里若关掉它，这里兜底补一次 Otsu
    if np.unique(gray).size > 2:
        gray = binarize.otsu(gray)
    return gray


def _process_one(src: Path, dst: Path, args: argparse.Namespace) -> list[Path]:
    """处理单个输入文件（PDF 逐页），返回实际写出的文件列表。"""
    pages = load_any(src)
    runner = Runner(_build_stages(args))
    finals: list[Image] = []
    for idx, img in enumerate(pages):
        log.info("%s 第 %d 页载入 %dx%d", src.name, idx + 1, img.shape[1], img.shape[0])
        result = runner.run(img)
        for warning in result.warnings:
            log.warning("%s：%s", src.name, warning)
        log.info(
            "%s 第 %d 页阶段链：%s（%.0f ms）",
            src.name, idx + 1, " → ".join(result.order) or "无", result.total_ms,
        )
        finals.append(_finalize(result.final, args.mode))

    if dst.suffix.lower() == ".pdf":
        return [export_pdf(finals, dst)]

    if len(finals) == 1:
        return [save_image(finals[0], dst)]
    return [
        save_image(im, dst.with_name(f"{dst.stem}_p{i + 1:02d}{dst.suffix}"))
        for i, im in enumerate(finals)
    ]


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    if not args.cli:
        from docscan.app import run_gui

        return run_gui()

    if not args.inputs:
        print("错误：CLI 模式需要至少一个输入文件", file=sys.stderr)
        return 2
    if not args.output:
        print("错误：CLI 模式需要 -o 指定输出路径", file=sys.stderr)
        return 2

    out_path = Path(args.output)
    inputs = [Path(p) for p in args.inputs]
    if len(inputs) > 1 and out_path.suffix:
        print("错误：多输入时 -o 必须是目录", file=sys.stderr)
        return 2

    for src in inputs:
        dst = out_path if len(inputs) == 1 else out_path / f"{src.stem}.png"
        try:
            written = _process_one(src, dst, args)
        except (FileNotFoundError, ValueError) as exc:
            print(f"错误：{exc}", file=sys.stderr)
            return 1
        for path in written:
            log.info("已写出 %s", path)
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
