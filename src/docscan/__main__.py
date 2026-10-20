"""命令行入口。

GUI:  ``python -m docscan``
CLI:  ``python -m docscan --cli samples/demo/a.jpg -o out.png --mode bw``

M0 阶段的 CLI 直接串起「装载 → 几何校正 → 二值化 → 导出」最薄链路；
M1 起改为调用 ``pipeline.Runner``，CLI 只负责参数解析。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from docscan import __version__
from docscan.core import binarize, geometry
from docscan.io.exporter import save_image
from docscan.io.loader import load_any

log = logging.getLogger("docscan")


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
    parser.add_argument("-v", "--verbose", action="store_true", help="输出详细日志")
    parser.add_argument("--version", action="version", version=f"docscan {__version__}")
    return parser


def _process_one(src: Path, dst: Path, args: argparse.Namespace) -> None:
    """处理单个输入文件（PDF 逐页输出）。"""
    pages = load_any(src)
    for idx, img in enumerate(pages):
        log.info("%s 载入 %dx%d", src.name, img.shape[1], img.shape[0])
        work = img

        if not args.no_warp:
            quad = geometry.detect_quad(work, min_confidence=args.min_confidence)
            if quad.is_fallback:
                log.warning(
                    "角点检测失败（置信度 %.2f），退化为整图，请检查拍摄背景或改用手动选点",
                    quad.confidence,
                )
            else:
                log.info("角点检测成功，置信度 %.2f", quad.confidence)
                work = geometry.warp_perspective(work, quad)

        if args.mode == "bw":
            gray = binarize.to_gray(work)
            work = binarize.adaptive(gray, block_size=args.block_size, c=args.c)
            work = binarize.clean_small_components(work, min_area=8)
        elif args.mode == "gray":
            work = binarize.to_gray(work)

        out = dst if len(pages) == 1 else dst.with_name(f"{dst.stem}_p{idx + 1:02d}{dst.suffix}")
        save_image(work, out)
        log.info("已写出 %s (%dx%d)", out, work.shape[1], work.shape[0])


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
            _process_one(src, dst, args)
        except (FileNotFoundError, ValueError) as exc:
            print(f"错误：{exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
