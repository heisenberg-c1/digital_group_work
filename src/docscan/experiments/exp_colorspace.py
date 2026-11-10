"""实验② 色彩空间对比：只改亮度（Y / V）与逐通道处理的色相漂移。

对应课程要求②：RGB 与 HSV/YCbCr 转换，分离亮度与色度。

为什么要做这个实验：``apply_on_luma`` / ``apply_on_value`` 两个高阶函数都宣称
「只动亮度、不动颜色」，但「亮度」在两套空间里定义不同——YCbCr 的 Y 是加权和
``0.299R+0.587G+0.114B``，HSV 的 V 是 ``max(R,G,B)``。同一张彩色文档走两条路，
对比度提升不同、高光裁剪程度不同、色相残留也不同。只给一张 before/after 图
看不出这些差别，必须用色相漂移这个客观量量化。

对照组是**逐通道处理**（BGR 三个通道各走同一条映射曲线，不做任何分离）：
它是最容易写出来的做法，代价是色相整体偏移——彩色印章会变色、荧光笔会偏色。

指标口径：
* 色相漂移（度）—— 仅统计饱和度 > 0.15 的像素，避免灰色纸面上的噪声放大；
  色相差取圆周距离 ``min(|ΔH|, 360-|ΔH|)``，否则 359° 与 1° 会被算成 358°；
* 饱和度变化 —— 色相不变但饱和度掉光，视觉上同样是「掉色」；
* 裁剪比例 —— 落在 0 或 255 的通道占比，衡量「亮度映射是否把高光压爆」；
* CER —— 二值化后送 OCR，确认彩色处理没有牺牲可读性。

用法::

    python -m docscan.experiments.exp_colorspace --out data/results

产物：colorspace_compare.csv + colorspace_compare.png
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

from docscan.core import colorspace, metrics
from docscan.experiments import utils
from docscan.experiments.utils import setup_matplotlib_cjk
from docscan.pipeline.stages import stage_binarize

log = logging.getLogger("docscan.experiments")

GAMMA = 0.6          # 提亮曲线，模拟「阴影页需要提对比度」
SAT_FLOOR = 0.15     # 参与色相统计的饱和度下限


def _color_page(width: int = 900, height: int = 1200, font_px: int = 34):
    """文本页 + 彩色元素（红章 / 黄荧光笔 / 蓝批注），用于检验色相保持。"""
    page, text = utils.render_text_page(width=width, height=height, font_px=font_px)

    # 黄色荧光笔：半透明叠加，避免直接盖掉文字
    overlay = page.copy()
    cv2.rectangle(overlay, (70, 300), (width - 70, 336), (0, 235, 255), -1)
    page = cv2.addWeighted(overlay, 0.45, page, 0.55, 0)

    # 红色印章（空心圆 + 内圈）
    cv2.circle(page, (width - 150, 150), 64, (40, 40, 215), 6)
    cv2.circle(page, (width - 150, 150), 44, (40, 40, 215), 3)

    # 蓝色批注下划线
    cv2.line(page, (80, 980), (width - 80, 1010), (215, 90, 30), 5)
    return page, text


def _curve(x: np.ndarray) -> np.ndarray:
    """亮度映射：0..255 上的伽马提亮。三条路径共用同一条曲线才可比。"""
    return 255.0 * np.power(np.clip(np.asarray(x, dtype=np.float64), 0.0, 255.0) / 255.0, GAMMA)


def _per_channel(bgr: np.ndarray) -> np.ndarray:
    """逐通道处理：不做亮度/色度分离（对照组）。"""
    out = np.stack([_curve(bgr[..., c] * 255.0) / 255.0 for c in range(3)], axis=-1)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _hue_stats(ref: np.ndarray, out: np.ndarray) -> tuple[float, float]:
    """返回 (平均色相漂移[度], 平均饱和度变化)，只在有色像素上统计。"""
    h0, s0, _ = colorspace.bgr_to_hsv(ref)
    h1, s1, _ = colorspace.bgr_to_hsv(out)
    mask = (s0 > SAT_FLOOR) & (s1 > SAT_FLOOR)
    if not mask.any():
        return 0.0, 0.0
    diff = np.abs(h1 - h0)
    diff = np.minimum(diff, 360.0 - diff)  # 圆周距离
    return float(diff[mask].mean()), float((s1 - s0)[mask].mean())


def _clip_ratio(bgr: np.ndarray) -> float:
    return float(np.mean((bgr <= 1e-6) | (bgr >= 1.0 - 1e-6)))


def run(*, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    page, text = _color_page()
    ref = page.astype(np.float32) / 255.0

    methods: tuple[tuple[str, str], ...] = (
        ("luma", "仅 Y（YCbCr 加权和）"),
        ("value", "仅 V（HSV 最大值）"),
        ("bgr", "逐通道（不分离）"),
    )

    rows: list[dict] = []
    images: dict[str, np.ndarray] = {"原图": page}
    for key, label in methods:
        if key == "luma":
            out = colorspace.apply_on_luma(ref, _curve)
        elif key == "value":
            out = colorspace.apply_on_value(ref, _curve)
        else:
            out = _per_channel(ref)

        hue, sat = _hue_stats(ref, out)
        u8 = np.rint(np.clip(out, 0, 1) * 255).astype(np.uint8)
        binary, _, _ = stage_binarize(u8, {"method": "adaptive", "block_size": 31, "c": 10.0})
        cer = metrics.cer(text, utils.TextRecognizer().text(binary) or "")
        psnr = min(metrics.psnr(ref, out), 99.0)

        rows.append({
            "method": key,
            "label": label,
            "hue_drift_deg": hue,
            "sat_delta": sat,
            "clip_ratio": _clip_ratio(out),
            "psnr_db": psnr,
            "cer": cer,
        })
        images[label] = u8
        log.info("%-22s 色相漂移=%.3f° ΔS=%+.4f 裁剪=%.3f PSNR=%.2f CER=%.4f",
                 label, hue, sat, rows[-1]["clip_ratio"], psnr, cer)

    csv_path = out_dir / "colorspace_compare.csv"
    header = ["method", "label", "hue_drift_deg", "sat_delta", "clip_ratio", "psnr_db", "cer"]
    utils.write_csv(csv_path, header, [
        [r["method"], r["label"], f"{r['hue_drift_deg']:.4f}", f"{r['sat_delta']:+.4f}",
         f"{r['clip_ratio']:.4f}", f"{r['psnr_db']:.2f}", f"{r['cer']:.4f}"]
        for r in rows
    ])
    _grid(images, rows, out_dir / "colorspace_compare.png")
    log.info("已写出 %s", csv_path)
    return csv_path


def _grid(images: dict[str, np.ndarray], rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    setup_matplotlib_cjk()
    import matplotlib.pyplot as plt

    panels = list(images.items())
    fig, axes = plt.subplots(1, len(panels), figsize=(3.1 * len(panels), 5.2), dpi=140)
    by_label = {r["label"]: r for r in rows}
    for ax, (label, bgr) in zip(axes, panels, strict=True):
        ax.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if label in by_label:
            ax.set_title(f"{label}\nΔH={by_label[label]['hue_drift_deg']:.2f}°", fontsize=9)
        else:
            ax.set_title(label, fontsize=9)
        ax.axis("off")

    fig.suptitle("色彩空间对比：只改亮度后色相是否漂移（同一条伽马曲线 γ=0.6）", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))  # 给总标题留出空间，避免压住面板标题
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="色彩空间与色相保持实验")
    parser.add_argument("--out", default="data/results", help="输出目录")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(out_dir=Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
