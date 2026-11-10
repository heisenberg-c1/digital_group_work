"""实验③ 空间域增强对比：伽马 / 对比度拉伸 / 直方图均衡 / CLAHE。

对应课程要求③：空间域直方图均衡、伽马校正、对比度拉伸的效果对比。

用法::

    python -m docscan.experiments.exp_enhance --out data/results

产物：enhance_compare.csv + enhance_compare.png（2×3 对比图，标题带指标）

为什么要单独做这个实验：这四个算子的差别不是「谁更好」，而是「谁在什么场景下
更好」——伽马只改映射曲线不动直方图形状，拉伸是线性拉伸，直方图均衡追求全局
均匀但会放大平坦区噪声，CLAHE 用分块 + 削波把「局部对比」提上来而不炸噪声。
只看一张 before/after 图看不出这些差别，必须给同一张低对比 + 侧光的页面上
跑同一套指标。

指标口径：
* CER —— 二值化后送 OCR，衡量「可读性」这个最终目标；
* 全局对比度 std、局部对比度（16×16 分块 std 的均值）、清晰度（拉普拉斯方差）
  —— 三个不依赖 GT 的客观量，避免「增强后亮度变了导致 PSNR 失真」的坑。
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

from docscan.core import metrics
from docscan.experiments import utils
from docscan.experiments.synth import add_shadow
from docscan.experiments.utils import setup_matplotlib_cjk
from docscan.pipeline.stages import stage_binarize, stage_enhance

log = logging.getLogger("docscan.experiments")

METHODS: tuple[tuple[str, str, dict], ...] = (
    ("none", "不增强", {"method": "none", "gamma": 1.0}),
    ("gamma_0.6", "伽马 γ=0.6", {"method": "none", "gamma": 0.6}),
    ("gamma_1.8", "伽马 γ=1.8", {"method": "none", "gamma": 1.8}),
    ("stretch", "对比度拉伸", {"method": "stretch", "gamma": 1.0}),
    ("histeq", "直方图均衡", {"method": "histeq", "gamma": 1.0}),
    ("clahe", "CLAHE", {"method": "clahe", "clip_limit": 2.0, "grid_size": 8, "gamma": 1.0}),
)


def _local_contrast(gray: np.ndarray, block: int = 16) -> float:
    """局部对比度：分块标准差的中位数（比全局 std 更贴近「看得清不清」）。"""
    h, w = gray.shape
    bh, bw = max(1, h // block), max(1, w // block)
    vals = []
    for y in range(0, h - bh + 1, bh):
        for x in range(0, w - bw + 1, bw):
            vals.append(float(gray[y:y + bh, x:x + bw].std()))
    return float(np.median(vals)) if vals else 0.0


def _degrade(gray_u8: np.ndarray) -> np.ndarray:
    """造一张「难处理」的页：动态范围压缩 + 侧光梯度（模拟欠曝手机照）。"""
    low = add_shadow(gray_u8.astype(np.float32) / 255.0, lo=0.45, angle_deg=20.0)
    return np.clip(low * 0.45 + 0.30, 0.0, 1.0).astype(np.float32)


def run(*, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    page, gt_text = utils.render_text_page(width=1240, height=1754, font_px=42)
    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    degraded = _degrade(gray)
    recognizer = utils.TextRecognizer()
    use_cer = recognizer.uses_ocr

    rows: list[dict] = []
    images: dict[str, np.ndarray] = {"退化输入": degraded}
    for key, label, params in METHODS:
        enhanced, note, _ = stage_enhance(degraded, params)
        arr = np.asarray(enhanced)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32) / 255.0
        arr = np.clip(arr, 0.0, 1.0)

        binary, _, _ = stage_binarize(arr, {"method": "adaptive", "block_size": 31, "c": 10.0})
        cer_value = (
            min(metrics.cer(gt_text, recognizer.text(binary) or ""), 1.0)
            if use_cer
            else float("nan")
        )
        rows.append(
            {
                "method": key,
                "label": label,
                "detail": note,
                "cer": cer_value,
                "std": float(arr.std()),
                "local_contrast": _local_contrast(arr),
                "sharpness": metrics.sharpness(arr),
                "psnr": min(metrics.psnr(gray.astype(np.float32) / 255.0, arr), 99.0),
            }
        )
        images[label] = arr
        log.info(
            "%-10s CER=%.4f std=%.4f 局部=%.4f 清晰度=%.5f PSNR=%.2f",
            label, cer_value, arr.std(), _local_contrast(arr),
            metrics.sharpness(arr), rows[-1]["psnr"],
        )

    csv_path = out_dir / "enhance_compare.csv"
    header = ["method", "label", "cer", "std", "local_contrast", "sharpness", "psnr_db"]
    # label 是自由文本（如「伽马 γ=0.6」），必须走转义写入
    utils.write_csv(csv_path, header, [
        [
            r["method"], r["label"], f"{r['cer']:.4f}", f"{r['std']:.4f}",
            f"{r['local_contrast']:.4f}", f"{r['sharpness']:.5f}", f"{r['psnr']:.2f}",
        ]
        for r in rows
    ])
    _grid(images, rows, out_dir / "enhance_compare.png")
    log.info("已写出 %s", csv_path)
    return csv_path


def _grid(images: dict[str, np.ndarray], rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    setup_matplotlib_cjk()
    import matplotlib.pyplot as plt

    by_label = {r["label"]: r for r in rows}
    panels = list(images.items())
    ncols = 4
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 4.4 * nrows), dpi=130)
    for ax, (title, img) in zip(np.asarray(axes).flat, panels, strict=False):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        r = by_label.get(title)
        if r is None:
            ax.set_title(f"{title}\nstd={img.std():.3f}", fontsize=10)
        else:
            ax.set_title(
                f"{title}\nCER={r['cer']:.3f}  局部={r['local_contrast']:.3f}",
                fontsize=10,
            )
        ax.axis("off")
    for ax in np.asarray(axes).flat[len(panels):]:
        ax.axis("off")
    fig.suptitle("空间域增强方法对比（同一张低对比 + 侧光页面）", fontsize=13)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="空间域增强方法对比")
    parser.add_argument("--out", default="data/results", help="输出目录")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(out_dir=Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
