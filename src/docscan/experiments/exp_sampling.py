"""实验① 采样与量化：DPI × 位深 → CER 热力图。

对应课程要求①：改变采样率与量化位数对结果的影响。

用法::

    python -m docscan.experiments.exp_sampling --out data/results

产物：sampling.csv（dpi,bits,metric 矩阵）+ sampling_heatmap.png
DPI 用「面积插值降采样 + 立方插值还原」模拟；位深用均匀量化模拟。
OCR 可用时指标为 CER（越小越好），否则为 SSIM 代理（越大越好）。
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

from docscan.core import metrics
from docscan.core.metrics import cer
from docscan.experiments import utils
from docscan.experiments.utils import setup_matplotlib_cjk
from docscan.io.exporter import save_image

log = logging.getLogger("docscan.experiments")

DPIS = (75, 100, 150, 200, 300)
BITS = (1, 2, 4, 8)


def run(*, out_dir: Path, base_dpi: int = 300) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    page, gt_text = utils.render_text_page()
    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    recognizer = utils.TextRecognizer()
    use_cer = recognizer.uses_ocr

    rows: list[str] = []
    table = np.zeros((len(DPIS), len(BITS)), dtype=np.float64)
    for i, dpi in enumerate(DPIS):
        for j, bits in enumerate(BITS):
            processed = utils.resample_dpi(gray, dpi, base_dpi=base_dpi)
            processed = utils.quantize(processed, bits)
            if bits == 1:
                # 1-bit 场景：模拟扫描仪的直接二值化输出
                processed = (processed > 128).astype(np.uint8) * 255

            if use_cer:
                hyp = recognizer.text(processed) or ""
                value = min(cer(gt_text, hyp), 1.0)
                label = "cer"
            else:
                value = 1.0 - metrics.ssim(gray, processed)  # SSIM 损失
                label = "ssim_loss"
            table[i, j] = value
            log.info("dpi=%d bits=%d %s=%.4f", dpi, bits, label, value)

    header = "dpi," + ",".join(f"{b}bit" for b in BITS)
    for i, dpi in enumerate(DPIS):
        rows.append(f"{dpi}," + ",".join(f"{table[i, j]:.4f}" for j in range(len(BITS))))
    csv_path = out_dir / "sampling.csv"
    csv_path.write_text(
        f"# metric={label} (越小越好)\n{header}\n" + "\n".join(rows), encoding="utf-8"
    )

    _heatmap(table, label, out_dir / "sampling_heatmap.png")
    save_image(page, out_dir / "sampling_source_page.png")
    log.info("已写出 %s", csv_path)
    return csv_path


def _heatmap(table: np.ndarray, label: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    setup_matplotlib_cjk()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.6), dpi=140)
    im = ax.imshow(table, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(BITS)), [f"{b} bit" for b in BITS])
    ax.set_yticks(range(len(DPIS)), [f"{d} DPI" for d in DPIS])
    for i in range(len(DPIS)):
        for j in range(len(BITS)):
            ax.text(j, i, f"{table[i, j]:.3f}", ha="center", va="center", fontsize=9)
    ax.set_title(f"采样率 × 量化位深 → {label}（越小越好）")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DPI × 位深 CER 热力图实验")
    parser.add_argument("--out", default="data/results", help="输出目录")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(out_dir=Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
