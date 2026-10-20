"""实验⑤ 维纳滤波 K 扫描：PSNR(K) 曲线。

对应课程要求⑤：说明正则化参数 K 的选择依据。

用法::

    python -m docscan.experiments.exp_wiener --out data/results

产物：wiener_K_sweep.csv + wiener_K_sweep.png（PSNR 曲线 + 最优点标注）
在两组噪声水平（低噪/高噪）下扫描 K，验证「K 应近似噪信比」的理论。
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

from docscan.core import metrics, restore
from docscan.experiments import utils
from docscan.experiments.utils import setup_matplotlib_cjk

log = logging.getLogger("docscan.experiments")

K_GRID = np.geomspace(1e-5, 1.0, 21)


def _sweep(clean: np.ndarray, blurred: np.ndarray, psf: np.ndarray) -> list[float]:
    scores = []
    for k in K_GRID:
        out = restore.wiener(blurred, psf, K=float(k))
        scores.append(metrics.psnr(clean, out))
    return scores


def run(*, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    page, _ = utils.render_text_page(width=800, height=1000, font_px=34)
    clean = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    length, angle = 9, 25.0
    psf = restore.motion_psf(max(31, length * 2 | 1), length, angle)

    curves: dict[str, list[float]] = {}
    best: dict[str, float] = {}
    for label, sigma in (("低噪声 σ=0.003", 0.003), ("高噪声 σ=0.015", 0.015)):
        blurred = restore.convolve_blur(clean, psf, noise_sigma=sigma)
        scores = _sweep(clean, blurred, psf)
        curves[label] = scores
        i = int(np.argmax(scores))
        best[label] = float(K_GRID[i])
        log.info("%s: 最优 K=%.4g, PSNR=%.2f dB", label, K_GRID[i], scores[i])

    csv_lines = ["K," + ",".join(curves.keys())]
    for j, k in enumerate(K_GRID):
        csv_lines.append(f"{k:.6g}," + ",".join(f"{curves[name][j]:.3f}" for name in curves))
    csv_path = out_dir / "wiener_K_sweep.csv"
    csv_path.write_text("\n".join(csv_lines), encoding="utf-8")

    _plot(curves, best, out_dir / "wiener_K_sweep.png")
    log.info("已写出 %s", csv_path)
    return csv_path


def _plot(curves: dict[str, list[float]], best: dict[str, float], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    setup_matplotlib_cjk()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=140)
    for label, scores in curves.items():
        ax.plot(K_GRID, scores, marker="o", ms=3, label=label)
        i = int(np.argmax(scores))
        ax.annotate(
            f"最优 K={K_GRID[i]:.3g}\nPSNR={scores[i]:.1f}dB",
            (K_GRID[i], scores[i]), textcoords="offset points", xytext=(8, 8), fontsize=8,
            arrowprops=dict(arrowstyle="->", lw=0.8),
        )
    ax.set_xscale("log")
    ax.set_xlabel("维纳参数 K（噪信比）")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("维纳滤波 K 扫描：PSNR 先升后降，峰值随噪声增大右移")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="维纳滤波 K 扫描实验")
    parser.add_argument("--out", default="data/results", help="输出目录")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(out_dir=Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
