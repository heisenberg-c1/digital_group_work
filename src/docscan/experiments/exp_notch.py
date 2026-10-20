"""实验④ 去网纹对比：理想陷波 vs Butterworth 陷波 vs 中值滤波。

对应课程要求④：对比不同频域滤波器的效果，核心论据是振铃。

用法::

    python -m docscan.experiments.exp_notch --out data/results

产物：notch_compare.csv + notch_compare.png（四联对比图）
每个滤波器报告：峰能量下降(dB)、SSIM、振铃指标、耗时(ms)。
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from docscan.core import freq, metrics
from docscan.experiments import utils
from docscan.experiments.synth import add_moire
from docscan.experiments.utils import setup_matplotlib_cjk

log = logging.getLogger("docscan.experiments")


def _median_filter(gray: np.ndarray, ksize: int = 5) -> np.ndarray:
    return cv2.medianBlur(gray, ksize)


def run(*, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    page, _ = utils.render_text_page(width=1024, height=1024, font_px=36)
    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)

    # 合成摩尔纹（先压内容避免削顶谐波；注意归一到 [0,1] 再交给合成器）
    clean = (gray.astype(np.float32) / 255.0 * 0.75).astype(np.float32)
    striped = add_moire(clean, wavelength=28, angle_deg=25.0, amplitude=0.2)

    # 自动找峰（条纹频率已知且背景无强周期结构时可靠）
    spec = freq.magnitude_spectrum(striped)
    peaks = freq.find_periodic_peaks(spec, top_n=2)
    log.info("检出周期峰: %s", peaks)

    results: list[dict] = []
    outputs: dict[str, np.ndarray] = {}

    def record(name: str, out: np.ndarray, elapsed_ms: float) -> None:
        results.append(
            {
                "method": name,
                "peak_drop_db": -freq.energy_drop_db(striped, out, peaks[0]),
                "ssim": metrics.ssim(clean, out),
                "ringing": metrics.ringing_index(out, clean),
                "time_ms": elapsed_ms,
            }
        )
        outputs[name] = out

    t0 = time.perf_counter()
    record("无处理", striped, (time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    out = freq.notch_filter(striped, peaks, kind="ideal", radius=4)
    record("理想陷波", out, (time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    out = freq.notch_filter(striped, peaks, kind="butterworth", radius=4, order=4)
    record("Butterworth陷波(n=4)", out, (time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    record("中值滤波(5x5)", _median_filter(striped), (time.perf_counter() - t0) * 1000)

    csv_path = out_dir / "notch_compare.csv"
    header = "method,peak_drop_db,ssim,ringing,time_ms"
    lines = [header] + [
        f"{r['method']},{r['peak_drop_db']:.2f},{r['ssim']:.4f},{r['ringing']:.4f},{r['time_ms']:.1f}"
        for r in results
    ]
    csv_path.write_text("\n".join(lines), encoding="utf-8")
    _grid(striped, outputs, clean, path=out_dir / "notch_compare.png")
    log.info("已写出 %s", csv_path)
    return csv_path


def _grid(
    striped: np.ndarray, outputs: dict[str, np.ndarray], clean: np.ndarray, path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    setup_matplotlib_cjk()
    import matplotlib.pyplot as plt

    panels = [("摩尔纹输入", striped), *outputs.items(), ("干净参考", clean)]
    ncols = 3
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.2 * nrows), dpi=130)
    for ax, (title, img) in zip(np.asarray(axes).flat, panels, strict=False):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    for ax in np.asarray(axes).flat[len(panels):]:
        ax.axis("off")
    fig.suptitle("去网纹方法对比", fontsize=13)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="陷波器对比实验")
    parser.add_argument("--out", default="data/results", help="输出目录")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(out_dir=Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
