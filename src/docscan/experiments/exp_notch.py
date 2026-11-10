"""实验④ 去网纹对比：理想陷波 vs Butterworth 陷波 vs 中值滤波。

对应课程要求④：对比不同频域滤波器的效果，核心论据是振铃。

用法::

    python -m docscan.experiments.exp_notch --out data/results

产物：notch_compare.csv + notch_compare.png（四联对比图）
每个滤波器报告：真实摩尔纹峰的能量下降(dB)、SSIM、振铃指标、耗时(ms)。

验收口径（写进 CSV 的 status 列，失败必须可见）：

* ``peak_drop_db`` 在**理论峰**上度量（由波长与角度反推），不用自动检出的峰
  ——否则命中伪峰时指标会「好看」，而条纹纹丝不动；
* ``ssim`` 必须 > 0.85 才算去纹成功；
* 检出峰与理论峰一并写入 CSV 注释行，便于人工核对。
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from pathlib import Path

import cv2
import numpy as np

from docscan.core import freq, metrics
from docscan.experiments import utils
from docscan.experiments.synth import add_moire
from docscan.experiments.utils import setup_matplotlib_cjk

log = logging.getLogger("docscan.experiments")

WAVELENGTH = 28
ANGLE_DEG = 25.0
AMPLITUDE = 0.2
SSIM_OK = 0.85


def _median_filter(gray: np.ndarray, ksize: int = 5) -> np.ndarray:
    return cv2.medianBlur(gray, ksize)


def _theoretical_peak(size: int, wavelength: int, angle_deg: float) -> tuple[int, int]:
    """由条纹波长与角度反推理论峰坐标（相对 DC 的偏移）。"""
    radius = size / float(wavelength)
    theta = math.radians(angle_deg)
    return round(radius * math.cos(theta)), round(radius * math.sin(theta))


def run(*, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    page, _ = utils.render_text_page(width=1024, height=1024, font_px=36)
    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)

    # 合成摩尔纹：clean 是「压暗后无条纹」的同一内容，作为严格配对的 GT
    clean = (gray.astype(np.float32) / 255.0 * 0.75).astype(np.float32)
    striped, base = add_moire(
        clean, wavelength=WAVELENGTH, angle_deg=ANGLE_DEG,
        amplitude=AMPLITUDE, return_base=True,
    )

    true_peak = _theoretical_peak(1024, WAVELENGTH, ANGLE_DEG)
    spec = freq.magnitude_spectrum(striped)
    peaks = freq.find_periodic_peaks(spec, top_n=2)
    log.info("理论峰 %s，检出峰 %s", true_peak, peaks)

    detected_ok = any(
        abs(du - true_peak[0]) <= 2 and abs(dv - true_peak[1]) <= 2 for du, dv in peaks
    )
    if not detected_ok:
        log.warning("自动找峰未命中理论峰 %s（检出 %s），陷波结果不可信", true_peak, peaks)

    results: list[dict] = []
    outputs: dict[str, np.ndarray] = {}

    def record(name: str, out: np.ndarray, elapsed_ms: float) -> None:
        s = metrics.ssim(base, out)
        results.append(
            {
                "method": name,
                "peak_drop_db": -freq.energy_drop_db(striped, out, true_peak),
                "ssim": s,
                "ringing": metrics.ringing_index(out, base),
                "time_ms": elapsed_ms,
                "status": "ok" if s >= SSIM_OK else "failed",
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
    header = "method,peak_drop_db,ssim,ringing,time_ms,status"
    comment = (
        f"# wavelength={WAVELENGTH}px angle={ANGLE_DEG}deg amplitude={AMPLITUDE}\n"
        f"# theoretical_peak={true_peak} detected_peaks={peaks} "
        f"detected_ok={int(detected_ok)} ssim_ok_threshold={SSIM_OK}\n"
    )
    lines = [header] + [
        f"{r['method']},{r['peak_drop_db']:.2f},{r['ssim']:.4f},"
        f"{r['ringing']:.4f},{r['time_ms']:.1f},{r['status']}"
        for r in results
    ]
    csv_path.write_text(comment + "\n".join(lines), encoding="utf-8")
    _grid(striped, outputs, base, path=out_dir / "notch_compare.png")
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
