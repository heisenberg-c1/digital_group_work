"""实验⑦ 端到端评测：五类典型场景跑完整处理链。

对应课程要求⑦：正常 / 阴影 / 摩尔纹 / 模糊 / 偏色五场景的量化汇总。

用法::

    python -m docscan.experiments.exp_e2e --data data/raw --out data/results

data/raw 里有真实照片则逐张评测；没有则用合成文本页（保证实验可复现）。
产物：e2e_summary.csv + e2e_summary.png（分组柱状图）
指标：OCR 可用时报 CER（对 GT 文本），否则报 SSIM（对干净参考）。
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from docscan.core import metrics
from docscan.experiments import utils
from docscan.experiments.synth import SCENARIOS, synthesize
from docscan.experiments.utils import setup_matplotlib_cjk
from docscan.pipeline.runner import Runner
from docscan.pipeline.stages import build_default_stages

log = logging.getLogger("docscan.experiments")

# 各场景建议启用的阶段（对应真实使用策略）
STAGE_POLICY = {
    "normal": {"warp": True, "illumination": True, "binarize": True},
    "shadow": {"warp": True, "illumination": True, "binarize": True},
    "moire": {"warp": False, "freq": True, "illumination": True, "binarize": True},
    "motion": {"warp": False, "restore": True, "sharpen": True, "binarize": True},
    "cast": {"warp": False, "white_balance": True, "illumination": True, "binarize": True},
}


def _apply_policy(stages, policy: dict[str, bool]) -> None:
    for stage in stages:
        stage.enabled = bool(policy.get(stage.name, False))


def run(*, out_dir: Path, data_dir: Path | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    recognizer = utils.TextRecognizer()
    use_cer = recognizer.uses_ocr

    # 输入图像：真实照片优先，否则合成页
    inputs: list[tuple[str, np.ndarray, str | None]] = []
    if data_dir and data_dir.exists():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif"}
        for p in sorted(data_dir.iterdir()):
            if p.suffix.lower() in exts:
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if img is not None and min(img.shape[:2]) >= 400:
                    inputs.append((p.stem, img, None))
    synthetic = not inputs
    if synthetic:
        page, gt_text = utils.render_text_page(width=900, height=1200, font_px=36)
        gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
        for name in SCENARIOS:
            inputs.append((name, gray, gt_text))
        log.info("data/raw 无可用照片，改用合成文本页评测")

    rows: list[dict] = []
    for name, gray, gt_text in inputs:
        # 场景名从文件名猜测（如 moire_01.jpg），合成路径直接用场景名
        scene = name.split("_")[0].lower()
        if scene not in STAGE_POLICY:
            scene = "normal"

        if synthetic:
            if scene == "normal":
                degraded, clean = gray.astype(np.float32) / 255.0, gray
            else:
                degraded, clean = synthesize(gray, scene)
        else:
            degraded, clean = gray, gray  # 真实照片无 GT，SSIM 列为 1 占位

        stages = build_default_stages()
        if scene == "motion":
            # 倒谱自动估计在文字密集页会锁定文字行周期（见实验记录 M4），
            # 此处按合成参数给手动 PSF，等价「已知 PSF」场景
            for s in stages:
                if s.name == "restore":
                    s.params["use_estimation"] = 0.0
                    s.params["length"] = 11.0
                    s.params["angle_deg"] = 0.0
        _apply_policy(stages, STAGE_POLICY[scene])
        runner = Runner(stages)

        t0 = time.perf_counter()
        result = runner.run(degraded)  # synthesize 输出已是 float32[0,1]
        elapsed = (time.perf_counter() - t0) * 1000.0

        out_img = result.final
        if out_img.ndim == 3:
            out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2GRAY)
        out_u8 = np.rint(np.clip(out_img, 0, 1) * 255).astype(np.uint8)

        if use_cer and gt_text:
            hyp = recognizer.text(out_u8) or ""
            cer_value = metrics.cer(gt_text, hyp)
            value = cer_value
            metric_name = "cer"
        else:
            value = 1.0 - metrics.ssim(
                cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR) if clean.ndim == 2 else clean, out_u8
            )
            metric_name = "ssim_loss"

        # 退化基线（不处理直接二值化）做对照
        base_u8 = (
            degraded if degraded.dtype == np.uint8
            else np.rint(degraded * 255).astype(np.uint8)
        )
        base_u8 = cv2.adaptiveThreshold(base_u8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 31, 10)
        base_value = (
            metrics.cer(gt_text, recognizer.text(base_u8) or "")
            if use_cer and gt_text
            else 1.0 - metrics.ssim(clean, base_u8)
        )

        rows.append(
            {
                "scene": name,
                "metric": value,
                "baseline": base_value,
                "psnr": min(metrics.psnr(clean, out_u8), 99.0),
                "time_ms": elapsed,
            }
        )
        log.info("%s: %s=%.4f (基线 %.4f), %.0f ms", name, metric_name, value, base_value, elapsed)

    csv_path = out_dir / "e2e_summary.csv"
    lines = [f"scene,{metric_name},baseline_{metric_name},psnr_db,time_ms"] + [
        f"{r['scene']},{r['metric']:.4f},{r['baseline']:.4f},"
        f"{r['psnr']:.2f},{r['time_ms']:.0f}"
        for r in rows
    ]
    csv_path.write_text("\n".join(lines), encoding="utf-8")
    _bar(rows, metric_name, out_dir / "e2e_summary.png")
    log.info("已写出 %s", csv_path)
    return csv_path


def _bar(rows: list[dict], metric_name: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    setup_matplotlib_cjk()
    import matplotlib.pyplot as plt

    names = [r["scene"] for r in rows]
    ours = [r["metric"] for r in rows]
    base = [r["baseline"] for r in rows]
    x = np.arange(len(names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=140)
    ax.bar(x - width / 2, base, width, label="基线（仅二值化）", color="#c2c8d0")
    ax.bar(x + width / 2, ours, width, label="完整处理链", color="#2563eb")
    for xi, v in zip(x + width / 2, ours, strict=True):
        ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, names)
    lower_better = metric_name.startswith(("cer", "ssim_loss"))
    ax.set_ylabel(f"{metric_name}（{'越小越好' if lower_better else '越大越好'}）")
    ax.set_title("端到端五场景评测")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="端到端五场景评测")
    parser.add_argument("--data", default="data/raw", help="真实照片目录（可选）")
    parser.add_argument("--out", default="data/results", help="输出目录")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    data = Path(args.data)
    run(out_dir=Path(args.out), data_dir=data if data.exists() else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
