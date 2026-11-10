"""实验⑦ 端到端评测：七类典型场景跑完整处理链。

对应课程要求⑦：正常 / 透视畸变 / 阴影 / 摩尔纹 / 运动模糊 / 散焦 / 偏色
的量化汇总。

用法::

    python -m docscan.experiments.exp_e2e --data data/raw --out data/results

data/raw 里有真实照片则逐张评测；没有则用合成文本页（保证实验可复现）。

产物：e2e_summary.csv + e2e_summary.png（双面板柱状图）

指标口径（这一版修正了三处会「自欺」的地方）：

1. **PSNR / SSIM 在灰度档上算**：二值输出与连续灰度 GT 的 PSNR 理论上限只有
   ~20.3 dB，五个场景全挤在同一个值上，等于没有区分度。因此额外跑一遍
   「同参数但关闭二值化」的链，用它和 GT 比 PSNR/SSIM；
2. **每个场景都有真实退化**：``defocus`` 必须在 STAGE_POLICY 里（否则会被
   降级成 normal 而完全不退化）；``cast`` 必须用彩色输入（灰度输入下
   ``add_color_cast`` 会原样返回，白平衡阶段等于空转）；
3. **PSF 来源必须与注释一致**：motion 场景用 ``psf_mode="manual"`` 给已知
   PSF，不能用 ``use_estimation`` 这种不存在的键（``stage_restore`` 只认
   ``psf_mode``，写错键会静默退回倒谱估计）。
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
from docscan.pipeline.stages import Stage, build_default_stages

log = logging.getLogger("docscan.experiments")

# 各场景建议启用的阶段（对应真实使用策略）
STAGE_POLICY: dict[str, dict[str, bool]] = {
    "normal": {"warp": True, "illumination": True, "binarize": True},
    "perspective": {"warp": True, "illumination": True, "binarize": True},
    "shadow": {"warp": True, "illumination": True, "binarize": True},
    "moire": {"warp": False, "freq": True, "illumination": True, "binarize": True},
    "motion": {"warp": False, "denoise": True, "restore": True, "binarize": True},
    "defocus": {"warp": False, "denoise": True, "restore": True, "binarize": True},
    "cast": {"warp": False, "white_balance": True, "illumination": True, "binarize": True},
}


def _apply_policy(stages: list[Stage], policy: dict[str, bool]) -> None:
    for stage in stages:
        stage.enabled = bool(policy.get(stage.name, False))


def _resolve_scene(name: str, *, synthetic: bool) -> str:
    """把输入文件名映射到场景名。

    合成路径的场景名来自固定列表，出现未知值说明代码与数据脱节；此时
    **必须抛错**——静默按 ``normal`` 跑会让「退化图根本没退化」的行混进
    汇总表（历史缺陷 D2：defocus 与 shadow 的 PSNR 同为 20.32）。
    真实照片路径没有场景标注，允许回退并记 warning。
    """
    scene = name.split("_")[0].lower()
    if scene in STAGE_POLICY:
        return scene
    if synthetic:
        raise ValueError(f"未知场景 {scene!r}，可选 {sorted(STAGE_POLICY)}")
    log.warning("%s 场景名 %r 未识别，按 normal 处理", name, scene)
    return "normal"


def _make_stages(scene: str) -> list[Stage]:
    """按场景生成一套全新的阶段对象（含该场景的已知退化参数）。"""
    stages = build_default_stages()
    for stage in stages:
        if stage.name == "denoise" and scene in ("motion", "defocus"):
            # 维纳会放大噪声：先去噪（双边保边）再复原，SSIM 从 0.76 回到 0.90+
            stage.params["method"] = "bilateral"
            stage.params["sigma_color"] = 0.05
        if stage.name != "restore":
            continue
        if scene == "motion":
            # 已知 PSF：等价「已知退化模型」的定量实验（倒谱在文字密集页会锁定行距）
            stage.params["psf_type"] = "motion"
            stage.params["psf_mode"] = "manual"
            stage.params["length"] = 11.0
            stage.params["angle_deg"] = 0.0
        elif scene == "defocus":
            stage.params["psf_type"] = "defocus"
            stage.params["defocus_radius"] = 5.0
    _apply_policy(stages, STAGE_POLICY[scene])
    return stages


def _to_gray01(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = np.clip(arr.astype(np.float32), 0.0, 1.0)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    return arr


def _fit(ref: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """把参考图缩放到输出尺寸（几何校正会改变画布大小）。"""
    if ref.shape == shape:
        return ref
    return cv2.resize(ref, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)


def _run(
    scene: str, degraded: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], float]:
    """跑两遍：完整链（出二值图给 CER）+ 关闭二值化的灰度链（给 PSNR/SSIM）。"""
    t0 = time.perf_counter()
    full = Runner(_make_stages(scene)).run(degraded)
    elapsed = (time.perf_counter() - t0) * 1000.0

    gray_stages = _make_stages(scene)
    for stage in gray_stages:
        if stage.name == "binarize":
            stage.enabled = False
    gray_run = Runner(gray_stages).run(degraded)

    binary = np.rint(np.clip(_to_gray01(full.final), 0, 1) * 255).astype(np.uint8)
    gray = _to_gray01(gray_run.final)
    warnings = full.warnings + [w for w in gray_run.warnings if w not in full.warnings]
    return binary, gray, full.order, warnings, elapsed


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
                img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if img is not None and min(img.shape[:2]) >= 400:
                    inputs.append((p.stem, img, None))
    synthetic = not inputs
    page = None
    if synthetic:
        page, gt_text = utils.render_text_page(width=900, height=1200, font_px=36)
        gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
        for name in ("normal", *SCENARIOS):
            inputs.append((name, gray, gt_text))
        log.info("data/raw 无可用照片，改用合成文本页评测")

    rows: list[dict] = []
    for name, img, text in inputs:
        scene = _resolve_scene(name, synthetic=synthetic)

        if synthetic:
            if scene == "normal":
                degraded, clean = img.astype(np.float32) / 255.0, img
            elif scene == "cast":
                # 偏色必须作用在彩色图上，否则白平衡阶段无事可做
                degraded, clean = synthesize(page, "cast")
            else:
                degraded, clean = synthesize(img, scene)
        else:
            degraded, clean = img, img  # 真实照片无 GT，PSNR/SSIM 列留空

        binary, gray_out, order, warnings, elapsed = _run(scene, degraded)
        if warnings:
            log.warning("%s 阶段告警：%s", name, warnings)

        clean_gray = _to_gray01(clean)
        degraded_gray = _to_gray01(degraded)

        # 二值档 → CER；灰度档 → PSNR / SSIM（与同亮度 GT 配对）
        if use_cer and text:
            cer_value = metrics.cer(text, recognizer.text(binary) or "")
        else:
            cer_value = float("nan")

        base_u8 = np.rint(np.clip(degraded_gray, 0, 1) * 255).astype(np.uint8)
        base_bin = cv2.adaptiveThreshold(
            base_u8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
        )
        baseline_cer = (
            metrics.cer(text, recognizer.text(base_bin) or "")
            if use_cer and text
            else float("nan")
        )

        if synthetic:
            psnr_v = min(metrics.psnr(_fit(clean_gray, gray_out.shape), gray_out), 99.0)
            ssim_v = metrics.ssim(_fit(clean_gray, gray_out.shape), gray_out)
            base_psnr = min(
                metrics.psnr(_fit(clean_gray, degraded_gray.shape), degraded_gray), 99.0
            )
            base_ssim = metrics.ssim(_fit(clean_gray, degraded_gray.shape), degraded_gray)
        else:
            psnr_v = ssim_v = base_psnr = base_ssim = float("nan")

        rows.append(
            {
                "scene": name,
                "cer": cer_value,
                "baseline_cer": baseline_cer,
                "psnr": psnr_v,
                "baseline_psnr": base_psnr,
                "ssim": ssim_v,
                "baseline_ssim": base_ssim,
                "time_ms": elapsed,
                "stages": "+".join(order),
            }
        )
        log.info(
            "%s: CER=%.4f(基线 %.4f) PSNR=%.2f(基线 %.2f) SSIM=%.4f(基线 %.4f) [%s] %.0f ms",
            name, cer_value, baseline_cer, psnr_v, base_psnr, ssim_v, base_ssim,
            "+".join(order), elapsed,
        )

    csv_path = out_dir / "e2e_summary.csv"
    header = ["scene", "cer", "baseline_cer", "psnr_db", "baseline_psnr_db",
              "ssim", "baseline_ssim", "time_ms", "stages"]
    utils.write_csv(csv_path, header, [
        [
            r["scene"], f"{r['cer']:.4f}", f"{r['baseline_cer']:.4f}",
            f"{r['psnr']:.2f}", f"{r['baseline_psnr']:.2f}",
            f"{r['ssim']:.4f}", f"{r['baseline_ssim']:.4f}",
            f"{r['time_ms']:.0f}", r["stages"],
        ]
        for r in rows
    ])
    _bar(rows, out_dir / "e2e_summary.png")
    log.info("已写出 %s", csv_path)
    return csv_path


def _bar(rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    setup_matplotlib_cjk()
    import matplotlib.pyplot as plt

    names = [r["scene"] for r in rows]
    x = np.arange(len(names))
    width = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), dpi=140)

    ax = axes[0]
    ax.bar(x - width / 2, [r["baseline_cer"] for r in rows], width,
           label="基线（仅二值化）", color="#c2c8d0")
    ax.bar(x + width / 2, [r["cer"] for r in rows], width,
           label="完整处理链", color="#2563eb")
    for xi, r in zip(x + width / 2, rows, strict=True):
        ax.text(xi, r["cer"], f"{r['cer']:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, names, rotation=15)
    ax.set_ylabel("CER（越小越好）")
    ax.set_title("OCR 字符错误率")
    ax.legend()

    # normal 场景本身没有退化，基线 PSNR 为 ∞，放进柱状图会把其余柱子压扁
    psnr_rows = [r for r in rows if r["baseline_psnr"] < 90]
    xs = np.arange(len(psnr_rows))
    ax = axes[1]
    ax.bar(xs - width / 2, [r["baseline_psnr"] for r in psnr_rows], width,
           label="基线（不处理）", color="#c2c8d0")
    ax.bar(xs + width / 2, [r["psnr"] for r in psnr_rows], width,
           label="完整处理链", color="#16a34a")
    for xi, r in zip(xs + width / 2, psnr_rows, strict=True):
        ax.text(xi, r["psnr"], f"{r['psnr']:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs, [r["scene"] for r in psnr_rows], rotation=15)
    ax.set_ylabel("PSNR / dB（越高越好，灰度档）")
    ax.set_title("灰度档重建质量（不含无退化的 normal）")
    ax.legend()

    fig.suptitle("端到端场景评测（左：全场景 CER / 右：退化场景 PSNR）", fontsize=13)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="端到端七场景评测")
    parser.add_argument("--data", default="data/raw", help="真实照片目录（可选）")
    parser.add_argument("--out", default="data/results", help="输出目录")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    data = Path(args.data)
    run(out_dir=Path(args.out), data_dir=data if data.exists() else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
