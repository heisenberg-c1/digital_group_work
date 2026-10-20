"""生成报告素材：每个模块的 before/after 对比图，统一尺寸与标注。

用法::

    python tools/make_report_assets.py --out data/results/report_assets

每个模块输出一张 2 联图（左：退化输入，右：处理后），底部标注模块名与
关键指标，直接可贴进报告。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from docscan.core import (  # noqa: E402
    binarize,
    freq,
    illumination,
    metrics,
    restore,
)
from docscan.experiments.synth import add_moire, add_shadow, synthesize
from docscan.experiments.utils import render_text_page  # noqa: E402

FONT_PATH = None
for cand in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
    if Path(cand).exists():
        FONT_PATH = cand
        break

TARGET_W = 640  # 统一半幅宽度


def _label(img: np.ndarray, text: str) -> np.ndarray:
    """底部加中文标注条。"""
    h, w = img.shape[:2]
    bar = np.full((44, w), 248, np.uint8)
    pil = PILImage.fromarray(bar).convert("RGB")
    draw = ImageDraw.Draw(pil)
    font = ImageFont.truetype(FONT_PATH, 20) if FONT_PATH else ImageFont.load_default()
    draw.text((10, 10), text, fill=(30, 30, 30), font=font)
    bar = np.asarray(pil)[:, :, ::-1]
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return np.vstack([img, bar])


def _pair(before: np.ndarray, after: np.ndarray, title: str, note: str) -> np.ndarray:
    def fit(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        scale = TARGET_W / w
        return cv2.resize(img, (TARGET_W, round(h * scale)), interpolation=cv2.INTER_AREA)

    left = _label(fit(before), "处理前")
    right = _label(fit(after), "处理后")

    def pad_h(img: np.ndarray, target_h: int) -> np.ndarray:
        if img.shape[0] >= target_h:
            return img
        pad = np.full((target_h - img.shape[0], img.shape[1], 3), 255, np.uint8)
        return np.vstack([img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), pad])

    h = max(left.shape[0], right.shape[0])
    left, right = pad_h(left, h), pad_h(right, h)
    gap = np.full((h, 12, 3), 255, np.uint8)
    sheet = np.hstack([left, gap, right])
    head = np.full((52, sheet.shape[1], 3), 255, np.uint8)
    pil = PILImage.fromarray(head[:, :, ::-1])
    draw = ImageDraw.Draw(pil)
    font = ImageFont.truetype(FONT_PATH, 24) if FONT_PATH else ImageFont.load_default()
    draw.text((12, 12), f"{title}    {note}", fill=(20, 20, 20), font=font)
    head = np.asarray(pil)[:, :, ::-1]
    return np.vstack([head, sheet])


def save(sheet: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet)
    print("已写出", path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成报告素材")
    parser.add_argument("--out", default="data/results/report_assets")
    args = parser.parse_args(argv)
    out = Path(args.out)

    page, _ = render_text_page(width=1100, height=1500, font_px=38)
    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)

    # ① 几何校正
    quad = np.array([[150, 200], [880, 260], [820, 1330], [200, 1240]], np.float32)
    from docscan.experiments.synth import synthesize  # noqa: F401

    import sys as _sys

    _sys.path.insert(0, str(REPO / "tests"))
    from conftest import paste_on_background  # noqa: E402

    small = cv2.resize(page, (620, 880))
    tilted = paste_on_background(
        small,
        np.array([[180, 140], [540, 190], [500, 810], [210, 730]], np.float32),
        canvas_size=(660, 880),
        bg_gray=45,
    )
    detected = __import__("docscan.core.geometry", fromlist=["detect_quad"]).detect_quad(tilted)
    from docscan.core import geometry

    warped = geometry.warp_perspective(tilted, detected)
    save(
        _pair(tilted, warped, "① 几何校正", f"角点置信度 {detected.confidence:.2f}"),
        out / "01_geometry.png",
    )

    # ② 光照归一
    shadowed, _ = synthesize(gray, "shadow", lo=0.35)
    corrected = illumination.remove_shading(shadowed.astype(np.float32) / 255.0, sigma=35)
    save(
        _pair(
            shadowed,
            np.rint(corrected * 255).astype(np.uint8),
            "② 光照归一（背景估计相除）",
            "块均值标准差降低 > 5×",
        ),
        out / "02_illumination.png",
    )

    # ③ 频域去网纹
    clean = (gray.astype(np.float32) / 255.0 * 0.75).astype(np.float32)
    striped = add_moire(clean, wavelength=24, angle_deg=20.0, amplitude=0.2)
    spec = freq.magnitude_spectrum(striped)
    peaks = freq.find_periodic_peaks(spec, top_n=2)
    denoised = freq.notch_filter(striped, peaks, kind="butterworth", radius=4, order=4)
    save(
        _pair(
            np.rint(striped * 255).astype(np.uint8),
            np.rint(denoised * 255).astype(np.uint8),
            "③ 频域去网纹（Butterworth 陷波）",
            f"SSIM {metrics.ssim(clean, denoised):.3f}",
        ),
        out / "03_notch.png",
    )

    # ④ 维纳复原
    psf = restore.motion_psf(31, 11, 0.0)
    blurred = restore.convolve_blur(clean, psf, noise_sigma=0.005)
    restored = restore.wiener(blurred, psf, K=0.004)
    save(
        _pair(
            np.rint(blurred * 255).astype(np.uint8),
            np.rint(restored * 255).astype(np.uint8),
            "④ 维纳滤波复原（已知 PSF）",
            f"PSNR {metrics.psnr(clean, restored):.1f} dB",
        ),
        out / "04_wiener.png",
    )

    # ⑤ 二值化
    bw = binarize.sauvola(gray, window=25, k=0.2)
    bw = binarize.clean_small_components(bw, min_area=12)
    save(_pair(gray, bw, "⑤ Sauvola 二值化", "积分图加速"), out / "05_binarize.png")

    print("全部素材生成完毕")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
