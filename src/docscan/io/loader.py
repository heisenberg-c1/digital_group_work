"""图像装载：EXIF 方向校正、透明底压白、统一为 BGR uint8、尺寸策略。

这里是全流程唯一的「色彩空间入口」：任何外部文件进来后一律变成 BGR uint8，
之后所有模块都按 BGR 处理，避免红蓝通道反掉的低级错误。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage
from PIL import ImageOps

from docscan.core.types import Image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

DEFAULT_MAX_PIXELS = 20_000_000
DEFAULT_MAX_SIDE = 4000


def _pil_to_bgr(im: PILImage.Image) -> np.ndarray:
    """PIL 图像 → BGR uint8，透明区域压在白色背景上。"""
    if im.mode in ("I", "I;16", "I;16B", "F"):
        # 16 位/浮点灰度（扫描仪 TIFF 常见）：线性压到 8 位
        arr = np.asarray(im).astype(np.float32)
        hi = float(arr.max()) or 1.0
        gray = np.rint(arr / hi * 255.0).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    has_alpha = im.mode in ("RGBA", "LA", "PA") or (
        im.mode == "P" and "transparency" in im.info
    )
    if has_alpha:
        rgba = im.convert("RGBA")
        canvas = PILImage.new("RGB", rgba.size, (255, 255, 255))
        canvas.paste(rgba, mask=rgba.split()[3])
        rgb = np.asarray(canvas)
    else:
        rgb = np.asarray(im.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def load_image(
    path: str | Path,
    *,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    max_side: int = DEFAULT_MAX_SIDE,
    auto_downscale: bool = True,
) -> Image:
    """读取图像文件，输出 BGR uint8。

    处理三件事：
    1. EXIF 方向（竖拍照片常以横向像素存储，靠 Orientation 标记转正）；
    2. 透明底（PNG 抠图/贴纸）压到白底，避免变成黑块；
    3. 超大图按长边降采样——12MP 原图直接进算法链会让维纳/FFT 慢到不可用，
       而扫描件 4000px 长边已足够 300 DPI 输出。

    Args:
        path: 图像路径。
        max_pixels: 像素总数上限，超过则触发降采样。
        max_side: 降采样后的长边上限。
        auto_downscale: 为 False 时不做任何缩放（导出原尺寸用）。

    Returns:
        (H, W, 3) uint8 BGR。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: 扩展名不支持或图像无法解码。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"文件不存在：{p}")
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"不支持的图像格式：{p.suffix}")

    try:
        with PILImage.open(p) as im:
            im = ImageOps.exif_transpose(im)
            bgr = _pil_to_bgr(im)
    except (OSError, ValueError) as exc:
        raise ValueError(f"图像解码失败：{p.name}（{exc}）") from exc

    if auto_downscale:
        h, w = bgr.shape[:2]
        if h * w > max_pixels and max(h, w) > max_side:
            scale = max_side / float(max(h, w))
            bgr = cv2.resize(
                bgr,
                (max(1, round(w * scale)), max(1, round(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
    return bgr


def load_any(path: str | Path, **kwargs) -> list[Image]:
    """读取图像或 PDF，统一返回 BGR uint8 图像列表（PDF 每页一张）。"""
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        from docscan.io.pdf_input import render_pdf

        return render_pdf(p, **kwargs)
    return [load_image(p, **kwargs)]
