"""图像导出：PNG / JPG（Unicode 路径安全）。

``cv2.imwrite`` 在 Windows 上无法处理含中文的路径，故落盘统一走 Pillow。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from docscan.core.types import Image, to_uint8


def save_image(img: Image, path: str | Path, *, quality: int = 95) -> Path:
    """保存图像，按扩展名推断格式。

    Args:
        img: BGR 图像（uint8 或 float32[0,1]）；灰度二值图（2 维）亦可。
        path: 输出路径，支持 .png / .jpg / .jpeg / .bmp / .tif。
        quality: JPEG 质量。

    Returns:
        实际写入的路径。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    arr = to_uint8(img)
    pil = PILImage.fromarray(arr[:, :, ::-1]) if arr.ndim == 3 else PILImage.fromarray(arr)

    suffix = p.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        pil.save(p, quality=quality, subsampling=0)
    else:
        pil.save(p)
    return p


def bgr_to_pil(img: Image) -> PILImage.Image:
    """BGR ndarray → PIL.Image（供 tkinter 显示与 PDF 内嵌复用）。"""
    arr = to_uint8(np.asarray(img))
    if arr.ndim == 2:
        return PILImage.fromarray(arr)
    return PILImage.fromarray(arr[:, :, ::-1])


def export_pdf(
    images: list[Image],
    path: str | Path,
    *,
    dpi: int = 300,
) -> Path:
    """多页导出 PDF。

    二值页（2 维 {0,255}）以 1-bit + CCITT G4 压缩编码——黑白文档体积
    通常只有原图的 1/10 以下；彩色/灰度页以常规方式内嵌。

    Args:
        images: BGR / 灰度图像列表，每张一页。
        path: 输出 PDF 路径。
        dpi: 页面分辨率（决定打印尺寸）。

    Returns:
        实际写入的路径。
    """
    if not images:
        raise ValueError("至少需要一页图像")

    pages: list[PILImage.Image] = []
    for img in images:
        arr = to_uint8(np.asarray(img))
        if arr.ndim == 2:
            pil = PILImage.fromarray(arr).convert("1")  # 1-bit，供 G4 压缩
        else:
            pil = PILImage.fromarray(arr[:, :, ::-1]).convert("RGB")
        pages.append(pil)

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    head, tail = pages[0], pages[1:]
    kwargs: dict = {"resolution": float(dpi)}
    if head.mode == "1":
        kwargs["compression"] = "group4"
    head.save(p, save_all=True, append_images=tail, format="PDF", **kwargs)
    return p
