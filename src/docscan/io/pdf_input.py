"""PDF 输入：逐页渲染为 BGR uint8，与图像输入共用同一条装载链路。

渲染精度由 DPI 决定：屏幕预览 100~150 DPI 已够，OCR 建议 ≥200。
长边超过 ``max_side`` 时按比例缩小，防止 A0 图纸把内存吃穿。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from docscan.core.types import Image

DEFAULT_DPI = 150
DEFAULT_MAX_SIDE = 3000
DEFAULT_MAX_PAGES = 50


def render_pdf(
    path: str | Path,
    *,
    dpi: float = DEFAULT_DPI,
    max_side: int = DEFAULT_MAX_SIDE,
    max_pages: int = DEFAULT_MAX_PAGES,
) -> list[Image]:
    """把 PDF 每页渲染为 BGR uint8 图像。

    Args:
        path: PDF 文件路径。
        dpi: 渲染分辨率（PDF 内部单位是 1/72 英寸，scale = dpi/72）。
        max_side: 单页长边上限（像素），超出按比例缩小。
        max_pages: 页数上限，防止误选几百页的书稿把 UI 卡死。

    Returns:
        每页一张 (H, W, 3) uint8 BGR，顺序与文档一致。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: 不是 PDF、页数为 0 或超过 ``max_pages``。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"文件不存在：{p}")

    import pypdfium2 as pdfium

    try:
        doc = pdfium.PdfDocument(p)
    except Exception as exc:  # pypdfium2 对损坏文件抛的类型不稳定，统一收敛
        raise ValueError(f"PDF 无法打开：{p.name}（{exc}）") from exc

    try:
        n_pages = len(doc)
        if n_pages == 0:
            raise ValueError(f"PDF 没有任何页面：{p.name}")
        if n_pages > max_pages:
            raise ValueError(
                f"PDF 共 {n_pages} 页，超过上限 {max_pages}（批量处理请用 CLI）"
            )

        scale = max(dpi, 36.0) / 72.0
        pages: list[Image] = []
        for i in range(n_pages):
            page = doc[i]
            try:
                bitmap = page.render(scale=scale)
                try:
                    pil = bitmap.to_pil().convert("RGB")
                finally:
                    bitmap.close()
            finally:
                page.close()

            arr = np.asarray(pil)
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            h, w = bgr.shape[:2]
            longest = max(h, w)
            if longest > max_side:
                factor = max_side / longest
                bgr = cv2.resize(
                    bgr,
                    (max(1, round(w * factor)), max(1, round(h * factor))),
                    interpolation=cv2.INTER_AREA,
                )
            pages.append(bgr)
    finally:
        doc.close()
    return pages
