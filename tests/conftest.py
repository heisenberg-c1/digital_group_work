"""测试公共夹具：程序化生成「假文档」，不依赖任何真实照片。

真实照片不能进仓库（体积 + 隐私），而合成图的真值（页面四角、光照场、条纹频率）
是已知的，能把「算法对不对」变成可断言的数值。
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from PIL import Image as PILImage
from PIL import ImageDraw


def make_page(
    width: int = 620,
    height: int = 880,
    *,
    paper: int = 238,
    ink: int = 25,
    margin: int = 50,
    line_count: int = 24,
) -> np.ndarray:
    """生成一张白底黑字的合成文档（BGR uint8）。

    内容为等距的黑色横条，模拟文字行：既有一致的笔画宽度（便于测锐化/二值化），
    又有清晰的边缘（便于测 SSIM 与振铃）。
    """
    canvas = PILImage.new("RGB", (width, height), (paper, paper, paper))
    draw = ImageDraw.Draw(canvas)
    margin = max(6, min(margin, width // 6, height // 6))
    x0, x1 = margin, width - margin
    y = margin
    step = max(10, (height - 2 * margin) // max(line_count, 1))
    if x1 - x0 >= 12 and step >= 10:
        for i in range(line_count):
            if y + step > height - margin:
                break
            right = x1 - (i % 4) * ((x1 - x0) // 6)
            draw.rectangle([x0, y, right, y + max(4, step // 3)], fill=(ink, ink, ink))
            y += step
    else:
        # 画布太小放不下文字行：画一条中线保证仍有前景内容
        draw.rectangle([x0, height // 2 - 2, x1, height // 2 + 2], fill=(ink, ink, ink))
    # 外框：给四点检测一个明确的矩形边界
    draw.rectangle([2, 2, width - 3, height - 3], outline=(ink, ink, ink), width=3)
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def paste_on_background(
    page: np.ndarray,
    quad: np.ndarray,
    *,
    canvas_size: tuple[int, int] = (900, 1200),
    bg_gray: int = 60,
) -> np.ndarray:
    """把页面按给定四角贴到深灰背景上，用于模拟「手机斜拍」。

    Args:
        page: 页面图像（BGR）。
        quad: (4,2) 目标四角坐标（tl, tr, br, bl），即透视后的真值。
        canvas_size: 背景尺寸 (W, H)。
        bg_gray: 背景灰度。

    Returns:
        BGR uint8 画布。
    """
    cw, ch = canvas_size
    canvas = np.full((ch, cw, 3), bg_gray, dtype=np.uint8)
    ph, pw = page.shape[:2]
    src = np.array([[0, 0], [pw - 1, 0], [pw - 1, ph - 1], [0, ph - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, np.asarray(quad, dtype=np.float32))
    warped = cv2.warpPerspective(page, matrix, (cw, ch), borderValue=(bg_gray,) * 3)
    mask = cv2.warpPerspective(
        np.full((ph, pw), 255, np.uint8), matrix, (cw, ch), borderValue=0
    )
    canvas[mask > 0] = warped[mask > 0]
    return canvas


def quad_mask(quad: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """把四边形画成实心掩膜（uint8 {0,255}）。"""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.rint(quad).astype(np.int32), 255)
    return mask


def polygon_iou(a: np.ndarray, b: np.ndarray) -> float:
    """两个四边形的 IoU。"""
    m1 = quad_mask(a, (2000, 2000)) > 0
    m2 = quad_mask(b, (2000, 2000)) > 0
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return float(inter / union) if union else 0.0


@pytest.fixture
def synthetic_page():
    """返回 ``make_page`` 的工厂函数，可按需定制尺寸与内容。"""
    return make_page


@pytest.fixture
def tilted_page():
    """一张带已知透视的「手机斜拍」图，返回 (图像, 真值四边形)。"""
    page = make_page()
    quad = np.array(
        [[180, 140], [720, 210], [660, 1080], [230, 1010]],
        dtype=np.float32,
    )
    return paste_on_background(page, quad), quad
