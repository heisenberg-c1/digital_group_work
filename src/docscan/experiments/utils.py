"""实验公共工具：真实文本页渲染、OCR 文本获取、采样/量化模拟。

CER 实验需要「有真文字」的页面（纯色条无法算 CER），所以这里用系统
中文字体渲染文本页；OCR 用 RapidOCR（未安装时自动降级为 SSIM 代理指标，
实验仍然可跑，CSV 会标注 metric 列名）。
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from docscan.core.types import Image
from docscan.ocr.engine import get_engine

log = logging.getLogger("docscan.experiments")

# Windows 自带中文字体，按优先级探测
_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\arial.ttf",
]

SAMPLE_TEXT = (
    "数字图像处理课程设计：手机文档扫描与可读性增强系统。"
    "本系统支持透视校正、光照归一、频域去网纹、维纳滤波复原、"
    "Sauvola二值化与PDF导出。采样定理告诉我们，低于奈奎斯特频率的"
    "采样会产生混叠失真，量化位数决定灰度分辨率。"
)


def find_font() -> str | None:
    for p in _FONT_CANDIDATES:
        if Path(p).exists():
            return p
    return None


def render_text_page(
    text: str = SAMPLE_TEXT,
    *,
    width: int = 1240,
    height: int = 1754,  # A4 @150dpi 比例
    font_px: int = 42,
    margin: int = 90,
    paper: int = 235,
    ink: int = 25,
) -> tuple[Image, str]:
    """渲染一张真实文字页面（BGR uint8），返回 (图像, 全文)。"""
    from PIL import Image as PILImage
    from PIL import ImageDraw, ImageFont

    font_path = find_font()
    if font_path is None:
        raise RuntimeError("未找到系统中文字体（simhei/simsun/msyh），无法渲染文本页")

    pil = PILImage.new("RGB", (width, height), (paper, paper, paper))
    draw = ImageDraw.Draw(pil)
    font = ImageFont.truetype(font_path, font_px)
    y = margin
    line_h = int(font_px * 1.7)
    for paragraph in text.split("。"):
        if not paragraph:
            continue
        paragraph += "。"
        line = ""
        for ch in paragraph:
            line += ch
            if draw.textlength(line, font=font) > width - 2 * margin:
                draw.text((margin, y), line[:-1], fill=(ink, ink, ink), font=font)
                y += line_h
                line = ch
        if line:
            draw.text((margin, y), line, fill=(ink, ink, ink), font=font)
            y += line_h
        if y > height - margin:
            break
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR), text


def quantize(gray: Image, bits: int) -> Image:
    """位深度量化：bits=8 恒等，bits=1 只剩黑白。"""
    levels = 2**bits
    f = gray.astype(np.float64)
    q = np.rint(f / 255.0 * (levels - 1)) / (levels - 1) * 255.0
    return np.rint(q).astype(np.uint8)


def resample_dpi(gray: Image, target_dpi: int, *, base_dpi: int = 300) -> Image:
    """模拟以 target_dpi 采样：面积插值降采样后放大回原尺寸。"""
    scale = min(1.0, target_dpi / base_dpi)
    if scale >= 1.0:
        return gray
    h, w = gray.shape[:2]
    small = cv2.resize(gray, (max(1, round(w * scale)), max(1, round(h * scale))),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


class TextRecognizer:
    """OCR 文本获取（带缓存与降级说明）。"""

    def __init__(self) -> None:
        self.engine = get_engine()
        self._warned = False

    @property
    def uses_ocr(self) -> bool:
        return self.engine.available()

    def text(self, img: Image) -> str | None:
        if not self.engine.available():
            if not self._warned:
                log.warning("RapidOCR 未安装，CER 实验将改用 SSIM 代理指标")
                self._warned = True
            return None
        return self.engine.recognize(img)


def setup_matplotlib_cjk() -> None:
    """让 matplotlib 正确渲染中文（Windows 用微软雅黑/黑体）。"""
    import matplotlib

    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DengXian"]
    matplotlib.rcParams["axes.unicode_minus"] = False
