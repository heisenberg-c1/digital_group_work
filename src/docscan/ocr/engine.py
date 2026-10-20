"""OCR 封装：RapidOCR 懒加载，缺失时降级为不可用而非崩溃。

设计：``OcrEngine`` 是全项目唯一触碰 OCR 库的地方。核心算法层（core/）
不依赖 OCR —— CER 指标只接收「识别出的文本」字符串。
"""

from __future__ import annotations

import logging

from docscan.core.types import Image, to_uint8

log = logging.getLogger("docscan.ocr")


class OcrEngine:
    """懒加载的 OCR 引擎包装。

    用法::

        engine = OcrEngine()
        if engine.available():
            text = engine.recognize(img)
    """

    def __init__(self) -> None:
        self._engine = None
        self._tried = False
        self._available = False

    def available(self) -> bool:
        """OCR 库是否可用。首次调用会尝试 import 并缓存结果。"""
        if not self._tried:
            self._tried = True
            try:
                import rapidocr_onnxruntime  # noqa: F401

                self._available = True
            except ImportError:
                self._available = False
                log.info("RapidOCR 未安装，OCR 相关功能降级（安装：uv pip install -e '.[ocr]'）")
        return self._available

    def recognize(self, img: Image) -> str | None:
        """识别图像文本。

        Args:
            img: BGR 图像（uint8 或 float）。

        Returns:
            识别出的全文（各文本行以换行连接）；不可用或失败时返回 None。
        """
        if not self.available():
            return None
        if self._engine is None:
            from rapidocr_onnxruntime import RapidOCR

            self._engine = RapidOCR()

        u8 = to_uint8(img)
        try:
            result, _elapsed = self._engine(u8)
        except Exception as exc:
            log.warning("OCR 识别失败：%s", exc)
            return None

        if not result:
            return ""
        return "\n".join(str(line[1]) for line in result)


_default_engine: OcrEngine | None = None


def get_engine() -> OcrEngine:
    """进程级共享引擎（懒加载 + 缓存 import 判断）。"""
    global _default_engine
    if _default_engine is None:
        _default_engine = OcrEngine()
    return _default_engine
