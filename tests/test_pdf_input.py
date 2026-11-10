"""PDF 输入测试：用 Pillow 现场生成 PDF，对拍页数、尺寸与格式。"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image as PILImage

from docscan.io.loader import load_any
from docscan.io.pdf_input import render_pdf


@pytest.fixture
def one_page_pdf(tmp_path):
    """单页纯白 PDF（200×280 pt，四角留一个黑方块便于断言内容）。"""
    pil = PILImage.new("RGB", (200, 280), (255, 255, 255))
    arr = np.asarray(pil).copy()
    arr[10:30, 10:30] = (0, 0, 0)
    page = PILImage.fromarray(arr)
    path = tmp_path / "demo.pdf"
    page.save(path, format="PDF", resolution=72.0)
    return path, arr


class TestRenderPdf:
    def test_single_page_shape_and_dtype(self, one_page_pdf):
        path, _arr = one_page_pdf
        pages = render_pdf(path, dpi=72)
        assert len(pages) == 1
        img = pages[0]
        assert img.ndim == 3 and img.dtype == np.uint8
        # 72 DPI 时 1 pt = 1 px，渲染尺寸应与页面点数一致
        assert img.shape[:2] == (280, 200)

    def test_content_black_block_preserved(self, one_page_pdf):
        path, _arr = one_page_pdf
        pages = render_pdf(path, dpi=72, max_side=10000)
        img = pages[0]
        # 左上角黑块中心仍应是暗像素（BGR 均值 < 64）
        assert img[20, 20].mean() < 64

    def test_max_side_downscales(self, one_page_pdf):
        path, _arr = one_page_pdf
        pages = render_pdf(path, dpi=300, max_side=200)
        img = pages[0]
        assert max(img.shape[:2]) <= 200

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            render_pdf(tmp_path / "nope.pdf")


class TestLoadAnyPdf:
    def test_load_any_routes_pdf(self, one_page_pdf):
        path, _arr = one_page_pdf
        pages = load_any(path)
        assert len(pages) == 1
        assert pages[0].dtype == np.uint8
