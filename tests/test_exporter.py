"""PDF 导出测试。"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_page
from docscan.core import binarize
from docscan.io.exporter import export_pdf


def _bw_page() -> np.ndarray:
    return binarize.adaptive(binarize.to_gray(make_page(300, 420)), block_size=31, c=10.0)


class TestExportPdf:
    def test_page_count_roundtrip(self, tmp_path):
        """导出的 PDF 页数 = 输入页数，可被 pypdfium2 重新打开。"""
        pypdfium2 = pytest.importorskip("pypdfium2")
        pages = [_bw_page() for _ in range(3)]
        out = export_pdf(pages, tmp_path / "doc.pdf")
        assert out.exists()

        pdf = pypdfium2.PdfDocument(str(out))
        assert len(pdf) == 3
        pdf.close()

    def test_bw_pdf_g4_smaller_than_color(self, tmp_path):
        """1-bit G4 压缩的黑白 PDF 应显著小于彩色内嵌版本。"""
        page = make_page(400, 560)
        bw = binarize.adaptive(binarize.to_gray(page), block_size=31, c=10.0)
        p_bw = export_pdf([bw], tmp_path / "bw.pdf")
        p_color = export_pdf([page], tmp_path / "color.pdf")
        assert p_bw.stat().st_size < p_color.stat().st_size
        # 黑白文档 G4 压缩率极高
        raw = 400 * 560 / 8  # 1-bit 原始字节数
        assert p_bw.stat().st_size < raw / 2

    def test_empty_pages_raises(self, tmp_path):
        with pytest.raises(ValueError):
            export_pdf([], tmp_path / "empty.pdf")
