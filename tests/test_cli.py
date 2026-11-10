"""CLI 测试：命令行必须与 GUI 走同一条处理链，且能导出 PNG / PDF。

回归背景：早期 ``--cli`` 只做「校正 + 自适应二值化」，光照归一、去网纹、
复原全部缺失——README 却把 CLI 当主要入口展示。
"""

from __future__ import annotations

import cv2
import numpy as np

from conftest import make_page
from docscan.__main__ import main


def _write_page(path, **kwargs):
    cv2.imwrite(str(path), make_page(**kwargs))
    return path


class TestCliOutput:
    def test_bw_png_is_binary(self, tmp_path):
        src = _write_page(tmp_path / "page.png")
        out = tmp_path / "out.png"
        assert main(["--cli", str(src), "-o", str(out), "--mode", "bw"]) == 0
        img = cv2.imread(str(out), cv2.IMREAD_GRAYSCALE)
        assert img is not None
        assert set(np.unique(img)).issubset({0, 255})

    def test_color_png_keeps_three_channels(self, tmp_path):
        src = _write_page(tmp_path / "page.png")
        out = tmp_path / "out.png"
        assert main(["--cli", str(src), "-o", str(out), "--mode", "color"]) == 0
        img = cv2.imread(str(out))
        assert img is not None and img.ndim == 3

    def test_pdf_export_single_page(self, tmp_path):
        src = _write_page(tmp_path / "page.png")
        out = tmp_path / "out.pdf"
        assert main(["--cli", str(src), "-o", str(out), "--mode", "bw"]) == 0
        assert out.exists() and out.stat().st_size > 500

    def test_extra_stages_are_applied(self, tmp_path, monkeypatch):
        """--enable 必须真的把阶段接进链里（而不是只做二值化）。"""
        from docscan import __main__ as cli

        seen: list[list[str]] = []
        real = cli._process_one

        def spy(src, dst, args):
            result = real(src, dst, args)
            seen.append([s.name for s in cli._build_stages(args)])
            return result

        monkeypatch.setattr(cli, "_process_one", spy)
        src = _write_page(tmp_path / "page.png")
        out = tmp_path / "out.png"
        assert main(["--cli", str(src), "-o", str(out), "--enable", "denoise,freq"]) == 0
        assert "denoise" in seen[0] and "freq" in seen[0]

    def test_unknown_stage_is_rejected(self, tmp_path, capsys):
        src = _write_page(tmp_path / "page.png")
        out = tmp_path / "out.png"
        code = main(["--cli", str(src), "-o", str(out), "--enable", "nope"])
        assert code == 1
        assert "未知阶段" in capsys.readouterr().err

    def test_no_warp_disables_stage(self, tmp_path, monkeypatch):
        from docscan import __main__ as cli

        captured: dict = {}
        real = cli._build_stages

        def spy(args):
            stages = real(args)
            captured["warp"] = {s.name: s.enabled for s in stages}["warp"]
            return stages

        monkeypatch.setattr(cli, "_build_stages", spy)
        src = _write_page(tmp_path / "page.png")
        main(["--cli", str(src), "-o", str(tmp_path / "o.png"), "--no-warp"])
        assert captured["warp"] is False

    def test_missing_input_reports_error(self, tmp_path, capsys):
        code = main(["--cli", str(tmp_path / "nope.png"), "-o", str(tmp_path / "o.png")])
        assert code == 1
        assert "错误" in capsys.readouterr().err
