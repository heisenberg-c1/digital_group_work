"""实验工具测试：CSV 写盘的字段转义（历史缺陷 D8 的守卫）。"""

from __future__ import annotations

import csv

import pytest

from docscan.experiments import utils


class TestWriteCsv:
    def test_comma_in_field_round_trips(self, tmp_path):
        """方法名含逗号时不得把字段拆成两列（D8 原症状）。"""
        path = utils.write_csv(
            tmp_path / "binarize.csv",
            ["method", "ssim"],
            [["Sauvola(25,k=0.2)", "0.91"], ["自适应均值(31,C=10)", "0.88"]],
        )
        with path.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert [r["method"] for r in rows] == ["Sauvola(25,k=0.2)", "自适应均值(31,C=10)"]
        assert all(len(r) == 2 for r in rows)
        assert "KeyError" not in path.read_text(encoding="utf-8")

    def test_quoting_is_applied(self, tmp_path):
        path = utils.write_csv(tmp_path / "q.csv", ["a"], [["x,y"]])
        assert '"x,y"' in path.read_text(encoding="utf-8")

    def test_creates_parent_dir(self, tmp_path):
        path = utils.write_csv(tmp_path / "deep" / "nested" / "r.csv", ["a"], [[1]])
        assert path.exists()

    def test_header_only(self, tmp_path):
        path = utils.write_csv(tmp_path / "empty.csv", ["a", "b"], [])
        assert path.read_text(encoding="utf-8").strip() == "a,b"

    @pytest.mark.parametrize("bad", ['say "hi"', "line1\nline2"])
    def test_quotes_and_newlines_round_trip(self, tmp_path, bad):
        path = utils.write_csv(tmp_path / "e.csv", ["label"], [[bad]])
        with path.open(encoding="utf-8", newline="") as fh:
            assert next(iter(csv.DictReader(fh)))["label"] == bad
