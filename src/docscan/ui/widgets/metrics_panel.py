"""指标面板：以「键-值」两列展示当前阶段或整链的量化指标。"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from docscan.ui.theme import Theme


class MetricsPanel(ttk.Frame):
    """右侧指标面板。"""

    def __init__(self, master, theme: Theme, **kwargs) -> None:
        super().__init__(master, style="Panel.TFrame", **kwargs)
        self._theme = theme
        self._rows: dict[str, tuple[ttk.Label, ttk.Label]] = {}
        self._last_render = ("overall", (None, None, None, False))  # 主题切换后重绘用

        head = ttk.Label(self, text="指标", style="PanelHead.TLabel")
        head.pack(anchor="w", padx=12, pady=(12, 4))
        ttk.Label(self, text="整链汇总或当前阶段的量化结果", style="PanelDim.TLabel").pack(
            anchor="w", padx=12, pady=(0, 8)
        )
        self._body = ttk.Frame(self, style="Panel.TFrame")
        self._body.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

        self.show_overall(None)

    def _clear(self) -> None:
        for child in self._body.winfo_children():
            child.destroy()
        self._rows.clear()

    def show_overall(
        self,
        total_ms: float | None,
        psnr: float | None = None,
        ssim: float | None = None,
        aligned: bool = False,
    ) -> None:
        """显示整链汇总（含与原图的 PSNR / SSIM 对比）。"""
        self._last_render = ("overall", (total_ms, psnr, ssim, aligned))
        self._clear()
        if total_ms is None:
            self._add_row("状态", "尚未运行")
            self._add_row("总耗时", "—")
            self._add_row("PSNR", "—")
            self._add_row("SSIM", "—")
        else:
            self._add_row("状态", "完成", color=self._theme.colors["ok"])
            self._add_row("总耗时", f"{total_ms:.0f} ms")
            self._add_row("PSNR", _fmt_psnr(psnr))
            self._add_row("SSIM", "—" if ssim is None else f"{ssim:.4f}")
            if aligned:
                self._add_note("几何校正改变了尺寸，对比前已按结果尺寸对齐参考图")

    def show_stage(
        self, label: str, notes: str, metrics: dict[str, float], elapsed_ms: float
    ) -> None:
        """显示单个阶段的指标。"""
        self._last_render = ("stage", (label, notes, metrics, elapsed_ms))
        self._clear()
        self._add_row("阶段", label)
        self._add_row("决策", notes)
        self._add_row("耗时", f"{elapsed_ms:.0f} ms")
        for key, value in metrics.items():
            text = f"{value:.3f}" if isinstance(value, float) else str(value)
            self._add_row(key, text)

    def _add_row(self, key: str, value: str, color: str | None = None) -> None:
        c = self._theme.colors
        row = ttk.Frame(self._body, style="Panel.TFrame")
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text=key, style="PanelDim.TLabel", width=10, anchor="w").pack(side=tk.LEFT)
        val = ttk.Label(
            row,
            text=value,
            style="Metric.TLabel",
            foreground=color or c["text"],
            anchor="e",
        )
        val.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _add_note(self, text: str) -> None:
        ttk.Label(
            self._body, text=text, style="PanelDim.TLabel", wraplength=180, justify=tk.LEFT
        ).pack(fill=tk.X, pady=(4, 0))

    def apply_theme(self) -> None:
        """明暗切换后按最后一次渲染重绘（颜色取自最新 token）。"""
        kind, args = self._last_render
        if kind == "overall":
            self.show_overall(*args)
        else:
            self.show_stage(*args)


def _fmt_psnr(value: float | None) -> str:
    if value is None:
        return "—"
    if value == float("inf"):
        return "∞（与原图逐像素一致）"
    return f"{value:.2f} dB"
