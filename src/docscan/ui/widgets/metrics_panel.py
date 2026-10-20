"""指标面板：以「键-值」两列展示当前阶段的量化指标。"""

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

        head = ttk.Label(self, text="指标", style="PanelHead.TLabel")
        head.pack(anchor="w", padx=12, pady=(12, 4))
        ttk.Label(self, text="当前查看阶段的量化结果", style="PanelDim.TLabel").pack(
            anchor="w", padx=12, pady=(0, 8)
        )
        self._body = ttk.Frame(self, style="Panel.TFrame")
        self._body.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

        self.show_overall(None)

    def _clear(self) -> None:
        for child in self._body.winfo_children():
            child.destroy()
        self._rows.clear()

    def show_overall(self, total_ms: float | None) -> None:
        """显示整链汇总。"""
        self._clear()
        if total_ms is None:
            self._add_row("状态", "尚未运行")
            self._add_row("总耗时", "—")
        else:
            self._add_row("状态", "完成", color=self._theme.colors["ok"])
            self._add_row("总耗时", f"{total_ms:.0f} ms")

    def show_stage(
        self, label: str, notes: str, metrics: dict[str, float], elapsed_ms: float
    ) -> None:
        """显示单个阶段的指标。"""
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
