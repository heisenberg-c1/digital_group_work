"""处理链列表：每行一个阶段的「启用勾选 + 点击查看输出」。"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from docscan.pipeline.stages import Stage


class StageList(ttk.Frame):
    """左侧处理链面板。

    回调：
        on_toggle(name, enabled) —— 勾选/取消某阶段
        on_select(name) —— 点击阶段名，请求查看该阶段输出
    """

    def __init__(self, master, on_toggle, on_select, theme, **kwargs) -> None:
        super().__init__(master, style="Panel.TFrame", **kwargs)
        self._on_toggle = on_toggle
        self._on_select = on_select
        self._theme = theme
        self._vars: dict[str, tk.BooleanVar] = {}
        self._rows: dict[str, ttk.Label] = {}
        self._selected: str | None = None

        head = ttk.Label(self, text="处理链", style="PanelHead.TLabel")
        head.pack(anchor="w", padx=12, pady=(12, 4))
        ttk.Label(self, text="勾选启用 · 点击名称查看该阶段输出", style="PanelDim.TLabel").pack(
            anchor="w", padx=12, pady=(0, 8)
        )
        self._body = ttk.Frame(self, style="Panel.TFrame")
        self._body.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 12))

    def rebuild(self, stages: list[Stage]) -> None:
        """按当前阶段列表重建行。"""
        for child in self._body.winfo_children():
            child.destroy()
        self._vars.clear()
        self._rows.clear()
        self._selected = None

        for stage in stages:
            row = ttk.Frame(self._body, style="Panel.TFrame")
            row.pack(fill=tk.X, pady=2)

            var = tk.BooleanVar(value=stage.enabled)
            var.trace_add(
                "write", lambda *_a, n=stage.name, v=var: self._on_toggle(n, bool(v.get()))
            )
            cb = ttk.Checkbutton(row, variable=var, style="Panel.TCheckbutton", cursor="hand2")
            cb.pack(side=tk.LEFT)

            label = ttk.Label(
                row,
                text=f" {stage.label}",
                style="Panel.TLabel",
                cursor="hand2",
                padding=(6, 4),
            )
            label.pack(side=tk.LEFT, fill=tk.X, expand=True)
            label.bind("<Button-1>", lambda _e, n=stage.name: self.select(n))
            self._vars[stage.name] = var
            self._rows[stage.name] = label

    def select(self, name: str) -> None:
        """高亮并通知选中某阶段。"""
        c = self._theme.colors
        for n, label in self._rows.items():
            label.configure(background=c["accent_soft"] if n == name else c["panel"])
        self._selected = name
        self._on_select(name)

    @property
    def selected(self) -> str | None:
        return self._selected

    def set_stage_image_available(self, name: str, available: bool) -> None:
        """有输出可看的阶段名加粗提示。"""
        label = self._rows.get(name)
        if label is not None:
            label.configure(font=self._theme.font("sm", bold=available))
