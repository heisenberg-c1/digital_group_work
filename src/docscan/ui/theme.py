"""ttk 主题：明/暗配色 token、字号阶梯、间距体系。

间距只用 4/8/12/16/24 五档（4 的倍数），字号阶梯 xs/sm/md/lg。
所有颜色集中在这里，widget 内不得出现裸十六进制色值。
"""

from __future__ import annotations

import contextlib
import tkinter as tk
from tkinter import ttk

FONT_FAMILY = "Microsoft YaHei UI"

# 字号阶梯（tkinter 负数表示像素高度）
FONT_SIZES = {"xs": -11, "sm": -12, "md": -13, "lg": -16}

# 间距体系
SP_1, SP_2, SP_3, SP_4, SP_5 = 4, 8, 12, 16, 24

LIGHT = {
    "bg": "#f4f6f8",
    "panel": "#ffffff",
    "border": "#d8dee6",
    "text": "#1f2937",
    "text_dim": "#697280",
    "accent": "#2563eb",
    "accent_soft": "#e8effc",
    "ok": "#059669",
    "warn": "#b45309",
    "error": "#b91c1c",
    "canvas": "#2b2f36",
}

DARK = {
    "bg": "#1e2228",
    "panel": "#262b33",
    "border": "#3a4150",
    "text": "#e5e9f0",
    "text_dim": "#9aa3b2",
    "accent": "#5b9bff",
    "accent_soft": "#2c3b55",
    "ok": "#34d399",
    "warn": "#fbbf24",
    "error": "#f87171",
    "canvas": "#14171c",
}


class Theme:
    """当前生效的主题 token 集合。"""

    def __init__(self, dark: bool = False) -> None:
        self.dark = dark
        self.colors = dict(DARK if dark else LIGHT)

    def font(self, size_key: str = "md", bold: bool = False) -> tuple:
        weight = "bold" if bold else "normal"
        return (FONT_FAMILY, FONT_SIZES[size_key], weight)

    def apply(self, root: tk.Misc) -> None:
        """把主题写入 ttk Style 与根窗口。"""
        c = self.colors
        root.configure(bg=c["bg"])
        style = ttk.Style(root)
        with contextlib.suppress(tk.TclError):
            style.theme_use("clam")

        def cfg(
            name: str,
            bg: str,
            fg: str,
            size: str = "sm",
            bold: bool = False,
            **extra: object,
        ) -> None:
            style.configure(
                name, background=bg, foreground=fg, font=self.font(size, bold), **extra
            )

        style.configure(".", background=c["bg"], foreground=c["text"], borderwidth=0)

        # 容器与标签：前缀 Panel/Dim/Head 区分背景层级与字重
        cfg("TFrame", c["bg"], c["text"])
        cfg("Panel.TFrame", c["panel"], c["text"])
        cfg("TLabel", c["bg"], c["text"])
        cfg("Panel.TLabel", c["panel"], c["text"])
        cfg("Dim.TLabel", c["bg"], c["text_dim"], size="xs")
        cfg("PanelDim.TLabel", c["panel"], c["text_dim"], size="xs")
        cfg("Head.TLabel", c["bg"], c["text"], size="md", bold=True)
        cfg("PanelHead.TLabel", c["panel"], c["text"], size="md", bold=True)
        cfg("Metric.TLabel", c["panel"], c["text"], bold=True)
        cfg("Status.TLabel", c["panel"], c["text_dim"], size="xs")

        # 按钮：普通态用面板色，悬停压浅蓝
        cfg("TButton", c["panel"], c["text"], padding=(SP_2, SP_1))
        style.map(
            "TButton",
            background=[("active", c["accent_soft"]), ("pressed", c["accent_soft"])],
        )
        cfg("Accent.TButton", c["accent"], "#ffffff")
        style.map("Accent.TButton", background=[("active", c["accent"])])

        # 复选框
        cfg("TCheckbutton", c["bg"], c["text"])
        style.map("TCheckbutton", background=[("active", c["bg"])])
        cfg("Panel.TCheckbutton", c["panel"], c["text"])
        style.map("Panel.TCheckbutton", background=[("active", c["panel"])])

        cfg("TLabelframe", c["bg"], c["text"], bordercolor=c["border"])
        cfg("TLabelframe.Label", c["bg"], c["text_dim"], size="xs")
        cfg("TPanedwindow", c["border"], c["text"])


def current_theme(dark: bool = False) -> Theme:
    return Theme(dark=dark)
