"""ttk 主题：DPI 自适应字号、明/暗 token、sv-ttk 优先（缺失时退回 clam 精修）。

间距只用 4/8/12/16/24 五档（4 的倍数），字号阶梯 xs/sm/md/lg（96dpi 基准像素，
实际渲染乘 ``Theme.scale`` = dpi/96）。所有颜色集中在这里，widget 内不得出现
裸十六进制色值。

两套引擎：
* ``sv``    —— 安装了 sv-ttk 时，原生控件交给 Sun Valley 主题，本模块只补充
               项目自有样式（Panel.* / Dim.* / Head.* 等）并同步结构色；
* ``clam``  —— 无 sv-ttk 时的回退，完全用本项目 token 手工配置。
"""

from __future__ import annotations

import contextlib
import tkinter as tk
from tkinter import ttk

FONT_FAMILY = "Microsoft YaHei UI"

# 字号阶梯（96dpi 基准像素高度；tkinter 负数表示像素）
BASE_FONT_SIZES = {"xs": -11, "sm": -12, "md": -13, "lg": -16}

# 间距体系
SP_1, SP_2, SP_3, SP_4, SP_5 = 4, 8, 12, 16, 24

# ---- clam 回退配色（本项目自绘） ----
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

# ---- Sun Valley 同步配色（结构色取自 sv-ttk 2.6 主题源，语义色沿用本项目） ----
SV_LIGHT = {
    "bg": "#fafafa",
    "panel": "#ffffff",
    "border": "#e0e0e0",
    "text": "#1c1c1c",
    "text_dim": "#5f5f5f",
    "accent": "#005fb8",
    "accent_soft": "#e6eff9",
    "ok": "#059669",
    "warn": "#b45309",
    "error": "#b91c1c",
    "canvas": "#2b2f36",
}

SV_DARK = {
    "bg": "#1c1c1c",
    "panel": "#262626",
    "border": "#3b3b3b",
    "text": "#fafafa",
    "text_dim": "#a0a0a0",
    "accent": "#57c8ff",
    "accent_soft": "#173045",
    "ok": "#34d399",
    "warn": "#fbbf24",
    "error": "#f87171",
    "canvas": "#14171c",
}


class Theme:
    """当前生效的主题 token 集合。"""

    def __init__(self, dark: bool = False) -> None:
        self.dark = dark
        self.scale = 1.0  # dpi / 96，app 启动时写入
        self.engine = "clam"  # "sv" | "clam"
        self.colors = dict(DARK if dark else LIGHT)

    # ---- 字体 ----

    def font(self, size_key: str = "md", bold: bool = False) -> tuple:
        weight = "bold" if bold else "normal"
        px = max(9, round(abs(BASE_FONT_SIZES[size_key]) * self.scale))
        return (FONT_FAMILY, -px, weight)

    # ---- 应用 ----

    def apply(self, root: tk.Misc) -> None:
        """把主题写入根窗口与 ttk Style；sv-ttk 可用则由其接管原生控件。"""
        sv_ok = False
        with contextlib.suppress(Exception):
            import sv_ttk

            sv_ttk.set_theme("dark" if self.dark else "light", root)
            sv_ok = True
        self.engine = "sv" if sv_ok else "clam"

        base = (SV_DARK if self.dark else SV_LIGHT) if sv_ok else DARK if self.dark else LIGHT
        self.colors = dict(base)

        style = ttk.Style(root)
        root.configure(bg=self.colors["bg"])
        if not sv_ok:
            with contextlib.suppress(tk.TclError):
                style.theme_use("clam")

        # 中文界面统一字体（sv-ttk 未显式设字体，覆盖 "." 即可全局生效）
        style.configure(".", font=self.font("sm"))

        self._configure_common(style)
        if not sv_ok:
            self._configure_clam_base(style)

    def _configure_common(self, style: ttk.Style) -> None:
        """两种引擎共用的项目自有样式（均为本模块命名空间，不与原生冲突）。"""
        c = self.colors

        def cfg(
            name: str,
            bg: str | None,
            fg: str,
            size: str = "sm",
            bold: bool = False,
            **extra: object,
        ) -> None:
            options: dict = {"foreground": fg, "font": self.font(size, bold)}
            if bg is not None:
                options["background"] = bg
            style.configure(name, **options, **extra)

        # 面板容器与标签：bg=None 表示保持透明，随父容器底色
        cfg("Panel.TFrame", c["panel"], c["text"])
        cfg("Dim.TLabel", None, c["text_dim"], size="xs")
        cfg("PanelDim.TLabel", c["panel"], c["text_dim"], size="xs")
        cfg("Head.TLabel", None, c["text"], size="md", bold=True)
        cfg("PanelHead.TLabel", c["panel"], c["text"], size="md", bold=True)
        cfg("Metric.TLabel", c["panel"], c["text"], bold=True)
        cfg("Status.TLabel", c["bg"], c["text_dim"], size="xs")

        style.configure(
            "Panel.TCheckbutton",
            background=c["panel"], foreground=c["text"], font=self.font("sm"),
        )
        style.map(
            "Panel.TCheckbutton",
            background=[("active", c["panel"])],
            foreground=[("disabled", c["text_dim"])],
        )

        style.configure("TLabelframe", background=c["bg"], bordercolor=c["border"])
        style.configure("TLabelframe.Label", background=c["bg"], foreground=c["text_dim"])

    def _configure_clam_base(self, style: ttk.Style) -> None:
        """clam 回退路径：基础控件全部手工着色。"""
        c = self.colors

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
        cfg("TFrame", c["bg"], c["text"])
        cfg("TLabel", c["bg"], c["text"])
        cfg("TButton", c["panel"], c["text"], padding=(SP_2, SP_1))
        style.map(
            "TButton",
            background=[("active", c["accent_soft"]), ("pressed", c["accent_soft"])],
        )
        cfg("Accent.TButton", c["accent"], "#ffffff")
        style.map("Accent.TButton", background=[("active", c["accent"])])
        cfg("TCheckbutton", c["bg"], c["text"])
        style.map("TCheckbutton", background=[("active", c["bg"])])
        cfg("TPanedwindow", c["bg"], c["text"])
        # PanedWindow 的分隔条
        style.configure("Sash", background=c["border"], sashthickness=4)

    def toggle(self, root: tk.Misc) -> None:
        """明暗切换（sv-ttk 支持运行时换肤，clam 路径靠重新 configure）。"""
        self.dark = not self.dark
        self.apply(root)


def current_theme(dark: bool = False) -> Theme:
    return Theme(dark=dark)
