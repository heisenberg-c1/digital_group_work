"""日志面板：只读文本，追加式写入（时间戳 + 级别着色）。"""

from __future__ import annotations

import datetime
import tkinter as tk
from tkinter import ttk

from docscan.ui.theme import Theme

MAX_LINES = 500


class LogView(ttk.Frame):
    """底部日志区。"""

    def __init__(self, master, theme: Theme, height: int = 5, **kwargs) -> None:
        super().__init__(master, **kwargs)
        self._theme = theme
        self.text = tk.Text(
            self,
            height=height,
            state=tk.DISABLED,
            font=theme.font("xs"),
            background=theme.colors["panel"],
            foreground=theme.colors["text_dim"],
            relief=tk.FLAT,
            wrap=tk.WORD,
            highlightthickness=0,
            borderwidth=0,
            padx=8,
            pady=4,
        )
        self._bar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.text.yview)
        self.text.configure(yscrollcommand=self._bar.set)
        self._bar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._configure_tags()

    def _configure_tags(self) -> None:
        c = self._theme.colors
        self.text.tag_configure("ts", foreground=c["border"])
        for name in ("ok", "warn", "error", "text"):
            self.text.tag_configure(name, foreground=c.get(name, c["text"]))

    def append(self, message: str, *, color: str | None = None) -> None:
        """追加一行日志（自动带时间戳），超过上限自动丢弃最旧行。"""
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.text.configure(state=tk.NORMAL)
        self.text.insert(tk.END, f"{ts}  ", "ts")
        if color:
            # 既接受语义名（ok/warn/error，已建 tag），也接受十六进制色值（动态建 tag）
            if color not in self.text.tag_names():
                self.text.tag_configure(color, foreground=color)
            tag = color
        else:
            tag = "text"
        self.text.insert(tk.END, message + "\n", tag)
        lines = int(self.text.index("end-1c").split(".")[0])
        if lines > MAX_LINES:
            self.text.delete("1.0", f"{lines - MAX_LINES}.0")
        self.text.see(tk.END)
        self.text.configure(state=tk.DISABLED)

    def clear(self) -> None:
        self.text.configure(state=tk.NORMAL)
        self.text.delete("1.0", tk.END)
        self.text.configure(state=tk.DISABLED)

    def apply_theme(self) -> None:
        c = self._theme.colors
        self.text.configure(background=c["panel"], foreground=c["text_dim"])
        self._configure_tags()
