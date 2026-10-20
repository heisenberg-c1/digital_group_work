"""日志面板：只读文本，追加式写入。"""

from __future__ import annotations

import tkinter as tk
from tkinter import scrolledtext, ttk

from docscan.ui.theme import Theme

MAX_LINES = 500


class LogView(ttk.Frame):
    """底部日志区。"""

    def __init__(self, master, theme: Theme, height: int = 6, **kwargs) -> None:
        super().__init__(master, **kwargs)
        self._theme = theme
        self.text = scrolledtext.ScrolledText(
            self,
            height=height,
            state=tk.DISABLED,
            font=theme.font("xs"),
            background=theme.colors["panel"],
            foreground=theme.colors["text_dim"],
            relief=tk.FLAT,
            wrap=tk.WORD,
        )
        self.text.pack(fill=tk.BOTH, expand=True)

    def append(self, message: str, *, color: str | None = None) -> None:
        """追加一行日志，超过上限自动丢弃最旧行。"""
        self.text.configure(state=tk.NORMAL)
        if color:
            self.text.tag_configure(color, foreground=color)
            self.text.insert(tk.END, message + "\n", color)
        else:
            self.text.insert(tk.END, message + "\n")
        lines = int(self.text.index("end-1c").split(".")[0])
        if lines > MAX_LINES:
            self.text.delete("1.0", f"{lines - MAX_LINES}.0")
        self.text.see(tk.END)
        self.text.configure(state=tk.DISABLED)

    def clear(self) -> None:
        self.text.configure(state=tk.NORMAL)
        self.text.delete("1.0", tk.END)
        self.text.configure(state=tk.DISABLED)
