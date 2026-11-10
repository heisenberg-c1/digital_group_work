"""保存结果对话框：输出格式（PNG / JPEG / PDF）× 输出模式（彩色 / 灰度 / 黑白）。"""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, ttk

from docscan.ui.theme import SP_1, SP_2, SP_3, Theme

FORMATS: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    # (key, 标签, 默认扩展名, 可用扩展名)
    ("png", "PNG 图像（无损）", ".png", (".png",)),
    ("jpeg", "JPEG 图像（体积小）", ".jpg", (".jpg", ".jpeg")),
    ("pdf", "PDF 文档（打印 / 归档）", ".pdf", (".pdf",)),
)
MODES: tuple[tuple[str, str], ...] = (
    ("color", "彩色（保留当前画面）"),
    ("gray", "灰度"),
    ("bw", "黑白二值（扫描件）"),
)


class SaveDialog(tk.Toplevel):
    """模态保存对话框；``result`` 为 ``(path, fmt, mode)``，取消时为 None。"""

    def __init__(self, master, theme: Theme, suggested_name: str = "docscan") -> None:
        super().__init__(master)
        self._theme = theme
        self.result: tuple[str, str, str] | None = None
        self.title("保存结果")
        self.resizable(False, False)
        self.configure(bg=theme.colors["bg"])

        body = ttk.Frame(self, padding=SP_3)
        body.pack(fill=tk.BOTH, expand=True)

        ttk.Label(body, text="保存到", style="Dim.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, SP_1)
        )
        self._path_var = tk.StringVar(value=suggested_name)
        path_row = ttk.Frame(body)
        path_row.grid(row=1, column=0, sticky="ew", pady=(0, SP_3))
        ttk.Entry(path_row, textvariable=self._path_var, width=36).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        ttk.Button(path_row, text="浏览…", command=self._browse).pack(
            side=tk.LEFT, padx=(SP_2, 0)
        )

        ttk.Label(body, text="格式", style="Dim.TLabel").grid(
            row=2, column=0, sticky="w", pady=(0, SP_1)
        )
        self._fmt_var = tk.StringVar(value=FORMATS[0][1])
        ttk.Combobox(
            body, textvariable=self._fmt_var, state="readonly", values=[f[1] for f in FORMATS]
        ).grid(row=3, column=0, sticky="ew", pady=(0, SP_3))

        ttk.Label(body, text="输出模式", style="Dim.TLabel").grid(
            row=4, column=0, sticky="w", pady=(0, SP_1)
        )
        self._mode_var = tk.StringVar(value=MODES[2][1])
        ttk.Combobox(
            body, textvariable=self._mode_var, state="readonly", values=[m[1] for m in MODES]
        ).grid(row=5, column=0, sticky="ew")

        actions = ttk.Frame(body)
        actions.grid(row=6, column=0, sticky="e", pady=(SP_3, 0))
        ttk.Button(actions, text="取消", command=self.destroy).pack(side=tk.RIGHT)
        ttk.Button(
            actions, text="保存", style="Accent.TButton", command=self._confirm
        ).pack(side=tk.RIGHT, padx=(0, SP_2))

        body.columnconfigure(0, weight=1)
        self.bind("<Return>", lambda _e: self._confirm())
        self.bind("<Escape>", lambda _e: self.destroy())
        self.transient(master)
        self.grab_set()
        self.wait_visibility()
        self.focus_set()

    # ---- 交互 ----

    def _browse(self) -> None:
        types = [("所有支持的格式", "*.png *.jpg *.jpeg *.pdf"), ("所有文件", "*.*")]
        path = filedialog.asksaveasfilename(
            parent=self, title="选择保存位置", filetypes=types
        )
        if path:
            self._path_var.set(path)

    def _confirm(self) -> None:
        fmt = next((f for f in FORMATS if f[1] == self._fmt_var.get()), FORMATS[0])
        mode = next((m for m in MODES if m[1] == self._mode_var.get()), MODES[2])
        path = self._path_var.get().strip().strip('"')
        if not path:
            from tkinter import messagebox

            messagebox.showwarning("保存结果", "请先填写保存路径", parent=self)
            return
        lower = path.lower()
        # 手填的扩展名优先于下拉框：用户写 out.jpg 却选了 PNG 时，
        # 不应该悄悄存成 out.jpg.png
        for candidate in FORMATS:
            if lower.endswith(candidate[3]):
                fmt = candidate
                break
        else:
            path += fmt[2]
        self.result = (path, fmt[0], mode[0])
        self.destroy()
