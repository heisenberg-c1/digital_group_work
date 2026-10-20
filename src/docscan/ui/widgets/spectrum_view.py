"""频谱交互视图：显示对数幅度谱，点击选点生成陷波坐标。

交互约定：单击 = 添加陷波点（自动补上 DC 对称点）；右键 = 清空。
坐标以相对 DC 的 (du, dv) 表示，u 沿 x、v 沿 y。
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image as PILImage
from PIL import ImageTk

from docscan.core import freq
from docscan.core.types import Image
from docscan.ui.theme import Theme


class SpectrumView(ttk.Frame):
    """幅度谱画布 + 选点列表。"""

    MARKER_R = 6

    def __init__(
        self,
        master,
        theme: Theme,
        on_points_changed=None,
        **kwargs,
    ) -> None:
        super().__init__(master, **kwargs)
        self._theme = theme
        self._on_points_changed = on_points_changed
        self._photo: ImageTk.PhotoImage | None = None
        self._pil: PILImage.Image | None = None
        self._scale = 1.0
        self._shape: tuple[int, int] | None = None
        self.points: list[tuple[int, int]] = []

        toolbar = ttk.Frame(self)
        toolbar.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(
            toolbar,
            text="左键点击添加陷波点（自动补对称点）· 右键清空",
            style="Dim.TLabel",
        ).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="清空选点", command=self.clear).pack(side=tk.RIGHT)

        self.canvas = tk.Canvas(
            self, highlightthickness=1, highlightbackground=theme.colors["border"],
            bg=theme.colors["canvas"], cursor="crosshair",
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Button-3>", lambda _e: self.clear())
        self.canvas.bind("<Configure>", lambda _e: self._redraw())

        self.status = ttk.Label(self, text="尚未生成频谱", style="Dim.TLabel")
        self.status.pack(anchor="w", pady=(4, 0))

    # ---- 数据 ----

    def show(self, image: Image) -> None:
        """计算并显示幅度谱。"""
        spec = freq.magnitude_spectrum(image)
        self._shape = spec.shape
        u8 = np.rint(spec * 255).astype(np.uint8)
        colored = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
        rgb = colored[:, :, ::-1]
        self._pil = PILImage.fromarray(rgb)
        self.points.clear()
        self._emit()
        self.status.configure(text=f"频谱 {spec.shape[1]}×{spec.shape[0]}，点击添加陷波点")

    def clear(self) -> None:
        self.points.clear()
        self._emit()
        self._redraw()

    def get_points(self) -> list[tuple[int, int]]:
        return list(self.points)

    # ---- 绘制 ----

    def _redraw(self) -> None:
        self.canvas.delete("all")
        if self._pil is None:
            return
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        iw, ih = self._pil.size
        self._scale = min(cw / iw, ch / ih, 2.0)
        disp = (
            self._pil
            if self._scale >= 1.0
            else self._pil.resize(
                (max(1, round(iw * self._scale)), max(1, round(ih * self._scale))),
                PILImage.Resampling.NEAREST,  # 频谱用最近邻，保持峰值锐利
            )
        )
        self._photo = ImageTk.PhotoImage(disp)
        self.canvas.create_image(cw // 2, ch // 2, image=self._photo, anchor=tk.CENTER)

        c = self._theme.colors
        cy_disp, cx_disp = ch / 2, cw / 2
        half_h = ih * self._scale / 2
        half_w = iw * self._scale / 2
        for du, dv in self.points:
            x = cx_disp - half_w + (self._shape[1] // 2 + du) * self._scale
            y = cy_disp - half_h + (self._shape[0] // 2 + dv) * self._scale
            for px in (x, 2 * cx_disp - x):  # 本点 + 对称点
                self.canvas.create_oval(
                    px - self.MARKER_R, y - self.MARKER_R,
                    px + self.MARKER_R, y + self.MARKER_R,
                    outline=c["error"], width=2,
                )

    def _on_click(self, event) -> None:
        if self._pil is None or self._shape is None:
            return
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        ih, iw = self._shape
        disp_w, disp_h = iw * self._scale, ih * self._scale
        x0, y0 = (cw - disp_w) / 2, (ch - disp_h) / 2
        sx = int((event.x - x0) / self._scale)
        sy = int((event.y - y0) / self._scale)
        if not (0 <= sx < iw and 0 <= sy < ih):
            return
        du, dv = sx - iw // 2, sy - ih // 2
        if du == 0 and dv == 0:
            self.status.configure(text="DC 点不能作为陷波点")
            return
        self.points.append((du, dv))
        self._emit()
        self._redraw()

    def _emit(self) -> None:
        if self._on_points_changed is not None:
            self._on_points_changed(list(self.points))
