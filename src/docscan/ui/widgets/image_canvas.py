"""图像画布：自适应缩放、滚轮缩放（带滚动条）、像素坐标回显。

缩放两态 + 自由档：``fit`` 适应窗口、``1.0`` 原始像素，滚轮在固定档位间
步进（0.25×~4×）。超出窗口时自动出现滚动条；颜色一律取自 theme token。
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import Image as PILImage
from PIL import ImageTk

from docscan.io.exporter import bgr_to_pil
from docscan.ui.theme import Theme

ZOOM_STEPS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)


class ImageCanvas(ttk.Frame):
    """居中显示一张图像。

    Args:
        theme: 主题（背景、空状态文字色取自 token）。
        on_pixel_move: 鼠标移动回调 ``(x, y, gray_or_None)``，x/y 为原图像素坐标。
        on_zoom_changed: 缩放变化回调 ``(scale: float, mode: str)``，
            mode 为 ``"fit"`` 或 ``"manual"``。
    """

    def __init__(
        self, master, theme: Theme, on_pixel_move=None, on_zoom_changed=None, **kwargs
    ) -> None:
        super().__init__(master, **kwargs)
        self._theme = theme
        self._pil: PILImage.Image | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._scale = 1.0
        self._zoom: float | str = "fit"
        self._offset = (0, 0)
        self._on_pixel_move = on_pixel_move
        self._on_zoom_changed = on_zoom_changed

        self.canvas = tk.Canvas(
            self, highlightthickness=0, bg=theme.colors["canvas"], cursor="crosshair"
        )
        self._xbar = ttk.Scrollbar(self, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self._ybar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(
            xscrollcommand=self._xbar.set, yscrollcommand=self._ybar.set
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self._ybar.grid(row=0, column=1, sticky="ns")
        self._xbar.grid(row=1, column=0, sticky="ew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        # 图像放得下时收起滚动条（grid_remove 保留布局参数，恢复时无需重设）
        self._xbar.grid_remove()
        self._ybar.grid_remove()

        self.canvas.bind("<Configure>", lambda _e: self._redraw())
        self.canvas.bind("<Motion>", self._handle_motion)
        self.canvas.bind("<Leave>", lambda _e: self._handle_motion(None))
        self.canvas.bind("<MouseWheel>", self._on_wheel)  # Windows

    # ---- 显示 ----

    def show(self, image: np.ndarray, *, keep_zoom: bool = False) -> None:
        """显示 BGR 或灰度 ndarray（uint8）；默认回到「适应窗口」。"""
        self._pil = bgr_to_pil(image)
        if not keep_zoom:
            self._zoom = "fit"
        self._redraw()

    def clear(self) -> None:
        self._pil = None
        self._zoom = "fit"
        self._redraw()

    def set_zoom(self, mode: str | float) -> None:
        """切换缩放：``"fit"`` 适应窗口、``"1:1"`` 原始像素、正数为自定义倍率。"""
        if mode == "fit":
            self._zoom = "fit"
        elif mode == "1:1":
            self._zoom = 1.0
        else:
            self._zoom = min(max(float(mode), 0.1), 6.0)
        self._redraw()

    @property
    def zoom_mode(self) -> str | float:
        return self._zoom

    @property
    def zoom_percent(self) -> int:
        return max(1, round(self._scale * 100))

    # ---- 绘制 ----

    def _redraw(self) -> None:
        c = self._theme.colors
        self.canvas.delete("all")
        self.canvas.configure(bg=c["canvas"])
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)

        if self._pil is None:
            self.canvas.configure(scrollregion=(0, 0, cw, ch))
            self._toggle_bars(False, False)
            self._offset = (0, 0)
            self._scale = 1.0
            cx, cy = cw // 2, max(ch // 2 - self._theme.scale * 10, 10)
            self.canvas.create_text(
                cx, cy, text="打开一张图像开始", fill=c["text_dim"],
                font=self._theme.font("lg"),
            )
            self.canvas.create_text(
                cx, cy + self._theme.scale * 34,
                text="Ctrl+O 打开 JPG / PNG / PDF，或在左侧勾选处理阶段",
                fill=c["border"], font=self._theme.font("sm"),
            )
            self._notify_zoom()
            return

        iw, ih = self._pil.size
        if self._zoom == "fit":
            self._scale = min(cw / iw, ch / ih, 1.0)
        else:
            self._scale = float(self._zoom)
        dw = max(1, round(iw * self._scale))
        dh = max(1, round(ih * self._scale))
        if abs(self._scale - 1.0) < 1e-3:
            disp = self._pil
        elif self._scale < 1.0:
            disp = self._pil.resize((dw, dh), PILImage.Resampling.LANCZOS)
        else:
            disp = self._pil.resize((dw, dh), PILImage.Resampling.BILINEAR)
        self._photo = ImageTk.PhotoImage(disp)

        # 画布内容小于窗口时居中（scrollregion 扩到窗口大小避免死滚动）
        total_w, total_h = max(cw, dw), max(ch, dh)
        ox = (total_w - dw) // 2
        oy = (total_h - dh) // 2
        self._offset = (ox, oy)
        self.canvas.configure(scrollregion=(0, 0, total_w, total_h))
        self.canvas.create_image(ox, oy, image=self._photo, anchor=tk.NW)
        if dw < cw:
            self.canvas.xview_moveto(0.0)
        if dh < ch:
            self.canvas.yview_moveto(0.0)
        self._toggle_bars(dw > cw, dh > ch)
        self._notify_zoom()

    def _toggle_bars(self, need_x: bool, need_y: bool) -> None:
        if need_x:
            self._xbar.grid()
        else:
            self._xbar.grid_remove()
        if need_y:
            self._ybar.grid()
        else:
            self._ybar.grid_remove()

    def _notify_zoom(self) -> None:
        if self._on_zoom_changed is not None:
            mode = "fit" if self._zoom == "fit" else "manual"
            self._on_zoom_changed(self._scale, mode)

    # ---- 交互 ----

    def _on_wheel(self, event) -> None:
        if self._pil is None:
            return
        direction = 1 if event.delta > 0 else -1
        if self._zoom == "fit":
            if direction <= 0:
                return  # 已经是全图，不再缩小
            new_zoom: str | float = 1.0
        else:
            idx = min(
                range(len(ZOOM_STEPS)),
                key=lambda i: abs(ZOOM_STEPS[i] - float(self._zoom)),
            )
            nxt = idx + direction
            if nxt < 0:
                new_zoom = "fit"
            elif nxt >= len(ZOOM_STEPS):
                return
            else:
                new_zoom = ZOOM_STEPS[nxt]
        self.set_zoom(new_zoom)

    def _handle_motion(self, event) -> None:
        if self._on_pixel_move is None:
            return
        if event is None or self._pil is None or self._scale <= 0:
            self._on_pixel_move(None, None, None)
            return
        iw, ih = self._pil.size
        sx = self.canvas.canvasx(event.x)
        sy = self.canvas.canvasy(event.y)
        ox, oy = self._offset
        x = int((sx - ox) / self._scale)
        y = int((sy - oy) / self._scale)
        if 0 <= x < iw and 0 <= y < ih:
            value = self._pil.getpixel((x, y))
            gray = value if isinstance(value, int) else int(sum(value[:3]) / 3)
            self._on_pixel_move(x, y, gray)
        else:
            self._on_pixel_move(None, None, None)

    # ---- 主题 ----

    def apply_theme(self) -> None:
        self._redraw()
