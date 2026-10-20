"""图像画布：自适应缩放显示、适应窗口 / 1:1 切换、像素坐标回显。"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import Image as PILImage
from PIL import ImageTk

from docscan.io.exporter import bgr_to_pil


class ImageCanvas(ttk.Frame):
    """居中显示一张图像，支持 fit / 1:1 两种缩放。

    Args:
        on_pixel_move: 鼠标移动回调 ``(x, y, gray_or_None)``，
            x/y 为原图像素坐标，value 为该点灰度（彩色图取均值）。
    """

    def __init__(self, master, on_pixel_move=None, **kwargs) -> None:
        super().__init__(master, **kwargs)
        self._pil: PILImage.Image | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._scale = 1.0
        self._zoom_mode = "fit"
        self._on_pixel_move = on_pixel_move

        self.canvas = tk.Canvas(self, highlightthickness=0, bg="#2b2f36")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda _e: self._redraw())
        self.canvas.bind("<Motion>", self._handle_motion)
        self.canvas.bind("<Leave>", lambda _e: self._handle_motion(None))

    # ---- 显示 ----

    def show(self, image: np.ndarray) -> None:
        """显示 BGR 或灰度 ndarray（uint8）。"""
        self._pil = bgr_to_pil(image)
        self._redraw()

    def clear(self) -> None:
        self._pil = None
        self.canvas.delete("all")

    def set_zoom(self, mode: str) -> None:
        """切换缩放模式：``"fit"`` 适应窗口 / ``"1:1"`` 原始像素。"""
        if mode not in ("fit", "1:1"):
            raise ValueError(f"未知缩放模式：{mode}")
        self._zoom_mode = mode
        self._redraw()

    @property
    def zoom_mode(self) -> str:
        return self._zoom_mode

    def _redraw(self) -> None:
        self.canvas.delete("all")
        if self._pil is None:
            w = int(self.canvas.winfo_width() or 400)
            h = int(self.canvas.winfo_height() or 300)
            self.canvas.create_text(
                w // 2, h // 2,
                text="打开一张图像开始",
                fill="#9aa3b2",
                font=("Microsoft YaHei UI", -13),
            )
            return

        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        iw, ih = self._pil.size
        if self._zoom_mode == "fit":
            self._scale = min(cw / iw, ch / ih, 1.0)
        else:
            self._scale = 1.0
        disp_w, disp_h = max(1, round(iw * self._scale)), max(1, round(ih * self._scale))
        resized = (
            self._pil
            if self._scale >= 1.0
            else self._pil.resize((disp_w, disp_h), PILImage.Resampling.BILINEAR)
        )
        self._photo = ImageTk.PhotoImage(resized)
        self.canvas.create_image(cw // 2, ch // 2, image=self._photo, anchor=tk.CENTER)

    # ---- 交互 ----

    def _handle_motion(self, event) -> None:
        if self._on_pixel_move is None:
            return
        if event is None or self._pil is None or self._scale <= 0:
            self._on_pixel_move(None, None, None)
            return
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        iw, ih = self._pil.size
        disp_w, disp_h = iw * self._scale, ih * self._scale
        x = int((event.x - (cw - disp_w) / 2) / self._scale)
        y = int((event.y - (ch - disp_h) / 2) / self._scale)
        if 0 <= x < iw and 0 <= y < ih:
            value = self._pil.getpixel((x, y))
            gray = value if isinstance(value, int) else int(sum(value[:3]) / 3)
            self._on_pixel_move(x, y, gray)
        else:
            self._on_pixel_move(None, None, None)
