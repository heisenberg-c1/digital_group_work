"""主窗口：三栏布局（处理链 / 画布 / 指标）+ 工具栏 + 日志 + 状态栏。"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

from docscan.core.types import Image, StageResult, to_uint8
from docscan.io.exporter import save_image
from docscan.pipeline.runner import RunResult
from docscan.ui.theme import SP_1, SP_2, SP_3, Theme
from docscan.ui.widgets.image_canvas import ImageCanvas
from docscan.ui.widgets.log_view import LogView
from docscan.ui.widgets.metrics_panel import MetricsPanel
from docscan.ui.widgets.stage_list import StageList

FILE_PATTERNS = [
    ("图像文件", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"),
    ("PDF 文件", "*.pdf"),
    ("所有文件", "*.*"),
]


class MainWindow(ttk.Frame):
    """应用主窗口，全部 UI 更新方法集中在这里供 controller 调用。"""

    def __init__(self, root: tk.Tk, controller, theme: Theme) -> None:
        super().__init__(root)
        self.root = root
        self.controller = controller
        self.theme = theme
        root.title("docscan · 手机文档扫描与可读性增强系统")
        root.geometry("1180x760")
        root.minsize(900, 620)
        self.pack(fill=tk.BOTH, expand=True)

        self._source: Image | None = None
        self._processed: Image | None = None

        self._build_toolbar()
        self._build_main_area()
        self._build_statusbar()
        self._build_shortcuts()

        self.stage_list.rebuild(controller.stages)

    # ---- 构建 ----

    def _build_toolbar(self) -> None:
        bar = ttk.Frame(self, padding=(SP_3, SP_2))
        bar.pack(fill=tk.X)
        ttk.Button(bar, text="打开图像", command=self._open_dialog).pack(side=tk.LEFT)
        self.btn_run = ttk.Button(bar, text="运行处理链", style="Accent.TButton", command=self._run)
        self.btn_run.pack(side=tk.LEFT, padx=(SP_2, 0))
        ttk.Button(bar, text="保存结果", command=self._save_result).pack(
            side=tk.LEFT, padx=(SP_2, 0)
        )
        ttk.Button(bar, text="频谱选点", command=self._open_spectrum).pack(
            side=tk.LEFT, padx=(SP_2, 0)
        )

        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=SP_3)
        self.zoom_var = tk.StringVar(value="fit")
        ttk.Radiobutton(
            bar, text="适应窗口", value="fit", variable=self.zoom_var,
            command=lambda: self.canvas.set_zoom("fit"),
        ).pack(side=tk.LEFT)
        ttk.Radiobutton(
            bar, text="1:1", value="1:1", variable=self.zoom_var,
            command=lambda: self.canvas.set_zoom("1:1"),
        ).pack(side=tk.LEFT, padx=(SP_2, 0))

    def _build_main_area(self) -> None:
        panes = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        panes.pack(fill=tk.BOTH, expand=True, padx=SP_3)

        # 左：处理链
        left = ttk.Frame(panes, width=200)
        panes.add(left, weight=0)
        self.stage_list = StageList(
            left,
            on_toggle=self.controller.set_stage_enabled,
            on_select=self.controller.select_stage,
            theme=self.theme,
        )
        self.stage_list.pack(fill=tk.BOTH, expand=True)

        # 中：画布 + 日志
        center = ttk.Frame(panes)
        panes.add(center, weight=1)
        self.canvas = ImageCanvas(center, on_pixel_move=self._show_pixel)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.log_view = LogView(center, self.theme, height=6)
        self.log_view.pack(fill=tk.X, pady=(SP_2, 0))

        # 右：指标
        right = ttk.Frame(panes, width=220)
        panes.add(right, weight=0)
        self.metrics = MetricsPanel(right, self.theme)
        self.metrics.pack(fill=tk.BOTH, expand=True)

    def _build_statusbar(self) -> None:
        bar = ttk.Frame(self, padding=(SP_3, SP_1))
        bar.pack(fill=tk.X, side=tk.BOTTOM)
        self.status = ttk.Label(bar, text="就绪", style="Status.TLabel")
        self.status.pack(side=tk.LEFT)
        self.pixel_info = ttk.Label(bar, text="", style="Status.TLabel")
        self.pixel_info.pack(side=tk.RIGHT)

    def _build_shortcuts(self) -> None:
        self.root.bind("<Control-o>", lambda _e: self._open_dialog())
        self.root.bind("<Return>", lambda _e: self._run())
        self.root.bind("<Control-s>", lambda _e: self._save_result())

    # ---- 用户动作 ----

    def _open_spectrum(self) -> None:
        """弹出频谱视图，选点结果写入去网纹阶段。"""
        if self._source is None:
            self.show_error("提示", "请先打开一张图像")
            return
        from docscan.ui.widgets.spectrum_view import SpectrumView

        win = tk.Toplevel(self)
        win.title("幅度谱 · 陷波选点")
        win.geometry("760x640")
        view = SpectrumView(win, self.theme, on_points_changed=self.controller.apply_manual_peaks)
        view.pack(fill=tk.BOTH, expand=True, padx=SP_3, pady=SP_3)
        view.show(self._source)

    def stage_list_rebuild_if_needed(self) -> None:
        """阶段启用状态在代码侧变化后，同步勾选框显示。"""
        selected = self.stage_list.selected
        self.stage_list.rebuild(self.controller.stages)
        if selected:
            self.stage_list.select(selected)

    def _open_dialog(self) -> None:
        path = filedialog.askopenfilename(title="打开图像", filetypes=FILE_PATTERNS)
        if path:
            self.controller.open_image(path)

    def _run(self) -> None:
        self.controller.run()

    def _save_result(self) -> None:
        img = self.canvas_current()
        if img is None:
            self.show_error("提示", "没有可保存的结果，请先运行处理链")
            return
        path = filedialog.asksaveasfilename(
            title="保存结果", defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")],
        )
        if path:
            out = save_image(img, path)
            self.log(f"结果已保存到 {out}")
            self.set_status(f"已保存：{Path(out).name}")

    def canvas_current(self) -> Image | None:
        """当前画布上显示的图像（保存用）。"""
        if self._processed is not None:
            return self._processed
        return self._source

    # ---- 供 controller 调用的更新方法 ----

    def show_source_image(self, img: Image) -> None:
        self._source = img
        self._processed = None
        self.canvas.show(to_uint8(img))
        self.metrics.show_overall(None)

    def show_processed_image(self, img: Image) -> None:
        self._processed = img
        self.canvas.show(to_uint8(img))

    def on_stage_done(self, name: str, sr: StageResult) -> None:
        self.stage_list.set_stage_image_available(name, True)
        self.log(f"[{name}] {sr.notes}（{sr.elapsed_ms:.0f} ms）")

    def on_run_finished(self, result: RunResult) -> None:
        self.show_processed_image(result.final)
        if result.order and self.stage_list.selected in (None, ""):
            self.stage_list.select(result.order[-1])
        self.metrics.show_overall(result.total_ms)
        for w in result.warnings:
            self.log(w, color=self.theme.colors["warn"])
        self.log(f"处理完成，总耗时 {result.total_ms:.0f} ms", color=self.theme.colors["ok"])
        self.set_status(f"完成：{result.total_ms:.0f} ms")
        self.set_busy(False)

    # ---- 杂项 ----

    def log(self, message: str, color: str | None = None) -> None:
        self.log_view.append(message, color=color)

    def set_status(self, text: str) -> None:
        self.status.configure(text=text)

    def set_busy(self, busy: bool) -> None:
        state = tk.DISABLED if busy else tk.NORMAL
        self.btn_run.configure(state=state)
        self.set_status("处理中…" if busy else "就绪")

    def show_error(self, title: str, message: str) -> None:
        from tkinter import messagebox

        messagebox.showerror(title, message, parent=self)
        self.log(f"[错误] {message}", color=self.theme.colors["error"])

    def _show_pixel(self, x, y, gray) -> None:
        if x is None:
            self.pixel_info.configure(text="")
        else:
            self.pixel_info.configure(text=f"({x}, {y})  灰度 {gray}")

    def after(self, ms: int, func) -> None:
        self.root.after(ms, func)
