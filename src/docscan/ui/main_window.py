"""主窗口：三栏布局（处理链 / 画布 / 指标）+ 工具栏 + 日志 + 状态栏。"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

import numpy as np

from docscan.core import binarize
from docscan.core.types import Image, StageResult, to_uint8
from docscan.io.exporter import export_pdf, save_image
from docscan.pipeline.runner import RunResult
from docscan.ui.theme import SP_1, SP_2, SP_3, Theme
from docscan.ui.widgets.image_canvas import ImageCanvas
from docscan.ui.widgets.log_view import LogView
from docscan.ui.widgets.metrics_panel import MetricsPanel
from docscan.ui.widgets.save_dialog import SaveDialog
from docscan.ui.widgets.stage_list import StageList

FILE_PATTERNS = [
    ("图像 / PDF", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp *.pdf"),
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
        root.geometry("1200x780")
        root.minsize(960, 640)
        self.pack(fill=tk.BOTH, expand=True)

        self._source: Image | None = None
        self._processed: Image | None = None
        self._source_path: str | None = None

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
        self.btn_run = ttk.Button(
            bar, text="运行处理链", style="Accent.TButton", command=self._run
        )
        self.btn_run.pack(side=tk.LEFT, padx=(SP_2, 0))
        ttk.Button(bar, text="保存结果", command=self._save_result).pack(
            side=tk.LEFT, padx=(SP_2, 0)
        )
        ttk.Button(bar, text="频谱选点", command=self._open_spectrum).pack(
            side=tk.LEFT, padx=(SP_2, 0)
        )
        ttk.Button(bar, text="加载预设", command=self._load_preset).pack(
            side=tk.LEFT, padx=(SP_2, 0)
        )
        ttk.Button(bar, text="保存预设", command=self._save_preset).pack(
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
        self.zoom_label = ttk.Label(bar, text="100%", style="Dim.TLabel", width=5, anchor="w")
        self.zoom_label.pack(side=tk.LEFT, padx=(SP_2, 0))

        self.btn_theme = ttk.Button(
            bar, text="明 / 暗", width=8, command=self.toggle_theme
        )
        self.btn_theme.pack(side=tk.RIGHT)

    def _build_main_area(self) -> None:
        panes = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        panes.pack(fill=tk.BOTH, expand=True, padx=SP_3)

        # 左：处理链（含展开参数区）
        left = ttk.Frame(panes, width=250)
        panes.add(left, weight=0)
        self.stage_list = StageList(
            left,
            on_toggle=self.controller.set_stage_enabled,
            on_select=self.controller.select_stage,
            on_param_change=self.controller.set_stage_param,
            on_params_reset=self.controller.reset_stage_params,
            theme=self.theme,
        )
        self.stage_list.pack(fill=tk.BOTH, expand=True)

        # 中：画布 + 日志
        center = ttk.Frame(panes)
        panes.add(center, weight=1)
        self.canvas = ImageCanvas(
            center,
            theme=self.theme,
            on_pixel_move=self._show_pixel,
            on_zoom_changed=self._on_zoom_changed,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.log_view = LogView(center, self.theme, height=5)
        self.log_view.pack(fill=tk.X, pady=(SP_2, 0))

        # 右：指标
        right = ttk.Frame(panes, width=230)
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
        self.root.bind("<Control-d>", lambda _e: self.toggle_theme())
        self.root.bind("<plus>", lambda _e: self._nudge_zoom(1.25))
        self.root.bind("<minus>", lambda _e: self._nudge_zoom(1 / 1.25))

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
        self.stage_list.rebuild(self.controller.stages)

    def _save_preset(self) -> None:
        path = filedialog.asksaveasfilename(
            parent=self,
            title="保存参数预设",
            defaultextension=".json",
            filetypes=[("JSON 预设", "*.json"), ("所有文件", "*.*")],
        )
        if path:
            self.controller.save_preset(path)

    def _load_preset(self) -> None:
        path = filedialog.askopenfilename(
            parent=self, title="加载参数预设", filetypes=[("JSON 预设", "*.json")]
        )
        if path:
            self.controller.load_preset(path)
            self.stage_list.rebuild(self.controller.stages)

    def _open_dialog(self) -> None:
        path = filedialog.askopenfilename(title="打开图像", filetypes=FILE_PATTERNS)
        if path:
            self.controller.open_image(path)

    def _run(self) -> None:
        self.controller.run()

    def _save_result(self) -> None:
        img = self.canvas_current()
        if img is None:
            self.show_error("提示", "没有可保存的结果，请先打开图像或运行处理链")
            return
        name = Path("docscan.png")
        if self._source_path is not None:
            name = Path(self._source_path).with_suffix(".png").name
        dlg = SaveDialog(self, self.theme, suggested_name=str(name))
        self.wait_window(dlg)
        if dlg.result is None:
            return
        path, fmt, mode = dlg.result
        try:
            out = self._export(img, path, fmt, mode)
        except (OSError, ValueError) as exc:
            self.show_error("保存失败", str(exc))
            return
        self.log(f"结果已保存到 {out}（{fmt.upper()} · {mode}）")
        self.set_status(f"已保存：{Path(out).name}")

    def _export(self, img: Image, path: str, fmt: str, mode: str) -> Path:
        """按 格式 × 输出模式 落盘；黑白档必要时补做自适应二值化。"""
        final = to_uint8(np.asarray(img))
        if mode == "gray":
            if final.ndim == 3:
                final = binarize.to_gray(final)
        elif mode == "bw":
            is_binary = final.ndim == 2 and set(np.unique(final)).issubset({0, 255})
            if not is_binary:
                final = binarize.adaptive(binarize.to_gray(final))
                final = binarize.clean_small_components(final)
        if fmt == "pdf":
            return export_pdf([final], path)
        return save_image(final, path)

    def canvas_current(self) -> Image | None:
        """当前画布上显示的图像（保存用）。"""
        if self._processed is not None:
            return self._processed
        return self._source

    def toggle_theme(self) -> None:
        """明暗切换：原生控件由主题引擎刷新，自绘控件逐个重着色。"""
        self.theme.toggle(self.root)
        self.canvas.apply_theme()
        self.log_view.apply_theme()
        self.stage_list.apply_theme()
        self.metrics.apply_theme()
        self.log(f"已切换到{'暗色' if self.theme.dark else '亮色'}主题")

    # ---- 供 controller 调用的更新方法 ----

    def show_source_image(self, img: Image, path: str | None = None) -> None:
        self._source = img
        self._processed = None
        self._source_path = path
        self.stage_list.reset_outputs()
        self.canvas.show(to_uint8(img))
        self.metrics.show_overall(None)

    def show_processed_image(self, img: Image) -> None:
        self._processed = img
        self.canvas.show(to_uint8(img), keep_zoom=True)

    def on_stage_done(self, name: str, sr: StageResult) -> None:
        self.stage_list.set_stage_image_available(name, True)
        self.log(f"[{name}] {sr.notes}（{sr.elapsed_ms:.0f} ms）")

    def on_run_finished(
        self,
        result: RunResult,
        psnr: float | None = None,
        ssim: float | None = None,
        aligned: bool = False,
    ) -> None:
        self.show_processed_image(result.final)
        if result.order and self.stage_list.selected in (None, ""):
            self.stage_list.select(result.order[-1])
        self.metrics.show_overall(result.total_ms, psnr, ssim, aligned)
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
            self.pixel_info.configure(
                text=f"({x}, {y})  灰度 {gray}  ·  {self.canvas.zoom_percent}%"
            )

    def _nudge_zoom(self, factor: float) -> None:
        """键盘 +/- 缩放：在当前比例上乘一个因子并夹到 [0.25, 4.0]。"""
        current = float(self.canvas.zoom_percent) / 100.0
        self.canvas.set_zoom(min(max(current * factor, 0.25), 4.0))

    def _on_zoom_changed(self, scale: float, mode: str) -> None:
        percent = max(1, round(scale * 100))
        self.zoom_label.configure(text=f"{percent}%")
        if mode == "fit":
            self.zoom_var.set("fit")
        elif abs(scale - 1.0) < 1e-6:
            self.zoom_var.set("1:1")
        else:
            self.zoom_var.set("custom")  # 两个单选都不选中

    def after(self, ms: int, func) -> None:
        self.root.after(ms, func)
