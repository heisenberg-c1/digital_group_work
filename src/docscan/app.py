"""GUI 启动入口。"""

from __future__ import annotations

import contextlib
import sys


def _enable_windows_dpi_awareness() -> None:
    """让进程以物理像素渲染（必须在创建 Tk 之前调用）。

    不开 DPI 感知时，Windows 会对整个窗口做位图拉伸，文字与图片全部发虚
    ——这就是高分屏上「界面模糊」的根因。
    """
    if sys.platform != "win32":
        return
    import ctypes

    try:
        # Per-Monitor DPI Aware：拖到不同缩放比的屏幕时按当前显示器重新渲染
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        with contextlib.suppress(Exception):
            ctypes.windll.user32.SetProcessDPIAware()


def run_gui(*, dark: bool = False) -> int:
    """创建主窗口并进入事件循环。"""
    import tkinter as tk

    from docscan.ui.controllers import AppController
    from docscan.ui.main_window import MainWindow
    from docscan.ui.theme import Theme

    _enable_windows_dpi_awareness()

    root = tk.Tk()
    # 感知开启后 Tk 才能拿到真实 DPI；scaling 影响字体点数换算，scale 影响像素字号
    dpi = max(96.0, float(root.winfo_fpixels("1i")))
    root.tk.call("tk", "scaling", dpi / 72.0)

    theme = Theme(dark=dark)
    theme.scale = dpi / 96.0
    theme.apply(root)

    # 控制器与视图互相引用：先建控制器，再建视图，最后回填引用并启动轮询
    controller = AppController(view=None)
    view = MainWindow(root, controller, theme)
    controller.view = view
    controller.start()

    root.mainloop()
    return 0
