"""GUI 启动入口。"""

from __future__ import annotations


def run_gui(*, dark: bool = False) -> int:
    """创建主窗口并进入事件循环。"""
    import tkinter as tk

    from docscan.ui.controllers import AppController
    from docscan.ui.main_window import MainWindow
    from docscan.ui.theme import Theme

    root = tk.Tk()
    theme = Theme(dark=dark)
    theme.apply(root)

    # 控制器与视图互相引用：先建控制器，再建视图，最后回填引用并启动轮询
    controller = AppController(view=None)
    view = MainWindow(root, controller, theme)
    controller.view = view
    controller.start()

    root.mainloop()
    return 0
