"""UI 冒烟测试：只验证「能构造、能接线」，不做像素级断言。

回归背景：``save_dialog.py`` 漏导入 ``SP_1``，导致点击「保存结果」直接
``NameError``，PNG/JPEG/PDF 三种导出全部不可达；而当时 ``ui/`` 覆盖率是 0%，
136 个测试没有一个会碰 UI 构造路径。这里补上最小的构造级断言。
"""

from __future__ import annotations

import pytest

tk = pytest.importorskip("tkinter")


@pytest.fixture(scope="session")
def _tk_root():
    """会话级 Tk 根窗口。

    每个用例各建一个 ``Tk()`` 再销毁，在 Windows 上会让后续用例偶发
    ``TclError``（"application has been destroyed"），表现为「有的用例跑、
    有的被跳过」。共用一个根窗口 + 每例清理子控件更稳。
    """
    try:
        window = tk.Tk()
    except tk.TclError:  # pragma: no cover - 无显示环境
        pytest.skip("无图形环境，跳过 UI 冒烟测试")
    window.withdraw()
    yield window
    window.destroy()


@pytest.fixture
def root(_tk_root):
    """每个用例开始前清掉上一个用例残留的 Toplevel / 子控件。"""
    for child in list(_tk_root.winfo_children()):
        child.destroy()
    return _tk_root


@pytest.fixture
def theme(root):
    from docscan.ui.theme import Theme

    t = Theme(dark=True)
    t.scale = 1.0
    t.apply(root)
    return t


class _StubView:
    """控制器需要的最小视图接口。"""

    def __init__(self) -> None:
        self.logs: list[str] = []
        self.status = ""

    def log(self, message: str, color=None) -> None:
        self.logs.append(message)

    def set_status(self, text: str) -> None:
        self.status = text


class TestSaveDialog:
    def test_dialog_constructs(self, root, theme, monkeypatch):
        from docscan.ui.widgets.save_dialog import SaveDialog

        monkeypatch.setattr(SaveDialog, "wait_visibility", lambda self: None)
        monkeypatch.setattr(SaveDialog, "grab_set", lambda self: None)
        dlg = SaveDialog(root, theme, suggested_name="demo")
        assert dlg.result is None
        dlg.destroy()

    def test_confirm_builds_result(self, root, theme, monkeypatch):
        from docscan.ui.widgets.save_dialog import SaveDialog

        monkeypatch.setattr(SaveDialog, "wait_visibility", lambda self: None)
        monkeypatch.setattr(SaveDialog, "grab_set", lambda self: None)
        dlg = SaveDialog(root, theme, suggested_name="demo")
        dlg._confirm()
        assert dlg.result is not None
        path, fmt, mode = dlg.result
        assert path.endswith(".png") and fmt == "png" and mode == "bw"

    def test_confirm_appends_missing_suffix(self, root, theme, monkeypatch):
        from docscan.ui.widgets.save_dialog import SaveDialog

        monkeypatch.setattr(SaveDialog, "wait_visibility", lambda self: None)
        monkeypatch.setattr(SaveDialog, "grab_set", lambda self: None)
        dlg = SaveDialog(root, theme, suggested_name="report")
        dlg._path_var.set("out.jpg")
        dlg._confirm()
        assert dlg.result[0] == "out.jpg"  # 手填扩展名优先，不追加 .png
        assert dlg.result[1] == "jpeg"


class TestMainWindow:
    def test_window_builds_with_all_panels(self, root, theme):
        from docscan.ui.controllers import AppController
        from docscan.ui.main_window import MainWindow

        controller = AppController(view=_StubView())
        view = MainWindow(root, controller, theme)
        assert view.stage_list is not None
        assert view.metrics is not None
        assert view.canvas is not None
        # 处理链面板应渲染出全部阶段
        assert len(controller.stages) == 9


class TestPreset:
    def test_preset_roundtrip_through_controller(self, tmp_path):
        from docscan.ui.controllers import AppController

        view = _StubView()
        controller = AppController(view=view)
        controller.stages[0].enabled = False  # warp
        for stage in controller.stages:
            if stage.name == "binarize":
                stage.params["block_size"] = 41

        preset = tmp_path / "preset.json"
        controller.save_preset(preset)
        assert preset.exists()
        assert any("预设已保存" in m for m in view.logs)

        fresh = AppController(view=_StubView())
        fresh.load_preset(preset)
        by_name = {s.name: s for s in fresh.stages}
        assert by_name["warp"].enabled is False
        assert by_name["binarize"].params["block_size"] == 41



def test_parameter_labels_have_separate_rows(root, theme):
    from tkinter import ttk

    from docscan.pipeline.stages import build_default_stages
    from docscan.ui.widgets.stage_list import StageList

    panel = StageList(root, lambda *_: None, lambda *_: None, theme)
    panel.rebuild(build_default_stages())
    panel.select("enhance")

    def check(widget):
        children = widget.winfo_children()
        labels = [c for c in children if isinstance(c, ttk.Label) and c.winfo_manager() == "grid"]
        controls = [c for c in children if isinstance(c, (ttk.Scale, ttk.Entry, ttk.Combobox))]
        for label in labels:
            for control in controls:
                assert label.grid_info()["row"] != control.grid_info()["row"]
        for child in children:
            check(child)

    check(panel)



def test_parameter_dropdown_stores_selected_value(root, theme):
    from tkinter import ttk

    from docscan.pipeline.stages import build_default_stages
    from docscan.ui.widgets.stage_list import StageList

    stages = build_default_stages()
    stage = next(s for s in stages if s.name == "enhance")
    panel = StageList(root, lambda *_: None, lambda *_: None, theme)
    panel.rebuild(stages)
    panel.select("enhance")

    def find_box(widget):
        for child in widget.winfo_children():
            if isinstance(child, ttk.Combobox):
                return child
            found = find_box(child)
            if found is not None:
                return found
        return None

    box = find_box(panel)
    assert box is not None
    box.current(1)
    box.event_generate("<<ComboboxSelected>>")
    spec = next(s for s in stage.params_spec if s.key == "method")
    assert stage.params["method"] == spec.choices[1][0]
