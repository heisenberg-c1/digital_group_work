"""处理链列表：每行一个阶段的「启用勾选 + 点击查看输出 + 展开参数调节」。

参数控件由 ``Stage.params_spec`` 自动生成：NumParam → 滑杆 + 数值框，
EnumParam → 只读下拉框。改动即时写回 ``stage.params``（Runner 支持热更新），
下次运行即生效。
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from docscan.pipeline.stages import EnumParam, NumParam, Stage
from docscan.ui.theme import SP_1, SP_2, SP_3, Theme

# 序号徽标（1~8 个阶段足够）
_CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩"

_DOT_OFF = "dot_off"
_DOT_ON = "dot_on"


class StageList(ttk.Frame):
    """左侧处理链面板。

    回调：
        on_toggle(name, enabled) —— 勾选/取消某阶段
        on_select(name) —— 点击阶段名，请求查看该阶段输出
        on_param_change(name, key, value) —— 参数被用户修改（用于日志）
        on_params_reset(name) —— 该阶段参数恢复默认
    """

    def __init__(
        self,
        master,
        on_toggle,
        on_select,
        theme: Theme,
        on_param_change=None,
        on_params_reset=None,
        **kwargs,
    ) -> None:
        super().__init__(master, style="Panel.TFrame", **kwargs)
        self._on_toggle = on_toggle
        self._on_select = on_select
        self._on_param_change = on_param_change
        self._on_params_reset = on_params_reset
        self._theme = theme
        self._vars: dict[str, tk.BooleanVar] = {}
        self._rows: dict[str, ttk.Label] = {}
        self._dots: dict[str, tk.Canvas] = {}
        self._has_output: dict[str, bool] = {}
        self._stages: list[Stage] = []
        self._selected: str | None = None

        head = ttk.Label(self, text="处理链", style="PanelHead.TLabel")
        head.pack(anchor="w", padx=SP_3, pady=(SP_3, SP_1))
        ttk.Label(self, text="勾选启用 · 点击名称查看输出与参数", style="PanelDim.TLabel").pack(
            anchor="w", padx=SP_3, pady=(0, SP_2)
        )
        self._body = ttk.Frame(self, style="Panel.TFrame")
        self._body.pack(fill=tk.BOTH, expand=True, padx=SP_2, pady=(0, SP_3))

    # ---- 构建 ----

    def rebuild(self, stages: list[Stage]) -> None:
        """按当前阶段列表重建行；保留选中态与「已运行」标记。"""
        self._stages = stages
        for child in self._body.winfo_children():
            child.destroy()
        self._vars.clear()
        self._rows.clear()
        self._dots.clear()

        for i, stage in enumerate(stages):
            self._build_stage_row(i, stage)
            if stage.name == self._selected:
                self._build_param_area(stage)

        self._paint_selection()

    def _build_stage_row(self, index: int, stage: Stage) -> None:
        row = ttk.Frame(self._body, style="Panel.TFrame")
        row.pack(fill=tk.X, pady=1)

        dot = tk.Canvas(
            row, width=9, height=9, highlightthickness=0, bg=self._theme.colors["panel"]
        )
        dot.pack(side=tk.LEFT, padx=(SP_1, 0), pady=(SP_2, 0))
        self._dots[stage.name] = dot

        var = tk.BooleanVar(value=stage.enabled)
        var.trace_add(
            "write", lambda *_a, n=stage.name, v=var: self._on_toggle(n, bool(v.get()))
        )
        ttk.Checkbutton(row, variable=var, style="Panel.TCheckbutton").pack(side=tk.LEFT)

        label = ttk.Label(
            row,
            text=f"{_CIRCLED[index % len(_CIRCLED)]} {stage.label}",
            style="Panel.TLabel",
            cursor="hand2",
            padding=(SP_1, SP_1),
        )
        label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        label.bind("<Button-1>", lambda _e, n=stage.name: self.select(n))
        self._vars[stage.name] = var
        self._rows[stage.name] = label
        self._paint_dot(stage.name)

    def _build_param_area(self, stage: Stage) -> None:
        """在选中阶段下方展开参数编辑区（无参数的阶段不展开）。"""
        if not stage.params_spec:
            return
        c = self._theme.colors
        area = ttk.Frame(self._body, style="Panel.TFrame")
        area.pack(fill=tk.X, padx=(SP_3, SP_2), pady=(SP_1, SP_2))

        bar = ttk.Frame(area, style="Panel.TFrame")
        bar.pack(fill=tk.X, pady=(0, SP_1))
        ttk.Label(bar, text="参数", style="PanelDim.TLabel").pack(side=tk.LEFT)
        ttk.Button(
            bar, text="恢复默认", padding=(SP_2, 0),
            command=lambda n=stage.name: self._reset_params(n),
        ).pack(side=tk.RIGHT)

        # 左侧细竖线示意「归属当前阶段」
        tk.Canvas(area, width=2, highlightthickness=0, bg=c["border"]).pack(
            side=tk.LEFT, fill=tk.Y, padx=(0, SP_2)
        )
        grid = ttk.Frame(area, style="Panel.TFrame")
        grid.pack(fill=tk.X, expand=True)

        for r, spec in enumerate(stage.params_spec):
            ttk.Label(grid, text=spec.label, style="PanelDim.TLabel", anchor="w").grid(
                row=r, column=0, sticky="w", pady=(2, 2), columnspan=2
            )
            if isinstance(spec, NumParam):
                self._build_num_control(grid, r, stage, spec)
            elif isinstance(spec, EnumParam):
                self._build_enum_control(grid, r, stage, spec)

    def _build_num_control(
        self, parent: ttk.Frame, row: int, stage: Stage, spec: NumParam
    ) -> None:
        _decimals_of(spec.step)
        var = tk.StringVar(value=_fmt(stage.params.get(spec.key, spec.default), spec))

        def commit(source: str) -> None:
            try:
                raw = float(var.get())
            except ValueError:
                var.set(_fmt(stage.params.get(spec.key, spec.default), spec))
                return
            value = min(max(raw, spec.minimum), spec.maximum)
            if spec.integer:
                value = float(round(value))
            var.set(_fmt(value, spec))
            stage.params[spec.key] = int(value) if spec.integer else value
            if source == "user" and self._on_param_change:
                self._on_param_change(stage.name, spec.key, stage.params[spec.key])

        scale = ttk.Scale(
            parent, from_=spec.minimum, to=spec.maximum, orient=tk.HORIZONTAL,
        )
        scale.set(float(stage.params.get(spec.key, spec.default)))
        entry = ttk.Entry(
            parent, textvariable=var, width=8, justify=tk.CENTER,
            font=self._theme.font("xs"),
        )

        def on_drag(_v: str) -> None:
            # 拖动即时写参数，但不打日志（松手/回车才算一次「用户修改」）
            value = float(scale.get())
            if spec.integer:
                value = float(round(value))
            stage.params[spec.key] = int(value) if spec.integer else value
            var.set(_fmt(value, spec))

        scale.configure(command=on_drag)
        scale.bind("<ButtonRelease-1>", lambda _e: commit("user"))
        entry.bind("<Return>", lambda _e: commit("user"))
        entry.bind("<FocusOut>", lambda _e: commit("user"))

        scale.grid(row=row, column=0, sticky="ew", padx=(0, SP_1))
        entry.grid(row=row, column=1, sticky="e")
        parent.columnconfigure(0, weight=1)

    def _build_enum_control(
        self, parent: ttk.Frame, row: int, stage: Stage, spec: EnumParam
    ) -> None:
        value = str(stage.params.get(spec.key, spec.default))
        labels = [label for _v, label in spec.choices]
        box = ttk.Combobox(
            parent, values=labels, state="readonly", width=12,
            font=self._theme.font("xs"), justify=tk.CENTER,
        )
        box.set(spec.label_of(value))

        def on_pick(_e=None) -> None:
            chosen = dict(spec.choices).get(box.get(), spec.default)
            stage.params[spec.key] = chosen
            if self._on_param_change:
                self._on_param_change(stage.name, spec.key, chosen)

        box.bind("<<ComboboxSelected>>", on_pick)
        box.grid(row=row, column=0, sticky="ew", columnspan=2)

    def _reset_params(self, name: str) -> None:
        for stage in self._stages:
            if stage.name == name:
                stage.reset_params()
                break
        if self._on_params_reset:
            self._on_params_reset(name)
        self.rebuild(self._stages)

    # ---- 状态 ----

    def select(self, name: str) -> None:
        """选中并展开某阶段，同时通知外部查看其输出。"""
        self._selected = name
        self.rebuild(self._stages)
        self._on_select(name)

    @property
    def selected(self) -> str | None:
        return self._selected

    def set_stage_image_available(self, name: str, available: bool) -> None:
        """该阶段已产出可查看的输出（行名加粗 + 状态点亮起）。"""
        self._has_output[name] = available
        label = self._rows.get(name)
        if label is not None:
            label.configure(font=self._theme.font("sm", bold=available))
        self._paint_dot(name)

    def reset_outputs(self) -> None:
        """换图后清空「已运行」状态。"""
        self._has_output.clear()
        for name in self._rows:
            self.set_stage_image_available(name, False)

    # ---- 绘制辅助 ----

    def _paint_selection(self) -> None:
        c = self._theme.colors
        for n, label in self._rows.items():
            label.configure(background=c["accent_soft"] if n == self._selected else c["panel"])

    def _paint_dot(self, name: str) -> None:
        dot = self._dots.get(name)
        if dot is None:
            return
        c = self._theme.colors
        color = c["ok"] if self._has_output.get(name) else c["border"]
        dot.delete("all")
        dot.create_oval(1, 1, 8, 8, fill=color, outline=color)

    def apply_theme(self) -> None:
        """明暗切换后重绘（标签底色、状态点）。"""
        self.rebuild(self._stages)


def _decimals_of(step: float) -> int:
    text = f"{step:.10f}".rstrip("0")
    if "." in text:
        return len(text.split(".")[1])
    return 0


def _fmt(value: float, spec: NumParam) -> str:
    if spec.integer:
        return str(round(value))
    return f"{value:.{_decimals_of(spec.step)}f}"
