"""线程桥接：worker 线程跑 Runner，队列回传，UI 线程轮询消费。

规则（架构 §4）：worker 绝不直接调用 tkinter，一切 UI 更新
都发生在 ``root.after`` 的轮询回调里。
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from pathlib import Path

from docscan.core.types import Image, StageResult
from docscan.io.loader import load_any
from docscan.pipeline.runner import Runner, RunResult
from docscan.pipeline.stages import Stage, build_default_stages

POLL_MS = 80  # 轮询周期，UI 无感知延迟且不占 CPU


@dataclass
class QueueMessage:
    """worker → UI 的消息。"""

    kind: str  # "stage" | "done" | "error"
    stage_name: str = ""
    stage_result: StageResult | None = None
    result: RunResult | None = None
    error: str = ""


class AppController:
    """连接 MainWindow 与 Runner。"""

    def __init__(self, view) -> None:
        self.view = view
        self.queue: queue.Queue[QueueMessage] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.stages: list[Stage] = build_default_stages()
        self.runner = Runner(self.stages)
        self.image: Image | None = None
        self.last_result: RunResult | None = None

    def start(self) -> None:
        """视图就绪后启动队列轮询。"""
        self._poll()

    # ---- 输入 ----

    def open_image(self, path: str | Path) -> None:
        """打开图像或 PDF（同步读取，文件一般 < 100ms）。"""
        try:
            pages = load_any(path)
        except (FileNotFoundError, ValueError) as exc:
            self.view.show_error("打开失败", str(exc))
            return
        self.image = pages[0]
        if len(pages) > 1:
            self.view.log(f"PDF 共 {len(pages)} 页，当前显示第 1 页（批处理请用 CLI）")
        self.view.show_source_image(self.image)
        self.view.log(f"已打开 {Path(path).name}（{self.image.shape[1]}×{self.image.shape[0]}）")
        self.view.set_status(f"已打开：{Path(path).name}")

    # ---- 运行 ----

    def run(self) -> None:
        """在 worker 线程执行处理链。"""
        if self.image is None:
            self.view.show_error("提示", "请先打开一张图像")
            return
        if self.worker is not None and self.worker.is_alive():
            self.view.log("上一次处理尚未结束，已忽略本次请求")
            return

        image = self.image
        q = self.queue
        self.view.set_busy(True)

        def worker() -> None:
            def on_stage(name: str, sr: StageResult) -> None:
                q.put(QueueMessage(kind="stage", stage_name=name, stage_result=sr))

            try:
                result = self.runner.run(image, progress_cb=on_stage)
                q.put(QueueMessage(kind="done", result=result))
            except Exception as exc:
                q.put(QueueMessage(kind="error", error=str(exc)))

        self.worker = threading.Thread(target=worker, name="docscan-worker", daemon=True)
        self.worker.start()

    def set_stage_enabled(self, name: str, enabled: bool) -> None:
        self.runner.set_enabled(name, enabled)
        stage = self.runner.get_stage(name)
        self.view.log(f"阶段「{stage.label}」{'启用' if enabled else '停用'}")

    def apply_manual_peaks(self, points: list[tuple[int, int]]) -> None:
        """频谱视图选点 → 写入去网纹阶段参数并启用。"""
        stage = self.runner.get_stage("freq")
        stage.params["manual_peaks"] = list(points)
        if points and not stage.enabled:
            self.runner.set_enabled("freq", True)
            self.view.stage_list_rebuild_if_needed()
        self.view.log(f"频谱选点 {len(points)} 个：{points[:6]}")

    def select_stage(self, name: str) -> None:
        """切换查看某阶段的输出。"""
        if self.last_result is None:
            return
        sr = self.last_result.stage_outputs.get(name)
        if sr is None:
            self.view.log(f"阶段 {name} 当前没有输出（未运行或已停用）")
            return
        self.view.show_processed_image(sr.image)
        self.view.metrics.show_stage(
            self.runner.get_stage(name).label, sr.notes, sr.metrics, sr.elapsed_ms
        )

    # ---- 轮询 ----

    def _poll(self) -> None:
        """UI 线程定时消费队列。"""
        try:
            while True:
                msg = self.queue.get_nowait()
                if msg.kind == "stage":
                    self.view.on_stage_done(msg.stage_name, msg.stage_result)
                elif msg.kind == "done":
                    self.last_result = msg.result
                    self.view.on_run_finished(msg.result)
                elif msg.kind == "error":
                    self.view.show_error("处理失败", msg.error)
                    self.view.set_busy(False)
        except queue.Empty:
            pass
        self.view.after(POLL_MS, self._poll)
