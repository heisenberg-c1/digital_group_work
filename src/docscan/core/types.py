"""核心数据类型与数值约定。

统一约定（全项目遵守）::

    Image   算法内部: float32, 取值 [0, 1]
    Image   边界(I/O): uint8, 取值 [0, 255]
    色彩空间: BGR（与 OpenCV 一致）；仅在显示给 GUI 前转 RGB
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Image = np.ndarray


@dataclass(frozen=True)
class Quad:
    """页面的四个角点。

    Attributes:
        points: (4, 2) float32，顺序固定为 左上、右上、右下、左下。
        confidence: 0~1，自动检测的置信度；手动与兜底时为 0.0。
        source: ``"auto"`` / ``"manual"`` / ``"fallback"``。
    """

    points: np.ndarray
    confidence: float = 0.0
    source: str = "auto"

    def __post_init__(self) -> None:
        pts = np.asarray(self.points, dtype=np.float32).reshape(4, 2)
        object.__setattr__(self, "points", pts)

    def as_int(self) -> np.ndarray:
        """转为 int32，供 ``cv2`` 的整数坐标 API 使用。"""
        return np.rint(self.points).astype(np.int32)

    @property
    def is_fallback(self) -> bool:
        return self.source == "fallback"


@dataclass
class StageResult:
    """单个处理阶段的输出。

    Attributes:
        image: 该阶段输出图像。
        notes: 供 UI 日志面板显示的一行决策信息，例如 ``"角点置信度 0.83"``。
        elapsed_ms: 该阶段耗时（毫秒）。
        metrics: 该阶段产出的标量指标。
    """

    image: Image
    notes: str = ""
    elapsed_ms: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)


def to_uint8(img: Image) -> Image:
    """float32[0,1] → uint8[0,255]，四舍五入并裁剪越界值。"""
    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        return arr
    return np.rint(np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)


def to_float(img: Image) -> Image:
    """uint8[0,255] → float32[0,1]。"""
    arr = np.asarray(img)
    if arr.dtype == np.float32:
        return arr
    return arr.astype(np.float32) / 255.0


def is_color(img: Image) -> bool:
    """判断是否为三通道彩色图。"""
    return img.ndim == 3 and img.shape[2] == 3
