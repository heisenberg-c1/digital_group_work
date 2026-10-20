"""Stage 抽象与默认处理链。

每个 Stage 是一个可独立开关的处理步骤：
* ``func(image, params) -> (image, notes, metrics)``，纯函数式，便于测试；
* ``params_spec`` 描述可调参数（名称、范围、默认值），UI 据此自动生成控件；
* 关闭的阶段由 Runner 直接跳过（函数不被调用）。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import numpy as np

from docscan.core import binarize, color, enhance, freq, geometry, illumination
from docscan.core.types import Image, to_uint8

# 参数规格：(键, 中文标签, 默认值, 最小值, 最大值)
ParamSpec = tuple[str, str, float, float, float]


@dataclass
class Stage:
    """处理链中的一个步骤。"""

    name: str
    label: str
    func: Callable[[Image, dict], tuple[Image, str, dict[str, float]]]
    params_spec: list[ParamSpec] = field(default_factory=list)
    params: dict = field(default_factory=dict)
    enabled: bool = True

    def reset_params(self) -> None:
        """把参数恢复为默认值。"""
        self.params = {key: default for key, _, default, _, _ in self.params_spec}


def stage_warp(
    img: Image, params: dict
) -> tuple[Image, str, dict[str, float]]:
    """几何校正阶段。"""
    min_conf = float(params.get("min_confidence", 0.6))
    quad = geometry.detect_quad(img, min_confidence=min_conf)
    if quad.is_fallback:
        note = f"角点检测失败（置信度 {quad.confidence:.2f}），保留整图"
        return img, note, {"confidence": quad.confidence, "warped": 0.0}
    out = geometry.warp_perspective(img, quad)
    note = f"角点置信度 {quad.confidence:.2f}"
    return out, note, {"confidence": quad.confidence, "warped": 1.0}


def stage_binarize(
    img: Image, params: dict
) -> tuple[Image, str, dict[str, float]]:
    """自适应阈值二值化阶段（输入彩色图时先转灰度）。"""
    block = int(params.get("block_size", 31))
    c = float(params.get("c", 10.0))
    gray = binarize.to_gray(img)
    out = binarize.adaptive(gray, block_size=block, c=c)
    out = binarize.clean_small_components(out, min_area=int(params.get("min_area", 8)))
    ink = float((out < 128).mean())
    return out, f"blockSize={block}, C={c:g}", {"ink_ratio": ink}


def stage_illumination(
    img: Image, params: dict
) -> tuple[Image, str, dict[str, float]]:
    """光照归一阶段：背景估计去阴影。"""
    sigma = float(params.get("sigma", 30.0))
    out = illumination.remove_shading(img, sigma=sigma)
    before = float(np.asarray(to_uint8(img)).std())
    after = float((to_uint8(out)).std())
    return out, f"sigma={sigma:g}", {"std_before": before, "std_after": after}


def stage_white_balance(
    img: Image, params: dict
) -> tuple[Image, str, dict[str, float]]:
    """白平衡阶段：灰世界 / 白块估计。"""
    method = str(params.get("method", "white_patch"))
    strength = float(params.get("strength", 1.0))
    if method == "gray_world":
        out = color.gray_world(img, strength=strength)
    elif method == "white_patch":
        out = color.white_patch(img, strength=strength)
    else:
        raise ValueError(f"未知白平衡方法：{method}")
    return out, f"{method}, strength={strength:g}", {}


def stage_enhance(
    img: Image, params: dict
) -> tuple[Image, str, dict[str, float]]:
    """对比度增强阶段：CLAHE（只在亮度通道）。"""
    clip = float(params.get("clip_limit", 2.0))
    grid = int(params.get("grid_size", 8))
    gamma = float(params.get("gamma", 1.0))
    out = enhance.clahe(img, clip_limit=clip, grid_size=grid)
    if gamma != 1.0:
        out = enhance.apply_gamma(out, gamma)
    return out, f"CLAHE clip={clip:g}, gamma={gamma:g}", {}


def stage_freq_denoise(
    img: Image, params: dict
) -> tuple[Image, str, dict[str, float]]:
    """频域去网纹阶段：自动找峰或使用 UI 手动选点。"""
    manual = params.get("manual_peaks") or []
    if manual:
        peaks = [(int(du), int(dv)) for du, dv in manual]
        source = "手动选点"
    else:
        spec = freq.magnitude_spectrum(img)
        peaks = freq.find_periodic_peaks(
            spec, top_n=int(params.get("top_n", 4))
        )
        source = "自动找峰"
    if not peaks:
        return img, "未发现周期峰，跳过滤波", {"peaks": 0.0}

    kind = str(params.get("kind", "butterworth"))
    out = freq.notch_filter(
        img, peaks, kind=kind,
        radius=int(params.get("radius", 0)) or None,
        order=int(params.get("order", 4)),
    )
    drop = freq.energy_drop_db(img, out, peaks[0])
    note = f"{source} {len(peaks)} 点: {peaks[:4]}"
    return out, note, {"peaks": float(len(peaks)), "peak_drop_db": drop}


WARP_SPEC: list[ParamSpec] = [
    ("min_confidence", "角点置信度阈值", 0.6, 0.0, 1.0),
]
ILLUM_SPEC: list[ParamSpec] = [
    ("sigma", "光照场尺度", 30.0, 5.0, 120.0),
]
WB_SPEC: list[ParamSpec] = [
    ("strength", "校正强度", 1.0, 0.0, 1.0),
]
ENHANCE_SPEC: list[ParamSpec] = [
    ("clip_limit", "CLAHE 削波", 2.0, 0.1, 10.0),
    ("gamma", "伽马", 1.0, 0.2, 3.0),
]
BINARIZE_SPEC: list[ParamSpec] = [
    ("block_size", "邻域边长(奇数)", 31, 3, 151),
    ("c", "阈值偏置 C", 10.0, 0.0, 40.0),
    ("min_area", "最小连通域", 8, 0, 200),
]
FREQ_SPEC: list[ParamSpec] = [
    ("top_n", "自动找峰数量", 4, 1, 12),
    ("radius", "陷波半径(0=自适应)", 0.0, 0.0, 40.0),
    ("order", "Butterworth 阶数", 4.0, 1.0, 16.0),
]


def build_default_stages() -> list[Stage]:
    """默认处理链：几何校正 → 光照归一 → 白平衡 → 增强 → 去网纹 → 二值化。

    白平衡、增强、去网纹默认关闭（按需在 UI 勾选），后续迭代追加复原阶段。
    """
    stages = [
        Stage("warp", "几何校正", stage_warp, WARP_SPEC),
        Stage("illumination", "光照归一", stage_illumination, ILLUM_SPEC),
        Stage("white_balance", "白平衡", stage_white_balance, WB_SPEC, enabled=False),
        Stage("enhance", "对比度增强", stage_enhance, ENHANCE_SPEC, enabled=False),
        Stage("freq", "去网纹", stage_freq_denoise, FREQ_SPEC, enabled=False),
        Stage("binarize", "二值化", stage_binarize, BINARIZE_SPEC),
    ]
    for s in stages:
        s.reset_params()
    return stages


def make_preview(image: Image, max_side: int = 900) -> Image:
    """生成 UI 预览用缩略图（uint8 BGR）。"""
    arr = to_uint8(image)
    h, w = arr.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return arr
    scale = max_side / longest
    return cv2.resize(
        arr, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA
    )
