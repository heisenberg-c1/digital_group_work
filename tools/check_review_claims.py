"""组员修复意见的机械复验：一条命令确认 D1–D9 / N1、N3 在本地已落地。

对照文档：``组员修复意见核对报告.md``（项目根目录上一层）。
每一项都是可执行的断言，而不是人工读代码——这样修复被回退时会直接失败。

用法::

    python tools/check_review_claims.py

退出码 0 表示全部通过，1 表示有项目未达标。
"""

from __future__ import annotations

import inspect
import traceback
from pathlib import Path

import cv2
import numpy as np

from docscan.core import colorspace, freq, metrics
from docscan.experiments import synth
from docscan.experiments.exp_e2e import STAGE_POLICY, _resolve_scene
from docscan.experiments.utils import render_text_page
from docscan.pipeline.stages import (
    ENHANCE_SPEC,
    RESTORE_SPEC,
    stage_enhance,
    stage_restore,
)

ROOT = Path(__file__).resolve().parents[1]


def _gray(x) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    return np.clip(arr, 0.0, 1.0)


def _dense_page(width: int = 320, height: int = 240) -> np.ndarray:
    """高密度「文字块」页：白底 + 随机黑色横条。

    不用 render_text_page 是因为它按段落排版，小画布下大半是空白，
    SSIM 会饱和到 1.0，散焦/复原的差异被抹平（复验会假通过）。
    """
    img = np.full((height, width), 0.92, np.float32)
    rng = np.random.default_rng(20261110)
    for y in range(10, height - 10, 14):
        x = 10
        while x < width - 12:
            seg = int(rng.integers(6, 30))
            if rng.random() < 0.75:
                img[y:y + 7, x:min(x + seg, width - 12)] = 0.15
            x += seg + 6
    return img


def d1_long_period_not_masked() -> str:
    """D1：λ=64 轴向网纹必须检出真实峰，陷波后优于不处理。"""
    size = 1024
    base = np.full((size, size), 0.35, np.float32)
    striped = synth.add_moire(base, wavelength=64, angle_deg=0.0, amplitude=0.2)
    peaks = freq.find_periodic_peaks(freq.magnitude_spectrum(striped), top_n=2)
    assert any(abs(abs(du) - 16) <= 2 and abs(dv) <= 2 for du, dv in peaks), f"峰不对：{peaks}"
    out = freq.notch_filter(striped, peaks, radius=4)
    before, after = metrics.ssim(base, striped), metrics.ssim(base, out)
    assert after > before + 0.1, f"陷波后 {after:.4f} 未优于 {before:.4f}"
    assert "max_period_px" in inspect.signature(freq.find_periodic_peaks).parameters
    return f"峰 {peaks[0]}，SSIM {before:.4f} → {after:.4f}"


def d2_defocus_is_real() -> str:
    """D2：defocus 必须在策略表里，且未知场景在合成路径抛错。"""
    assert "defocus" in STAGE_POLICY
    assert STAGE_POLICY["defocus"]["restore"] is True
    try:
        _resolve_scene("blur", synthetic=True)
    except ValueError:
        pass
    else:
        raise AssertionError("合成路径未知场景未抛错")
    page, _ = render_text_page(width=600, height=800, font_px=28)
    deg, gt = synth.synthesize(_gray(page), "defocus")
    ssim = metrics.ssim(_gray(gt), _gray(deg))
    assert ssim < 0.999, f"defocus 未真的退化（SSIM={ssim:.4f}）"
    return f"退化图 SSIM={ssim:.4f}"


def d3_hsv_has_caller() -> str:
    """D3：HSV 三函数存在，且实验脚本真的调用它们（不再是死代码）。"""
    for name in ("bgr_to_hsv", "hsv_to_bgr", "apply_on_value"):
        assert hasattr(colorspace, name), f"缺少 {name}"
    exp = ROOT / "src" / "docscan" / "experiments" / "exp_colorspace.py"
    assert exp.exists(), "exp_colorspace.py 不存在，HSV 仍无调用方"
    src = exp.read_text(encoding="utf-8")
    assert "apply_on_value" in src and "apply_on_luma" in src
    return f"调用方 {exp.relative_to(ROOT)}"


def d4_cast_uses_color() -> str:
    """D4：cast 场景必须吃彩色输入，退化后与 shadow 可区分。"""
    src = inspect.getsource(__import__("docscan.experiments.exp_e2e", fromlist=["x"]))
    assert 'synthesize(page, "cast")' in src
    page, _ = render_text_page(width=600, height=800, font_px=28)
    cast, _ = synth.synthesize(page, "cast")
    means = np.asarray(cast).reshape(-1, 3).mean(0)
    assert means.max() - means.min() > 0.1, f"通道均值几乎相同，偏色没生效：{means}"
    return "通道均值 B/G/R = " + "/".join(f"{m:.2f}" for m in means)


def d5_enhance_methods_wired() -> str:
    """D5：四种方法都可分派，he 别名可用，未知方法抛错。"""
    options = {k for k, _ in ENHANCE_SPEC[0].choices}
    assert {"clahe", "histeq", "stretch", "none"} <= options, options
    page, _ = render_text_page(width=400, height=520, font_px=24)
    he, note, _ = stage_enhance(page, {"method": "he", "gamma": 1.0})
    histeq, note2, _ = stage_enhance(page, {"method": "histeq", "gamma": 1.0})
    assert note == note2 == "直方图均衡", f"he 别名未映射：{note!r}"
    np.testing.assert_array_equal(he, histeq)
    try:
        stage_enhance(page, {"method": "bogus"})
    except ValueError:
        pass
    else:
        raise AssertionError("未知增强方法未抛错")
    return "clahe/histeq/stretch/none + he 别名，未知抛错"


def d6_disk_psf_wins() -> str:
    """D6：散焦用圆盘 PSF 必须优于线段 PSF，disk 别名可用，未知值抛错。"""
    options = {k for k, _ in RESTORE_SPEC[0].choices}
    assert {"motion", "defocus"} <= options
    gt = _dense_page()
    from docscan.core import restore

    degraded = cv2.filter2D(gt, -1, restore.disk_psf(31, 5))
    disk, _, m = stage_restore(degraded, {"psf_type": "disk", "defocus_radius": 5, "K": 0.003})
    motion, _, _ = stage_restore(
        degraded,
        {"psf_type": "motion", "psf_mode": "manual", "length": 11, "angle_deg": 0.0,
         "K": 0.003},
    )
    s_disk, s_motion = metrics.ssim(gt, disk), metrics.ssim(gt, motion)
    assert m["psf_radius"] == 5.0, "disk 别名未映射到散焦"
    assert s_disk > s_motion + 0.02, f"圆盘 {s_disk:.4f} 未显著优于线段 {s_motion:.4f}"
    try:
        stage_restore(gt, {"psf_type": "bogus"})
    except ValueError:
        pass
    else:
        raise AssertionError("未知 psf_type 未抛错")
    return f"圆盘 {s_disk:.4f} > 线段 {s_motion:.4f}"


def d7_matplotlib_declared() -> str:
    """D7：matplotlib 必须在 dev extra 里声明。"""
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    dev_block = text.split("dev = [", 1)[1].split("]", 1)[0]
    assert "matplotlib" in dev_block, "dev extra 缺少 matplotlib"
    return "dev extra 含 matplotlib"


def d8_csv_escapes_commas() -> str:
    """D8：写盘走 csv 模块，含逗号字段可被 DictReader 正确读回。"""
    import csv
    import tempfile

    from docscan.experiments import utils

    with tempfile.TemporaryDirectory() as tmp:
        path = utils.write_csv(
            Path(tmp) / "t.csv", ["method", "ssim"], [["Sauvola(25,k=0.2)", "0.91"]]
        )
        with path.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
    assert rows[0]["method"] == "Sauvola(25,k=0.2)", rows
    assert len(rows[0]) == 2, "字段被拆列"
    return "含逗号字段往返正确"


def d9_docs_consistent() -> str:
    """D9：注释引用的 CSV 都存在；README 不再写「五场景」。"""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "五场景" not in readme, "README 仍写「五场景」（实际 7 个）"
    assert "exp_colorspace" in readme and "exp_enhance" in readme, "README 漏记实验脚本"
    for exp in ("e2e_summary", "enhance_compare", "notch_compare", "sampling",
                "wiener_K_sweep", "colorspace_compare"):
        assert (ROOT / "data" / "results" / f"{exp}.csv").exists(), f"缺少产物 {exp}.csv"
    return "README 与 data/results 一致"


def n1_cli_shares_runner() -> str:
    """N1：CLI 必须走 pipeline.Runner，不能自拼处理链。"""
    src = (ROOT / "src" / "docscan" / "__main__.py").read_text(encoding="utf-8")
    assert "Runner(_build_stages(args))" in src, "CLI 未使用 Runner"
    assert "build_default_stages()" in src
    return "CLI 与 GUI 共用 Runner"


def n3_enum_params() -> str:
    """N3：参数元数据支持枚举，UI 可下拉选择。"""
    from docscan.pipeline.stages import EnumParam, ParamSpec  # noqa: F401

    assert isinstance(ENHANCE_SPEC[0], EnumParam)
    assert isinstance(RESTORE_SPEC[0], EnumParam)
    ui = (ROOT / "src" / "docscan" / "ui" / "widgets" / "stage_list.py").read_text(encoding="utf-8")
    assert "Combobox" in ui and "EnumParam" in ui, "UI 未渲染枚举控件"
    return "EnumParam + ttk.Combobox"


CHECKS = [
    ("D1", d1_long_period_not_masked),
    ("D2", d2_defocus_is_real),
    ("D3", d3_hsv_has_caller),
    ("D4", d4_cast_uses_color),
    ("D5", d5_enhance_methods_wired),
    ("D6", d6_disk_psf_wins),
    ("D7", d7_matplotlib_declared),
    ("D8", d8_csv_escapes_commas),
    ("D9", d9_docs_consistent),
    ("N1", n1_cli_shares_runner),
    ("N3", n3_enum_params),
]


def main() -> int:
    failed = 0
    for code, check in CHECKS:
        try:
            detail = check()
        except Exception as exc:
            failed += 1
            print(f"[FAIL] {code} {check.__name__}: {exc}")
            if not isinstance(exc, AssertionError):
                traceback.print_exc()
        else:
            print(f"[ OK ] {code} {check.__doc__.splitlines()[0]} —— {detail}")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} 项通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
