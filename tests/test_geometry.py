"""几何模块测试：四点排序、四边形检测、透视变换。"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from conftest import make_page, paste_on_background, polygon_iou
from docscan.core import geometry


class TestOrderPoints:
    def test_all_permutations_converge(self):
        """4! = 24 种输入排列必须得到完全一致的四点顺序。"""
        quad = np.array([[120, 90], [700, 150], [640, 950], [80, 880]], dtype=np.float32)
        base = geometry.order_points(quad)
        for perm in itertools.permutations(range(4)):
            got = geometry.order_points(quad[list(perm)])
            np.testing.assert_allclose(got, base, atol=1e-4)

    def test_starts_from_top_left(self):
        quad = np.array([[10, 10], [300, 10], [300, 400], [10, 400]], dtype=np.float32)
        got = geometry.order_points(quad)
        np.testing.assert_allclose(got[0], [10, 10])
        np.testing.assert_allclose(got[1], [300, 10])
        np.testing.assert_allclose(got[2], [300, 400])
        np.testing.assert_allclose(got[3], [10, 400])

    def test_rotated_rect_keeps_cyclic_order(self):
        """30° 旋转的矩形：排序结果仍为环序且起点是最上方的角。"""
        theta = np.deg2rad(30)
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        rect = np.array([[0, 0], [400, 0], [400, 600], [0, 600]], dtype=np.float64) - 200
        pts = (rect @ rot.T) + 200
        got = geometry.order_points(pts.astype(np.float32))
        ys = got[:, 1]
        assert ys[0] == ys.min()  # 起点是最高点
        # 环序：相邻边交替横/竖
        d = np.diff(np.vstack([got, got[:1]]), axis=0)
        assert (np.linalg.norm(d, axis=1) > 100).all()


class TestDetectQuad:
    def test_detects_tilted_page(self, tilted_page):
        img, gt = tilted_page
        quad = geometry.detect_quad(img)
        assert quad.source == "auto"
        assert quad.confidence >= 0.6
        iou = polygon_iou(quad.points, gt)
        assert iou > 0.95, f"IoU={iou:.3f}"

    def test_detects_straight_page(self, synthetic_page):
        img = paste_on_background(
            synthetic_page(600, 800),
            np.array([[150, 200], [749, 200], [749, 999], [150, 999]], dtype=np.float32),
            canvas_size=(900, 1200),
        )
        quad = geometry.detect_quad(img)
        assert quad.source == "auto"
        assert polygon_iou(quad.points, np.array([[150, 200], [749, 200], [749, 999], [150, 999]])) > 0.95

    def test_flat_texture_is_low_confidence(self):
        """无纸面的纯纹理图：应判为检测失败并给出低置信度。"""
        rng = np.random.default_rng(42)
        base = rng.uniform(90, 170, size=(700, 900, 1)).astype(np.float32)
        texture = base + rng.normal(0, 18, size=(700, 900, 1)).astype(np.float32)
        img = np.clip(np.repeat(texture, 3, axis=2), 0, 255).astype(np.uint8)
        quad = geometry.detect_quad(img, min_confidence=0.6)
        assert quad.is_fallback or quad.confidence < 0.6

    def test_uniform_background_is_fallback(self):
        """全白画面没有页面边界，必须显式报告失败而不是悄悄返回整图。"""
        img = np.full((800, 600, 3), 240, dtype=np.uint8)
        quad = geometry.detect_quad(img, min_confidence=0.6)
        assert quad.is_fallback

    def test_result_is_tl_tr_br_bl(self, tilted_page):
        img, _ = tilted_page
        quad = geometry.detect_quad(img)
        pts = quad.points
        assert pts[0, 0] < pts[1, 0] and pts[3, 0] < pts[2, 0]  # tl/bl 在左
        assert pts[0, 1] < pts[3, 1] and pts[1, 1] < pts[2, 1]  # tl/tr 在上


class TestWarpPerspective:
    def test_rect_quad_is_near_identity(self, synthetic_page):
        """与图像等大的矩形四边形：输出必须与原图逐像素一致。"""
        img = synthetic_page(320, 480)
        quad = np.array([[0, 0], [319, 0], [319, 479], [0, 479]], dtype=np.float32)
        out = geometry.warp_perspective(img, quad)
        assert out.shape == img.shape
        assert float(np.abs(out.astype(int) - img.astype(int)).mean()) < 0.5

    def test_warps_tilted_to_frontal(self):
        """斜拍的空白页校正后应成为均匀亮色（std 极小）。"""
        from conftest import make_page as mp

        page = mp(600, 850, line_count=0)  # 纯白页面，无文字干扰
        gt = np.array([[180, 140], [720, 210], [660, 1080], [230, 1010]], dtype=np.float32)
        img = paste_on_background(page, gt)
        out = geometry.warp_perspective(img, gt)
        # 只看页面内部：边缘 10px 受边界重采样影响，不算数
        inner = out[10:-10, 10:-10]
        assert float(inner.std()) < 5.0

    def test_output_size_from_edge_lengths(self, synthetic_page):
        img = synthetic_page(400, 600)
        quad = np.array([[0, 0], [200, 0], [200, 300], [0, 300]], dtype=np.float32)
        out = geometry.warp_perspective(img, quad)
        # 边长 200 → 像素坐标 0..200，共 201 像素
        assert (out.shape[1], out.shape[0]) == (201, 301)

    def test_degenerate_quad_raises(self, synthetic_page):
        img = synthetic_page(200, 200)
        quad = np.array([[10, 10], [50, 10], [50, 10], [10, 10]], dtype=np.float32)
        with pytest.raises(ValueError):
            geometry.warp_perspective(img, quad)


class TestAutoOrient:
    def test_landscape_rotated(self):
        img = make_page(800, 500)
        out = geometry.auto_orient(img)
        assert out.shape[0] > out.shape[1]

    def test_portrait_untouched(self):
        img = make_page(500, 800)
        out = geometry.auto_orient(img)
        assert out.shape == img.shape
