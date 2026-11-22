"""几何模块测试：四点排序、四边形检测、透视变换。"""

from __future__ import annotations

import itertools

import cv2
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

    @pytest.mark.parametrize("degrees", [-30, 30])
    def test_rotated_rect_keeps_cyclic_order(self, degrees):
        """正反倾斜时，打乱的纸张四角应恢复原来的左上起点和环序。"""
        theta = np.deg2rad(degrees)
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        rect = np.array([[0, 0], [400, 0], [400, 600], [0, 600]], dtype=np.float64) - 200
        pts = (rect @ rot.T) + 200
        got = geometry.order_points(pts[[2, 0, 3, 1]])
        np.testing.assert_allclose(got, pts, atol=1e-4)


def test_edge_candidates_include_page_inside_another_outline():
    edges = np.zeros((400, 500), dtype=np.uint8)
    cv2.rectangle(edges, (5, 5), (490, 390), 255, 1)
    cv2.rectangle(edges, (85, 50), (350, 340), 255, 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    page = np.float32([[85, 50], [350, 50], [350, 340], [85, 340]])
    candidates = geometry._edge_quads(edges, kernel)
    assert any(polygon_iou(geometry.order_points(q), page) > 0.98 for q in candidates)


def test_edge_support_rewards_observed_page_sides():
    # 已知的矩形纸边；平移 10 像素后，只有两条水平边仍能大部分重合。
    edges = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(edges, (40, 35), (150, 165), 255, 1)
    distance = cv2.distanceTransform(cv2.bitwise_not(edges), cv2.DIST_L2, 3)
    page = np.float32([[40, 35], [150, 35], [150, 165], [40, 165]])
    assert geometry._edge_support(page, distance, 2) == 1
    assert geometry._edge_support(page + [10, 0], distance, 2) < 0.55


class TestDetectQuad:
    def test_detects_tilted_page(self, tilted_page):
        img, gt = tilted_page
        quad = geometry.detect_quad(img)
        assert quad.source == "auto"
        assert quad.confidence >= 0.6
        iou = polygon_iou(quad.points, gt)
        assert iou > 0.95, f"IoU={iou:.3f}"

    def test_smooth_side_light_without_page_edges_keeps_full_frame(self):
        ramp = np.linspace(70, 220, 800, dtype=np.uint8)
        image = np.broadcast_to(ramp[None, :, None], (600, 800, 3)).copy()
        assert geometry.detect_quad(image).is_fallback

    def test_detects_full_page_under_shadow(self):
        img = np.full((600, 800, 3), 60, dtype=np.uint8)
        img[60:540, 180:620] = 235
        img[60:540, 180:400] = 90
        corners = np.array([[180, 60], [619, 60], [619, 539], [180, 539]], dtype=np.float32)
        detected = geometry.detect_quad(img)
        assert detected.source == "auto"
        assert polygon_iou(detected.points, corners) > 0.95

    def test_oblique_page_with_similar_gray_background(self):
        # 下半页与彩色桌面灰度几乎相同，只有颜色差异能补全这段纸边。
        img = np.full((560, 780, 3), (60, 120, 150), dtype=np.uint8)
        corners = np.int32([[155, 85], [590, 140], [690, 460], [85, 490]])
        page = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(page, corners, 255)
        img[page > 0] = 122
        img[:320][page[:320] > 0] = 160
        detected = geometry.detect_quad(img)
        assert detected.source == "auto"
        assert polygon_iou(detected.points, corners) > 0.95

    def test_low_contrast_page_with_dark_and_bright_background_details(self):
        img = np.full((500, 650, 3), 180, dtype=np.uint8)
        corners = np.int32([[90, 150], [370, 45], [555, 320], [240, 450]])
        cv2.fillConvexPoly(img, corners, (188, 188, 188))
        cv2.circle(img, (35, 30), 30, (20, 20, 20), -1)
        cv2.circle(img, (612, 450), 30, (250, 250, 250), -1)
        detected = geometry.detect_quad(img)
        assert detected.source == "auto"
        assert polygon_iou(detected.points, corners) > 0.98

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
