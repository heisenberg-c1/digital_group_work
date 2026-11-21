"""几何校正：四点排序、纸面四边形检测（含置信度）、透视变换。

算法链（检测在缩略图上做，变换在全分辨率上做）::

    缩略到 ≤800px → 多路纸面掩膜 → 开闭运算 → 外轮廓
        → 凸包 + approxPolyDP 取四点 → 多候选打分 → 透视变换

设计要点：
* 掩膜不用固定亮度阈值（暗光下会失效），改用 Otsu 自适应 + 低饱和高亮双路，
  两路结果一起参与候选打分，任一路成功即可；
* 检测失败绝不静默：返回 ``source="fallback"`` 的整图四边形，调用方必须提示；
* 四点排序用「质心极角 + 定向 + 起点归位」，对 4! 种输入排列结果完全一致。
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from docscan.core.types import Image, Quad

# 候选四边形打分的权重：面积占比 / 凸性 / 直角程度 / 长宽比接近 A4
_W_AREA = 0.35
_W_CONVEX = 0.25
_W_ANGLE = 0.20
_W_ASPECT = 0.20

_A4_ASPECT = math.sqrt(2.0)  # 1.414，纵向取倒数后等价
_MIN_AREA_RATIO = 0.03


def _polygon_area(pts: np.ndarray) -> float:
    """鞋带公式计算多边形面积（取绝对值）。"""
    x, y = pts[:, 0], pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def order_points(points: np.ndarray) -> np.ndarray:
    """把任意顺序的四个点规范为 左上、右上、右下、左下。

    做法：按质心极角排序得到环序 → 用鞋带公式统一绕向 → 旋转环序使
    最小 y（同 y 取最小 x）的点排在首位。相比 ``sum/diff`` 排序，该写法在
    极端退化四边形（细长、强旋转）上不会出现两点落到同一象限的错误。

    Args:
        points: (4, 2) 任意顺序的四个点。

    Returns:
        (4, 2) float32，顺序为 tl, tr, br, bl。
    """
    pts = np.asarray(points, dtype=np.float64).reshape(4, 2)
    centroid = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    order = np.argsort(angles, kind="stable")
    pts = pts[order]

    x, y = pts[:, 0], pts[:, 1]
    signed = float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    if signed < 0:
        pts = pts[::-1]

    start = int(np.lexsort((pts[:, 0], pts[:, 1]))[0])
    pts = np.roll(pts, -start, axis=0)
    return pts.astype(np.float32)


def full_image_quad(shape: tuple[int, ...], *, confidence: float = 0.0) -> Quad:
    """构造覆盖整幅图像的四边形，用作检测失败时的兜底。"""
    h, w = shape[:2]
    pts = np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
        dtype=np.float32,
    )
    return Quad(points=pts, confidence=confidence, source="fallback")


def _paper_masks(bgr: Image) -> list[np.ndarray]:
    """生成多路纸面掩膜候选（白=纸面）。"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    masks: list[np.ndarray] = []

    # 路 1：灰度 Otsu。暗背景下纸面是亮区，这一步等价于「自适应亮度阈值」，
    # 且不依赖绝对亮度值，欠曝场景仍然成立。
    _, m_gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks.append(m_gray)

    # 路 2：HSV 的 V 通道 Otsu + 低饱和约束。纸面「亮且不彩」，能排除彩色桌面。
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v_ch, s_ch = hsv[:, :, 2], hsv[:, :, 1]
    _, m_v = cv2.threshold(v_ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_sat = (s_ch < 70).astype(np.uint8) * 255
    masks.append(cv2.bitwise_and(m_v, low_sat))

    return masks


def _quad_metrics(quad: np.ndarray, img_area: float) -> dict[str, float]:
    """计算候选四边形的几何指标。"""
    area = _polygon_area(quad)
    hull = cv2.convexHull(quad.astype(np.int32))
    hull_area = _polygon_area(hull.reshape(-1, 2).astype(np.float64)) or 1.0

    cos_sum = 0.0
    for i in range(4):
        p_prev = quad[(i - 1) % 4]
        p_cur = quad[i]
        p_next = quad[(i + 1) % 4]
        v1 = p_prev - p_cur
        v2 = p_next - p_cur
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            cos_sum += 1.0
            continue
        cos_sum += abs(float(np.dot(v1, v2) / (n1 * n2)))

    w_top = np.linalg.norm(quad[1] - quad[0])
    w_bottom = np.linalg.norm(quad[2] - quad[3])
    h_left = np.linalg.norm(quad[3] - quad[0])
    h_right = np.linalg.norm(quad[2] - quad[1])
    w = max(w_top, w_bottom, 1e-6)
    h = max(h_left, h_right, 1e-6)

    return {
        "area_ratio": area / img_area,
        "convexity": min(area / hull_area, 1.0),
        "angle_cos": cos_sum / 4.0,
        "aspect": max(w, h) / min(w, h),
        "width": float(w),
        "height": float(h),
    }


def score_quad(quad: np.ndarray, img_area: float) -> float:
    """给候选四边形打分，返回 0~1 的置信度。

    公式::

        score = 0.35·min(area_ratio/0.6, 1) + 0.25·convexity
              + 0.20·(1 - mean|cosθ|) + 0.20·exp(-|ln(aspect/√2)|)

    面积占比按 0.6 饱和（超过就满分），因为近距离拍摄时纸面几乎铺满画面；
    长宽比以 A4 为理想值，但用指数衰减而非硬阈值，允许 B5 / 16K 等尺寸。
    """
    m = _quad_metrics(quad, img_area)
    area_score = min(m["area_ratio"] / 0.6, 1.0)
    angle_score = max(0.0, 1.0 - m["angle_cos"])
    aspect_score = math.exp(-abs(math.log(m["aspect"] / _A4_ASPECT)))
    return (
        _W_AREA * area_score
        + _W_CONVEX * m["convexity"]
        + _W_ANGLE * angle_score
        + _W_ASPECT * aspect_score
    )


def _contours_to_quads(mask: np.ndarray, kernel: np.ndarray) -> list[np.ndarray]:
    """从掩膜中提取四边形候选。"""
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_area = float(mask.shape[0] * mask.shape[1])
    quads: list[np.ndarray] = []
    for cnt in contours:
        area = _polygon_area(cnt.reshape(-1, 2).astype(np.float64))
        if area < _MIN_AREA_RATIO * img_area:
            continue
        if area > 0.985 * img_area:
            # 轮廓等于整幅画面：说明掩膜把背景也当成了纸面（纯色背景/全白图），
            # 此时没有「页面边界」可找，不应当作检测成功
            continue
        hull = cv2.convexHull(cnt)
        peri = cv2.arcLength(hull, True)
        found = None
        for eps_ratio in (0.02, 0.03, 0.04, 0.05, 0.08):
            approx = cv2.approxPolyDP(hull, eps_ratio * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                found = approx.reshape(4, 2).astype(np.float32)
                break
        if found is None:
            # 轮廓不闭合或边被遮挡时，退而用最小外接旋转矩形，仍可给出可用四边形
            (_, _), (rw, rh), _ = cv2.minAreaRect(hull)
            if rw < 8 or rh < 8:
                continue
            found = cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)
        quads.append(found)
    return quads


def detect_quad(
    img: Image,
    *,
    max_side: int = 800,
    min_confidence: float = 0.6,
) -> Quad:
    """检测文档页面的四个角点。

    在长边 ≤ ``max_side`` 的缩略图上检测（12MP 图检测耗时从 ~1.5s 降到 ~40ms），
    返回的坐标已映射回原图尺度。

    Args:
        img: BGR 图像（uint8 或 float32 均可）。
        max_side: 检测所用缩略图长边上限。
        min_confidence: 置信度阈值，低于此值判为检测失败。

    Returns:
        Quad。成功时 ``source="auto"``；失败时 ``source="fallback"`` 且返回整图四角，
        调用方应据此给出可见提示。
    """
    if img.ndim != 3:
        raise ValueError("detect_quad 需要三通道 BGR 图像")

    h, w = img.shape[:2]
    scale = min(1.0, max_side / float(max(h, w)))
    if scale < 1.0:
        small = cv2.resize(
            img, (max(1, round(w * scale)), max(1, round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = img

    if small.dtype != np.uint8:
        small = np.rint(np.clip(small, 0.0, 1.0) * 255.0).astype(np.uint8)

    sh, sw = small.shape[:2]
    img_area = float(sh * sw)
    k = max(3, round(min(sh, sw) * 0.01) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    best_quad: np.ndarray | None = None
    best_score = -1.0
    for mask in _paper_masks(small):
        for cand in _contours_to_quads(mask, kernel):
            cand = order_points(cand)
            score = score_quad(cand, img_area)
            # 四点全部贴边 = 检测到的其实是整幅画面，多半没找到纸面，压低置信度
            if _touches_border(cand, sw, sh):
                score *= 0.75
            if score > best_score:
                best_score, best_quad = score, cand

    if best_quad is None or best_score < min_confidence:
        return full_image_quad(img.shape, confidence=max(best_score, 0.0))

    pts = best_quad / scale
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    return Quad(points=order_points(pts), confidence=float(best_score), source="auto")


def _touches_border(quad: np.ndarray, w: int, h: int, tol: int = 3) -> bool:
    """四个角点是否都紧贴图像边界。"""
    near = 0
    for x, y in quad:
        if x <= tol or x >= w - 1 - tol or y <= tol or y >= h - 1 - tol:
            near += 1
    return near == 4


def warp_perspective(
    img: Image,
    quad: Quad | np.ndarray,
    *,
    aspect_prior: float | None = None,
    max_output_side: int | None = None,
) -> Image:
    """按四点做射影变换，得到正视的矩形页面。

    输出尺寸由四边形的对边长度决定（取较长者，避免拉伸），可再用
    ``aspect_prior`` 吸附到先验长宽比（A4 = √2，横放取倒数）。

    Args:
        img: BGR 图像。
        quad: Quad 或 (4,2) 点集，内部会重新排序。
        aspect_prior: 目标长宽比（宽/高）。给定时按它修正输出宽高。
        max_output_side: 输出长边上限，用于限制大角度倾斜产生的超宽画布。

    Returns:
        变换后的图像，尺寸为 (H', W')。

    Raises:
        ValueError: 四边形退化为零面积。
    """
    pts = order_points(quad.points if isinstance(quad, Quad) else quad)
    if _polygon_area(pts) < 1.0:
        raise ValueError("四边形面积过小，无法做透视变换")

    tl, tr, br, bl = pts
    w = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    h = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))

    if aspect_prior is not None and aspect_prior > 0:
        if w / max(h, 1e-6) < aspect_prior:
            w = h * aspect_prior
        else:
            h = w / aspect_prior

    if max_output_side is not None:
        longest = max(w, h)
        if longest > max_output_side:
            s = max_output_side / longest
            w, h = w * s, h * s

    # 像素坐标约定：坐标 x 对应第 x 个像素的中心，因此边长 w 覆盖 0..w 共 w+1 个像素
    out_w = max(2, round(w) + 1)
    out_h = max(2, round(h) + 1)
    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(
        img,
        matrix,
        (out_w, out_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def auto_orient(img: Image, *, aspect_prior: float = _A4_ASPECT) -> Image:
    """按 A4 先验把横向页面转正（竖放）。

    仅在长宽比明确偏横向（> 1.15·√2 的倒数关系）时旋转 90°，避免误转。
    """
    h, w = img.shape[:2]
    if w > h * 1.15:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img
