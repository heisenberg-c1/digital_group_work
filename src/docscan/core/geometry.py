"""几何校正：四点排序、纸面四边形检测（含置信度）、透视变换。

算法链（检测在缩略图上做，变换在全分辨率上做）::

    缩略到 ≤800px → 多路纸面掩膜 → 先闭后开 → 外轮廓
        → 凸包 + approxPolyDP 取四点；同时从闭合边缘提取候选
    → 按形状及四边与图像边缘的贴合程度打分 → 透视变换

设计要点：
* 掩膜不用固定亮度阈值（暗光下会失效），改用 Otsu 全局自动阈值 + 低饱和高亮掩膜，
  暗背景下另在较暗像素中估计阈值，使阴影中的纸面也能参与候选打分；
* 检测失败绝不静默：返回 ``source="fallback"`` 的整图四边形，调用方必须提示；
* 四点排序用「按质心极角排序，再以 x + y 最小的角为起点」，对 4! 种输入排列结果完全一致。
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
    """将凸四边形的四个角点按图像中的顺时针方向排列。

    先按各点相对四点中心的极角升序排列，再将 x + y 最小的角移到首位。
    对于接近正向拍摄的页面，该点通常是纸张左上角。页面大角度旋转时，
    选出的起点可能对应纸张的其他角。

    Args:
        points: (4, 2) 任意顺序的四个点。

    Returns:
        (4, 2) float32，页面接近正向时通常依次为左上 (tl)、右上 (tr)、右下 (br)、左下 (bl)。
    """
    pts = np.asarray(points, dtype=np.float64).reshape(4, 2)
    centroid = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    order = np.argsort(angles, kind="stable")
    pts = pts[order]

    # 按质心角排序后已经顺时针定序，无需使用鞋带公式统一绕向
    # x, y = pts[:, 0], pts[:, 1]
    # signed = float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    # if signed < 0:
    #     pts = pts[::-1]
    
    #  选 x + y 最小的角作为起点
    start = int(np.argmin(pts[:, 0] + pts[:, 1]))
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

    # 路 1：灰度 Otsu。按整幅图的灰度分布自动选一个阈值，适用于纸面比背景亮的情况。
    threshold, m_gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks.append(m_gray)

    # 路 2：HSV 的 V 通道 Otsu + 低饱和约束。纸面「亮且不彩」，能排除彩色桌面。
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v_ch, s_ch = hsv[:, :, 2], hsv[:, :, 1]
    _, m_v = cv2.threshold(v_ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_sat = (s_ch < 70).astype(np.uint8) * 255
    masks.append(cv2.bitwise_and(m_v, low_sat))

    # 亮区阈值可能只选中受光的半页。在暗部再次分割背景与受阴影的纸面。
    dark_pixels = gray[gray <= threshold]
    if dark_pixels.size and dark_pixels.min() < dark_pixels.max():
        shadow_threshold, _ = cv2.threshold(
            dark_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
        if shadow_threshold > 0 and np.percentile(border, 95) <= shadow_threshold:
            _, shadow_mask = cv2.threshold(gray, shadow_threshold, 255, cv2.THRESH_BINARY)
            masks.append(shadow_mask)

    return masks


def _quad_metrics(quad: np.ndarray, img_area: float) -> dict[str, float]:
    """计算候选四边形的几何指标。"""
    area = _polygon_area(quad)
    hull = cv2.convexHull(quad)
    hull_area = _polygon_area(hull.reshape(-1, 2)) or 1.0

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


def _edge_quads(edges: np.ndarray, kernel: np.ndarray) -> list[np.ndarray]:
    """沿闭合边界找四角，保留被桌面、屏幕等外轮廓包围的纸页。"""
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not _MIN_AREA_RATIO * edges.size <= area <= 0.985 * edges.size:
            continue
        hull = cv2.convexHull(contour)
        polygon = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
        if len(polygon) == 4:
            candidates.append(polygon.reshape(4, 2))
    return candidates


def _edge_support(quad: np.ndarray, distance: np.ndarray, tolerance: float) -> float:
    """统计每条候选边附近实际有图像边缘的比例，再对四边取平均。"""
    fractions = np.linspace(0.05, 0.95, 64, dtype=np.float32)
    ends = np.roll(quad, -1, axis=0)
    samples = quad[:, None, :] + (ends - quad)[:, None, :] * fractions[None, :, None]
    pixels = np.rint(samples).astype(int)
    x, y = pixels[..., 0], pixels[..., 1]
    h, w = distance.shape
    return float(np.mean(distance[y.clip(0, h - 1), x.clip(0, w - 1)] <= tolerance))


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
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    smoothed = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(smoothed, 30, 90)
    candidates = _edge_quads(edges, kernel)
    # 纸面与桌面灰度相近时，用饱和度变化补齐灰度边缘的断点。
    saturation = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 1]
    color_edges = cv2.Canny(cv2.GaussianBlur(saturation, (3, 3), 0), 20, 60)
    if np.any(color_edges):
        combined_edges = cv2.bitwise_or(edges, color_edges)
        candidates.extend(_edge_quads(combined_edges, kernel))
    distance = cv2.distanceTransform(cv2.bitwise_not(edges), cv2.DIST_L2, 3)
    for mask in _paper_masks(small):
        candidates.extend(_contours_to_quads(mask, kernel))

    def consider(quads: list[np.ndarray], edge_distance: np.ndarray) -> None:
        nonlocal best_quad, best_score
        for candidate in quads:
            cand = order_points(candidate)
            support = _edge_support(cand, edge_distance, k / 2)
            score = score_quad(cand, img_area) * (0.6 + 0.4 * support)
            if _touches_border(cand, sw, sh):
                score *= 0.75
            if score > best_score:
                best_score, best_quad = score, cand

    consider(candidates, distance)
    if best_score < min_confidence:
        # 同色浅背景可能只留下很弱的纸边。使用小核闭合，减少弱纹理互相粘连。
        weak_edges = cv2.Canny(smoothed, 8, 24)
        fine_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        weak_candidates = _edge_quads(weak_edges, fine_kernel)
        if weak_candidates:
            weak_distance = cv2.distanceTransform(cv2.bitwise_not(weak_edges), cv2.DIST_L2, 3)
            consider(weak_candidates, weak_distance)

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

    输出尺寸由四边形的对边长度决定（取较长者估算），可再用
    ``aspect_prior`` 吸附到先验长宽比（宽/高：竖放 A4 为 1/√2，横放为 √2）。

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
    """按图像宽高判断是否旋转为竖向。

    宽度超过高度的 1.15 倍时顺时针旋转 90°。当前判断不使用 aspect_prior。
    """
    h, w = img.shape[:2]
    if w > h * 1.15:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img
