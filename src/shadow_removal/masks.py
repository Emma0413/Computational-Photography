from __future__ import annotations

import cv2
import numpy as np


def _shadow_likelihood(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    l = lab[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)

    darkness = 255.0 - l
    saturation_penalty = cv2.normalize(s, None, 0, 80, cv2.NORM_MINMAX)
    value_darkness = 255.0 - v
    return np.clip(0.60 * darkness + 0.30 * value_darkness - 0.10 * saturation_penalty, 0, 255)


def auto_shadow_mask(rgb: np.ndarray) -> np.ndarray:
    """Estimate a shadow mask from low luminance and saturation/color cues."""
    score_u8 = _shadow_likelihood(rgb).astype(np.uint8)
    _, mask = cv2.threshold(score_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return refine_mask(mask > 0)


def istd_calibrated_shadow_mask(rgb: np.ndarray) -> np.ndarray:
    """Estimate a region-level shadow mask calibrated on the local ISTD subset."""
    score = _shadow_likelihood(rgb).astype(np.float32)
    smooth = cv2.GaussianBlur(score, (0, 0), sigmaX=5, sigmaY=5, borderType=cv2.BORDER_REFLECT)
    mask = smooth >= float(np.percentile(smooth, 75))
    return _region_cleanup(mask, open_size=5, close_size=41, min_area_frac=0.002)


def _region_cleanup(mask: np.ndarray, open_size: int, close_size: int, min_area_frac: float) -> np.ndarray:
    cleaned = refine_mask(mask, open_size=open_size, close_size=close_size)
    h, w = cleaned.shape
    min_area = max(64, int(min_area_frac * h * w))
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned.astype(np.uint8), connectivity=8)

    region_mask = np.zeros_like(cleaned, dtype=bool)
    for label in range(1, labels_count):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            region_mask |= labels == label

    return refine_mask(region_mask, open_size=3, close_size=close_size)


def guo_region_shadow_mask(rgb: np.ndarray) -> np.ndarray:
    """Estimate a mask with a Guo-style paired shadow/non-shadow region test.

    This is a lightweight approximation of Guo, Dai, and Hoiem's detection idea:
    propose dark connected regions, compare each region with a neighboring lit
    ring, and keep regions that are darker while retaining compatible texture and
    chromaticity.
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    texture = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    score = _shadow_likelihood(rgb).astype(np.uint8)
    threshold, _ = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    high_score = float(np.percentile(score, 80))
    candidates = score >= max(float(threshold), high_score)
    candidates = refine_mask(candidates, open_size=3, close_size=7)

    h, w = candidates.shape
    min_area = max(48, int(0.0004 * h * w))
    max_area = int(0.70 * h * w)
    keep = np.zeros_like(candidates, dtype=bool)
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(candidates.astype(np.uint8), connectivity=8)

    for label in range(1, labels_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue

        region = labels == label
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        radius = max(11, min(41, int(0.25 * max(width, height))))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        outer = cv2.dilate(region.astype(np.uint8), kernel) > 0
        ring = outer & ~region

        if ring.sum() < max(32, area // 12):
            x0 = max(0, x - radius)
            y0 = max(0, y - radius)
            x1 = min(w, x + width + radius)
            y1 = min(h, y + height + radius)
            ring = np.zeros_like(region)
            ring[y0:y1, x0:x1] = True
            ring &= ~region

        if ring.sum() < 32:
            continue

        shadow_lab = lab[region]
        lit_lab = lab[ring]
        shadow_l = float(np.median(shadow_lab[:, 0]))
        lit_l = float(np.median(lit_lab[:, 0]))
        if lit_l - shadow_l < 7.0:
            continue

        chroma_gap = float(np.linalg.norm(np.median(shadow_lab[:, 1:3], axis=0) - np.median(lit_lab[:, 1:3], axis=0)))
        l_ratio = lit_l / max(shadow_l, 1.0)
        shadow_texture = float(np.median(texture[region]))
        lit_texture = float(np.median(texture[ring]))
        texture_gap = abs(shadow_texture - lit_texture) / max(shadow_texture + lit_texture, 1.0)

        if l_ratio > 1.08 and chroma_gap < 22.0 and texture_gap < 0.65:
            keep |= region

    if keep.sum() < min_area:
        keep = score >= float(np.percentile(score, 85))

    return refine_mask(keep, open_size=3, close_size=11)


MASK_GENERATORS = {
    "basic": auto_shadow_mask,
    "guo": guo_region_shadow_mask,
    "istd": istd_calibrated_shadow_mask,
}


def refine_mask(mask: np.ndarray, open_size: int = 3, close_size: int = 9) -> np.ndarray:
    mask_u8 = (mask.astype(np.uint8) * 255)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    cleaned = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, open_kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel)
    return cleaned > 127


def feather_alpha(mask: np.ndarray, radius: int = 15) -> np.ndarray:
    if radius <= 0:
        return mask.astype(np.float32)
    mask_u8 = mask.astype(np.uint8)
    dist_inside = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    dist_outside = cv2.distanceTransform(1 - mask_u8, cv2.DIST_L2, 5)
    signed = dist_inside - dist_outside
    alpha = np.clip((signed + radius) / (2 * radius), 0.0, 1.0)
    alpha[mask] = np.maximum(alpha[mask], 0.5)
    return alpha.astype(np.float32)


def boundary_ring(mask: np.ndarray, radius: int = 5) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    dilated = cv2.dilate(mask.astype(np.uint8), kernel) > 0
    eroded = cv2.erode(mask.astype(np.uint8), kernel) > 0
    return dilated & ~eroded
