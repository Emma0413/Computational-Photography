from __future__ import annotations

import cv2
import numpy as np


def auto_shadow_mask(rgb: np.ndarray) -> np.ndarray:
    """Estimate a shadow mask from low luminance and saturation/color cues."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    l = lab[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)

    darkness = 255.0 - l
    saturation_penalty = cv2.normalize(s, None, 0, 80, cv2.NORM_MINMAX)
    value_darkness = 255.0 - v
    score = np.clip(0.60 * darkness + 0.30 * value_darkness - 0.10 * saturation_penalty, 0, 255)
    score_u8 = score.astype(np.uint8)
    _, mask = cv2.threshold(score_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return refine_mask(mask > 0)


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
