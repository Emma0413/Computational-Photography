from __future__ import annotations

import math

import cv2
import numpy as np

from .masks import boundary_ring


def mae(pred: np.ndarray, gt: np.ndarray, region: np.ndarray | None = None) -> float:
    diff = np.abs(pred.astype(np.float32) - gt.astype(np.float32))
    if region is not None:
        diff = diff[region]
    return float(diff.mean()) if diff.size else float("nan")


def rmse(pred: np.ndarray, gt: np.ndarray, region: np.ndarray | None = None) -> float:
    diff = (pred.astype(np.float32) - gt.astype(np.float32)) ** 2
    if region is not None:
        diff = diff[region]
    return float(np.sqrt(diff.mean())) if diff.size else float("nan")


def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    value = rmse(pred, gt)
    if value <= 0:
        return float("inf")
    return float(20 * math.log10(255.0 / value))


def rgb_to_lab_float(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)


def ssim_gray(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_gray = cv2.cvtColor(pred.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    gt_gray = cv2.cvtColor(gt.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu_x = cv2.GaussianBlur(pred_gray, (11, 11), 1.5)
    mu_y = cv2.GaussianBlur(gt_gray, (11, 11), 1.5)
    sigma_x = cv2.GaussianBlur(pred_gray * pred_gray, (11, 11), 1.5) - mu_x * mu_x
    sigma_y = cv2.GaussianBlur(gt_gray * gt_gray, (11, 11), 1.5) - mu_y * mu_y
    sigma_xy = cv2.GaussianBlur(pred_gray * gt_gray, (11, 11), 1.5) - mu_x * mu_y
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    return float((numerator / np.maximum(denominator, 1e-6)).mean())


def compute_metrics(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    non_shadow = ~mask
    boundary = boundary_ring(mask, radius=5)
    pred_lab = rgb_to_lab_float(pred)
    gt_lab = rgb_to_lab_float(gt)
    return {
        "mae_all": mae(pred, gt),
        "rmse_all": rmse(pred, gt),
        "psnr_all": psnr(pred, gt),
        "ssim_all": ssim_gray(pred, gt),
        "mae_shadow": mae(pred, gt, mask),
        "rmse_shadow": rmse(pred, gt, mask),
        "mae_non_shadow": mae(pred, gt, non_shadow),
        "rmse_non_shadow": rmse(pred, gt, non_shadow),
        "mae_boundary": mae(pred, gt, boundary),
        "rmse_boundary": rmse(pred, gt, boundary),
        "lab_mae_all": mae(pred_lab, gt_lab),
        "lab_rmse_all": rmse(pred_lab, gt_lab),
        "lab_mae_shadow": mae(pred_lab, gt_lab, mask),
        "lab_rmse_shadow": rmse(pred_lab, gt_lab, mask),
        "lab_mae_non_shadow": mae(pred_lab, gt_lab, non_shadow),
        "lab_rmse_non_shadow": rmse(pred_lab, gt_lab, non_shadow),
        "lab_mae_boundary": mae(pred_lab, gt_lab, boundary),
        "lab_rmse_boundary": rmse(pred_lab, gt_lab, boundary),
    }
