from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

from .masks import feather_alpha


EPS = 1e-6


def _safe_regions(rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shadow = rgb[mask].astype(np.float32)
    lit = rgb[~mask].astype(np.float32)
    if len(shadow) == 0:
        shadow = rgb.reshape(-1, 3).astype(np.float32)
    if len(lit) < 16:
        lit = rgb.reshape(-1, 3).astype(np.float32)
    return shadow, lit


def _blend(original: np.ndarray, corrected: np.ndarray, mask: np.ndarray, feather: int = 15) -> np.ndarray:
    alpha = feather_alpha(mask, feather)[:, :, None]
    return np.clip(original.astype(np.float32) * (1 - alpha) + corrected.astype(np.float32) * alpha, 0, 255)


def rgb_ratio(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    shadow, lit = _safe_regions(rgb, mask)
    gain = np.median(lit, axis=0) / np.maximum(np.median(shadow, axis=0), EPS)
    gain = np.clip(gain, 0.75, 3.0)
    corrected = rgb.astype(np.float32) * gain[None, None, :]
    return _blend(rgb, corrected, mask)


def lab_l_ratio(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    shadow, lit = _safe_regions(lab, mask)
    gain_l = np.median(lit[:, 0]) / max(float(np.median(shadow[:, 0])), EPS)
    gain_l = float(np.clip(gain_l, 0.8, 2.6))
    chroma_shift = np.median(lit[:, 1:3], axis=0) - np.median(shadow[:, 1:3], axis=0)
    chroma_shift = np.clip(chroma_shift, -12, 12)
    corrected = lab.copy()
    corrected[:, :, 0] *= gain_l
    corrected[:, :, 1:3] += chroma_shift[None, None, :]
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    corrected_rgb = cv2.cvtColor(corrected, cv2.COLOR_LAB2RGB)
    return _blend(rgb, corrected_rgb, mask)


def hsv_value(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    shadow, lit = _safe_regions(hsv, mask)
    gain_v = np.median(lit[:, 2]) / max(float(np.median(shadow[:, 2])), EPS)
    gain_v = float(np.clip(gain_v, 0.8, 2.8))
    sat_gain = np.median(lit[:, 1] + EPS) / np.median(shadow[:, 1] + EPS)
    sat_gain = float(np.clip(sat_gain, 0.75, 1.25))
    corrected = hsv.copy()
    corrected[:, :, 2] *= gain_v
    corrected[:, :, 1] *= sat_gain
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    corrected_rgb = cv2.cvtColor(corrected, cv2.COLOR_HSV2RGB)
    return _blend(rgb, corrected_rgb, mask)


def ycrcb_luma(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    shadow, lit = _safe_regions(ycrcb, mask)
    gain_y = np.median(lit[:, 0]) / max(float(np.median(shadow[:, 0])), EPS)
    gain_y = float(np.clip(gain_y, 0.8, 2.8))
    corrected = ycrcb.copy()
    corrected[:, :, 0] *= gain_y
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    corrected_rgb = cv2.cvtColor(corrected, cv2.COLOR_YCrCb2RGB)
    return _blend(rgb, corrected_rgb, mask)


def mean_std_transfer(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    shadow, lit = _safe_regions(rgb, mask)
    s_mean = shadow.mean(axis=0)
    s_std = np.maximum(shadow.std(axis=0), 1.0)
    l_mean = lit.mean(axis=0)
    l_std = np.maximum(lit.std(axis=0), 1.0)
    corrected = (rgb.astype(np.float32) - s_mean[None, None, :]) * (l_std / s_std)[None, None, :]
    corrected += l_mean[None, None, :]
    return _blend(rgb, corrected, mask)


def linear_regression(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    shadow, lit = _safe_regions(rgb, mask)
    s_mean = shadow.mean(axis=0)
    l_mean = lit.mean(axis=0)
    s_var = np.maximum(shadow.var(axis=0), 1.0)
    scale = np.clip(lit.var(axis=0) / s_var, 0.25, 4.0) ** 0.5
    bias = l_mean - scale * s_mean
    corrected = rgb.astype(np.float32) * scale[None, None, :] + bias[None, None, :]
    return _blend(rgb, corrected, mask)


def _boundary_light_ratio(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    shadow_edge = mask & (cv2.dilate((~mask).astype(np.uint8), kernel) > 0)
    lit_edge = (~mask) & (cv2.dilate(mask.astype(np.uint8), kernel) > 0)
    shadow_pixels = rgb[shadow_edge].astype(np.float32)
    lit_pixels = rgb[lit_edge].astype(np.float32)

    if len(shadow_pixels) < 32 or len(lit_pixels) < 32:
        shadow_pixels, lit_pixels = _safe_regions(rgb, mask)

    shadow_med = np.maximum(np.median(shadow_pixels, axis=0), 1.0)
    lit_med = np.maximum(np.median(lit_pixels, axis=0), 1.0)
    fallback = np.clip(lit_med / shadow_med - 1.0, 0.05, 2.5)

    ratios = []
    for channel in range(3):
        s = shadow_pixels[:, channel]
        l = lit_pixels[:, channel]
        n = int(min(len(s), len(l), 4000))
        if n < 32:
            ratios.append(fallback[channel])
            continue

        s_idx = np.linspace(0, len(s) - 1, n).astype(np.int32)
        l_idx = np.linspace(0, len(l) - 1, n).astype(np.int32)
        votes = l[l_idx] / np.maximum(s[s_idx], 1.0) - 1.0
        votes = votes[np.isfinite(votes)]
        votes = votes[(votes > 0.0) & (votes < 3.0)]
        if len(votes) < 16:
            ratios.append(fallback[channel])
            continue

        hist, edges = np.histogram(votes, bins=np.arange(0.0, 3.05, 0.05))
        mode_idx = int(np.argmax(hist))
        mode = 0.5 * (edges[mode_idx] + edges[mode_idx + 1])
        robust = 0.5 * mode + 0.5 * float(np.median(votes))
        ratios.append(np.clip(robust, 0.05, 2.5))

    return np.array(ratios, dtype=np.float32)


def _global_light_ratio(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    shadow, lit = _safe_regions(rgb, mask)

    # Stronger reference than median-vs-60th percentile.
    shadow_ref = np.maximum(np.percentile(shadow, 35, axis=0), 1.0)
    lit_ref = np.maximum(np.percentile(lit, 70, axis=0), 1.0)

    return np.clip(lit_ref / shadow_ref - 1.0, 0.05, 2.8).astype(np.float32)


def _patch_features(rgb_patch: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return LAB chroma mean, luminance mean, and texture strength."""
    lab = cv2.cvtColor(
        np.clip(rgb_patch, 0, 255).astype(np.uint8),
        cv2.COLOR_RGB2LAB
    ).astype(np.float32)

    l_mean = float(np.mean(lab[:, :, 0]))
    ab_mean = np.mean(lab[:, :, 1:3], axis=(0, 1))

    gray = lab[:, :, 0]
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    texture = float(np.mean(np.sqrt(gx * gx + gy * gy)))

    return ab_mean, l_mean, texture


def _boundary_light_ratio_with_k(rgb: np.ndarray, mask: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Estimate Guo-style direct/environment ratio from boundary patch pairs."""
    fallback = _boundary_light_ratio(rgb, mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    shadow_edge = mask & (cv2.dilate((~mask).astype(np.uint8), kernel) > 0)
    coords = np.column_stack(np.nonzero(shadow_edge))
    if len(coords) < 16:
        return fallback

    max_samples = 3000
    if len(coords) > max_samples:
        sample_idx = np.linspace(0, len(coords) - 1, max_samples).astype(np.int32)
        coords = coords[sample_idx]

    image = rgb.astype(np.float32)
    h, w = mask.shape
    radius = 6
    offset = 10
    votes = []
    dist_inside = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    dist_outside = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    signed = dist_inside - dist_outside
    grad_y = cv2.Sobel(signed, cv2.CV_32F, 0, 1, ksize=3)
    grad_x = cv2.Sobel(signed, cv2.CV_32F, 1, 0, ksize=3)

    for y, x in coords:
        normal = np.array([grad_y[y, x], grad_x[y, x]], dtype=np.float32)
        norm = float(np.linalg.norm(normal))
        if norm < EPS:
            continue
        normal /= norm

        shadow_center = np.rint(np.array([y, x], dtype=np.float32) + offset * normal).astype(np.int32)
        lit_center = np.rint(np.array([y, x], dtype=np.float32) - offset * normal).astype(np.int32)

        sy, sx = int(shadow_center[0]), int(shadow_center[1])
        ly, lx = int(lit_center[0]), int(lit_center[1])
        if sy - radius < 0 or sy + radius + 1 > h or sx - radius < 0 or sx + radius + 1 > w:
            continue
        if ly - radius < 0 or ly + radius + 1 > h or lx - radius < 0 or lx + radius + 1 > w:
            continue

        shadow_mask_patch = mask[sy - radius:sy + radius + 1, sx - radius:sx + radius + 1]
        lit_mask_patch = mask[ly - radius:ly + radius + 1, lx - radius:lx + radius + 1]
        if shadow_mask_patch.mean() < 0.65 or lit_mask_patch.mean() > 0.35:
            continue

        shadow_rgb_patch = image[sy - radius:sy + radius + 1, sx - radius:sx + radius + 1]
        lit_rgb_patch = image[ly - radius:ly + radius + 1, lx - radius:lx + radius + 1]
        shadow_k_patch = k[sy - radius:sy + radius + 1, sx - radius:sx + radius + 1]
        lit_k_patch = k[ly - radius:ly + radius + 1, lx - radius:lx + radius + 1]

        shadow_ab, shadow_l, shadow_tex = _patch_features(shadow_rgb_patch)
        lit_ab, lit_l, lit_tex = _patch_features(lit_rgb_patch)

        chroma_dist = float(np.linalg.norm(shadow_ab - lit_ab))
        if chroma_dist > 14.0:
            continue

        tex_ratio = (lit_tex + 1.0) / (shadow_tex + 1.0)
        if tex_ratio < 0.45 or tex_ratio > 2.2:
            continue

        if lit_l <= shadow_l + 3.0:
            continue

        brightness_ratio = lit_l / max(shadow_l, 1.0)
        if brightness_ratio > 2.6:
            continue

        shadow_rgb = np.maximum(shadow_rgb_patch.mean(axis=(0, 1)), 1.0)
        lit_rgb = np.maximum(lit_rgb_patch.mean(axis=(0, 1)), 1.0)
        shadow_k = float(shadow_k_patch.mean())
        lit_k = float(lit_k_patch.mean())
        if lit_k - shadow_k < 0.20:
            continue

        denom = shadow_rgb * lit_k - lit_rgb * shadow_k
        if np.any(np.abs(denom) < EPS):
            continue
        vote = (lit_rgb - shadow_rgb) / denom
        if np.all(np.isfinite(vote)) and np.all(vote > 0.0) and np.all(vote < 3.5):
            votes.append(vote)

    if len(votes) < 16:
        return fallback

    votes_arr = np.array(votes, dtype=np.float32)
    bins = np.floor(votes_arr / 0.1).astype(np.int32)
    unique_bins, counts = np.unique(bins, axis=0, return_counts=True)
    best_bin = unique_bins[int(np.argmax(counts))]
    in_bin = np.all(bins == best_bin[None, :], axis=1)
    if in_bin.sum() < 8:
        return np.clip(np.median(votes_arr, axis=0), 0.05, 2.5).astype(np.float32)

    return np.clip(np.median(votes_arr[in_bin], axis=0), 0.05, 2.5).astype(np.float32)

def _guided_filter_gray(guide: np.ndarray, src: np.ndarray, radius: int = 18, eps: float = 1e-3) -> np.ndarray:
    guide = guide.astype(np.float32)
    src = src.astype(np.float32)
    ksize = (2 * radius + 1, 2 * radius + 1)
    mean_i = cv2.boxFilter(guide, cv2.CV_32F, ksize, normalize=True, borderType=cv2.BORDER_REFLECT)
    mean_p = cv2.boxFilter(src, cv2.CV_32F, ksize, normalize=True, borderType=cv2.BORDER_REFLECT)
    corr_i = cv2.boxFilter(guide * guide, cv2.CV_32F, ksize, normalize=True, borderType=cv2.BORDER_REFLECT)
    corr_ip = cv2.boxFilter(guide * src, cv2.CV_32F, ksize, normalize=True, borderType=cv2.BORDER_REFLECT)
    var_i = corr_i - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize, normalize=True, borderType=cv2.BORDER_REFLECT)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize, normalize=True, borderType=cv2.BORDER_REFLECT)
    return np.clip(mean_a * guide + mean_b, 0.0, 1.0)


def _image_aware_soft_matte(rgb: np.ndarray, mask: np.ndarray, radius: int = 18) -> np.ndarray:
    initial = feather_alpha(mask, radius=22)
    guide = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    matte = _guided_filter_gray(guide, initial, radius=radius, eps=2e-3)
    matte[mask & (initial > 0.92)] = np.maximum(matte[mask & (initial > 0.92)], 0.92)
    matte[(~mask) & (initial < 0.08)] = np.minimum(matte[(~mask) & (initial < 0.08)], 0.08)
    return np.clip(matte, 0.0, 1.0).astype(np.float32)


def guo_lighting_model(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Paper-inspired Guo/Dai/Hoiem direct-light relighting model."""
    r = _boundary_light_ratio(rgb, mask)
    k = 1.0 - feather_alpha(mask, radius=30)
    factor = (r[None, None, :] + 1.0) / (k[:, :, None] * r[None, None, :] + 1.0)
    corrected = rgb.astype(np.float32) * factor

    lab_original = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_corrected = cv2.cvtColor(np.clip(corrected, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_original[:, :, 0] = lab_corrected[:, :, 0]
    conservative = cv2.cvtColor(np.clip(lab_original, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)

    # Apply the color model inside deep shadow, but keep penumbrae mostly luminance-based.
    alpha = feather_alpha(mask, radius=20)[:, :, None]
    chroma_alpha = np.clip(alpha - 0.35, 0.0, 1.0) / 0.65
    mixed = conservative.astype(np.float32) * (1.0 - chroma_alpha) + corrected.astype(np.float32) * chroma_alpha
    return np.clip(mixed, 0, 255).astype(np.uint8)


def guo_soft_matting(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Guo-style lighting model with an image-aware guided soft matte."""
    alpha = _image_aware_soft_matte(rgb, mask, radius=18)

    # Keep strong shadow, but avoid making every masked pixel full shadow.
    alpha = np.where(mask, np.maximum(alpha, 0.78), alpha).astype(np.float32)

    # Guo k: 1 = non-shadow, 0 = full shadow
    k = 1.0 - alpha

    patch_r = _boundary_light_ratio_with_k(rgb, mask, k)
    r = np.clip(patch_r, 0.05, 1.5).astype(np.float32)

    factor = (r[None, None, :] + 1.0) / (k[:, :, None] * r[None, None, :] + 1.0)
    corrected = rgb.astype(np.float32) * factor

    return np.clip(corrected, 0, 255).astype(np.uint8)


def retinex_shadow_edges(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mask-guided Retinex-style illumination normalization."""
    image = rgb.astype(np.float32) + 1.0
    log_image = np.log(image)
    corrected_log = log_image.copy()
    alpha = feather_alpha(mask, radius=25)

    for sigma, weight in ((15, 0.50), (80, 0.35), (250, 0.15)):
        illum = cv2.GaussianBlur(log_image, (0, 0), sigmaX=sigma, sigmaY=sigma)
        lit_illum = illum[~mask]
        if len(lit_illum) == 0:
            target = np.median(illum.reshape(-1, 3), axis=0)
        else:
            target = np.median(lit_illum, axis=0)
        correction = target[None, None, :] - illum
        corrected_log += weight * alpha[:, :, None] * correction

    corrected = np.exp(corrected_log) - 1.0
    lab_original = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_corrected = cv2.cvtColor(np.clip(corrected, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_original[:, :, 0] = lab_corrected[:, :, 0]
    l_only = cv2.cvtColor(np.clip(lab_original, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    return _blend(rgb, l_only, mask, feather=25)


def anchor_optimization(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Non-shadow anchor relighting with per-channel grid-search gains."""
    image = rgb.astype(np.float32)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l = lab[:, :, 0]
    shadow, lit = _safe_regions(image, mask)
    shadow_l = l[mask]
    lit_l = l[~mask]
    if len(shadow_l) == 0 or len(lit_l) == 0:
        return rgb.copy()

    anchor_l = float(np.percentile(lit_l, 55))
    current_l = float(np.percentile(shadow_l, 55))
    base_l_gain = np.clip(anchor_l / max(current_l, 1.0), 0.8, 2.8)

    target_color = np.percentile(lit, 55, axis=0)
    shadow_color = np.percentile(shadow, 55, axis=0)
    base_gain = np.clip(target_color / np.maximum(shadow_color, 1.0), 0.75, 3.0)

    candidates = []
    for scale in np.linspace(0.75, 1.35, 13):
        gain = np.clip(base_gain * scale, 0.75, 3.2)
        trial_shadow = np.clip(shadow * gain[None, :], 0, 255)
        trial_lab = cv2.cvtColor(trial_shadow.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB).reshape(-1, 3)
        l_error = abs(float(np.percentile(trial_lab[:, 0], 55)) - anchor_l)
        color_error = float(np.linalg.norm(np.median(trial_shadow, axis=0) - target_color))
        overexposure = float(np.mean(trial_shadow > 250.0)) * 40.0
        candidates.append((l_error + 0.08 * color_error + overexposure, gain))

    gain = min(candidates, key=lambda item: item[0])[1]
    l_gain = float(np.clip(base_l_gain, 0.9, 2.4))
    rgb_corrected = image * gain[None, None, :]

    lab_corrected = lab.copy()
    lab_corrected[:, :, 0] *= l_gain
    l_corrected = cv2.cvtColor(np.clip(lab_corrected, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)

    alpha = feather_alpha(mask, radius=20)[:, :, None]
    corrected = 0.55 * rgb_corrected + 0.45 * l_corrected
    return np.clip(image * (1.0 - alpha) + corrected * alpha, 0, 255)


def local_patch_match(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Patch-wise relighting using local non-shadow anchors around each shadow component."""
    image = rgb.astype(np.float32)
    corrected = image.copy()
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))

    for label in range(1, num_labels):
        component = labels == label
        if stats[label, cv2.CC_STAT_AREA] < 64:
            continue
        ring = (cv2.dilate(component.astype(np.uint8), kernel) > 0) & (~mask)
        if ring.sum() < 64:
            ring = ~mask
        component_pixels = image[component]
        ring_pixels = image[ring]
        if len(component_pixels) == 0 or len(ring_pixels) == 0:
            continue
        comp_med = np.maximum(np.median(component_pixels, axis=0), 1.0)
        ring_med = np.maximum(np.median(ring_pixels, axis=0), 1.0)
        gain = np.clip(ring_med / comp_med, 0.75, 3.0)
        corrected[component] = image[component] * gain[None, :]

    return _blend(rgb, corrected, mask, feather=18)


def hybrid_best(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    shadow, lit = _safe_regions(lab, mask)
    gain_l = np.percentile(lit[:, 0], 60) / max(float(np.percentile(shadow[:, 0], 60)), EPS)
    gain_l = float(np.clip(gain_l, 0.9, 2.2))
    chroma_shift = 0.35 * (np.median(lit[:, 1:3], axis=0) - np.median(shadow[:, 1:3], axis=0))
    chroma_shift = np.clip(chroma_shift, -8, 8)
    corrected = lab.copy()
    corrected[:, :, 0] = corrected[:, :, 0] * gain_l
    corrected[:, :, 1:3] = corrected[:, :, 1:3] + chroma_shift[None, None, :]
    corrected_rgb = cv2.cvtColor(np.clip(corrected, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    return _blend(rgb, corrected_rgb, mask, feather=25)


def material_local_hybrid(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Masked hybrid of material-patch Guo relighting and local patch correction."""
    image = rgb.astype(np.float32)
    guo = guo_soft_matting(rgb, mask).astype(np.float32)
    local = local_patch_match(rgb, mask).astype(np.float32)

    mixed = 0.35 * guo + 0.65 * local

    lab_original = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_mixed = cv2.cvtColor(np.clip(mixed, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    shadow_l = lab_original[:, :, 0][mask]
    lit_l = lab_original[:, :, 0][~mask]
    if len(shadow_l) and len(lit_l):
        target_l = float(np.percentile(lit_l, 50))
        max_lift = np.clip(target_l - float(np.percentile(shadow_l, 50)), 12.0, 85.0)
        lab_mixed[:, :, 0] = np.minimum(lab_mixed[:, :, 0], lab_original[:, :, 0] + max_lift)
        mixed = cv2.cvtColor(np.clip(lab_mixed, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)

    alpha = feather_alpha(mask, radius=10)
    alpha[~mask] = 0.0
    alpha[mask] = np.maximum(alpha[mask], 0.75)
    result = image * (1.0 - alpha[:, :, None]) + mixed * alpha[:, :, None]
    return np.clip(result, 0, 255).astype(np.uint8)


METHODS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "rgb_ratio": rgb_ratio,
    "lab_l_ratio": lab_l_ratio,
    "hsv_value": hsv_value,
    "ycrcb_luma": ycrcb_luma,
    "mean_std_transfer": mean_std_transfer,
    "linear_regression": linear_regression,
    "guo_lighting_model": guo_lighting_model,
    "guo_soft_matting": guo_soft_matting,
    "retinex_shadow_edges": retinex_shadow_edges,
    "anchor_optimization": anchor_optimization,
    "local_patch_match": local_patch_match,
    "hybrid_best": hybrid_best,
    "material_local_hybrid": material_local_hybrid,
}
