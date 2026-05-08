from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .io import ensure_dir, list_images, matching_path, read_mask, read_rgb, write_rgb
from .masks import boundary_ring


@dataclass(frozen=True)
class Candidate:
    path: Path
    gt_path: Path
    mask_path: Path | None
    category: str
    score: float
    shadow_area: float
    boundary_contrast: float
    texture_score: float
    saturation_score: float

    @property
    def scene_id(self) -> str:
        return self.path.stem.split("-")[0]


QUOTAS = {
    "hard_shadow": 10,
    "soft_shadow": 10,
    "textured": 5,
    "colored_surface": 5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select a small balanced paired shadow-removal subset.")
    parser.add_argument("--shadow-dir", required=True, help="Directory of shadow input images.")
    parser.add_argument("--gt-dir", required=True, help="Directory of shadow-free ground-truth images.")
    parser.add_argument("--mask-dir", help="Optional directory of binary shadow masks.")
    parser.add_argument("--invert-mask", action="store_true", help="Invert mask polarity after loading.")
    parser.add_argument("--out", default="data/selected", help="Output dataset directory.")
    parser.add_argument("--max-width", type=int, default=640, help="Resize copied images to this max width. Use 0 to keep originals.")
    parser.add_argument("--total", type=int, default=30, help="Total number of images to select.")
    parser.add_argument("--max-per-scene", type=int, default=2, help="Maximum selected images with the same filename prefix before '-'.")
    return parser.parse_args()


def resize_rgb(rgb: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0 or rgb.shape[1] <= max_width:
        return rgb
    scale = max_width / rgb.shape[1]
    return cv2.resize(rgb, (max_width, max(1, int(rgb.shape[0] * scale))), interpolation=cv2.INTER_AREA)


def resize_mask(mask: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0 or mask.shape[1] <= max_width:
        return mask
    scale = max_width / mask.shape[1]
    resized = cv2.resize(
        mask.astype(np.uint8) * 255,
        (max_width, max(1, int(mask.shape[0] * scale))),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized > 127


def auto_rough_mask(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l = lab[:, :, 0]
    threshold = np.percentile(l, 35)
    return l < threshold


def candidate_stats(rgb: np.ndarray, mask: np.ndarray) -> tuple[float, float, float, float]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    shadow_area = float(mask.mean())

    ring = boundary_ring(mask, radius=7)
    if ring.any() and mask.any() and (~mask).any():
        shadow_l = float(gray[mask].mean())
        lit_l = float(gray[~mask].mean())
        boundary_contrast = abs(lit_l - shadow_l)
    else:
        boundary_contrast = 0.0

    lap = cv2.Laplacian(gray, cv2.CV_32F)
    texture_score = float(np.percentile(np.abs(lap[~mask] if (~mask).any() else lap.reshape(-1)), 85))
    saturation_score = float(np.mean(hsv[:, :, 1]) / 255.0)
    return shadow_area, boundary_contrast, texture_score, saturation_score


def categorize(stats: tuple[float, float, float, float]) -> tuple[str, float]:
    shadow_area, boundary_contrast, texture_score, saturation_score = stats
    area_penalty = abs(shadow_area - 0.30)
    if saturation_score > 0.33:
        return "colored_surface", saturation_score - area_penalty
    if texture_score > 18:
        return "textured", texture_score / 50.0 - area_penalty
    if boundary_contrast > 45:
        return "hard_shadow", boundary_contrast / 80.0 - area_penalty
    return "soft_shadow", (45 - min(boundary_contrast, 45)) / 45.0 - area_penalty


def collect_candidates(args: argparse.Namespace) -> list[Candidate]:
    for directory, label in (
        (args.shadow_dir, "shadow input"),
        (args.gt_dir, "shadow-free ground truth"),
        (args.mask_dir, "mask"),
    ):
        if directory is not None and not Path(directory).exists():
            raise FileNotFoundError(f"Missing {label} directory: {directory}")

    candidates = []
    for image_path in list_images(args.shadow_dir):
        gt_path = matching_path(args.gt_dir, image_path)
        if gt_path is None:
            continue

        rgb = read_rgb(image_path)
        mask_path = matching_path(args.mask_dir, image_path) if args.mask_dir else None
        mask = read_mask(mask_path, rgb.shape[:2]) if mask_path else auto_rough_mask(rgb)
        if args.invert_mask:
            mask = ~mask
        stats = candidate_stats(rgb, mask)
        shadow_area, boundary_contrast, texture_score, saturation_score = stats
        if shadow_area < 0.03 or shadow_area > 0.80:
            continue

        category, score = categorize(stats)
        candidates.append(
            Candidate(
                path=image_path,
                gt_path=gt_path,
                mask_path=mask_path,
                category=category,
                score=score,
                shadow_area=shadow_area,
                boundary_contrast=boundary_contrast,
                texture_score=texture_score,
                saturation_score=saturation_score,
            )
        )
    return candidates


def can_add(candidate: Candidate, selected: list[Candidate], max_per_scene: int) -> bool:
    if max_per_scene <= 0:
        return True
    return sum(c.scene_id == candidate.scene_id for c in selected) < max_per_scene


def select_balanced(candidates: list[Candidate], total: int, max_per_scene: int) -> list[Candidate]:
    selected: list[Candidate] = []
    used: set[Path] = set()
    for category, quota in QUOTAS.items():
        category_candidates = sorted(
            (c for c in candidates if c.category == category),
            key=lambda c: c.score,
            reverse=True,
        )
        for candidate in category_candidates:
            if len([c for c in selected if c.category == category]) >= quota:
                break
            if not can_add(candidate, selected, max_per_scene):
                continue
            selected.append(candidate)
            used.add(candidate.path)

    if len(selected) < total:
        remaining = sorted((c for c in candidates if c.path not in used), key=lambda c: c.score, reverse=True)
        for candidate in remaining:
            if len(selected) >= total:
                break
            if can_add(candidate, selected, max_per_scene):
                selected.append(candidate)
    return selected[:total]


def copy_candidate(candidate: Candidate, out: Path, max_width: int, invert_mask: bool) -> None:
    image = resize_rgb(read_rgb(candidate.path), max_width)
    gt = resize_rgb(read_rgb(candidate.gt_path), max_width)
    write_rgb(out / "shadow" / candidate.path.name, image)
    write_rgb(out / "shadow_free" / candidate.path.name, gt)

    if candidate.mask_path is not None:
        mask = resize_mask(read_mask(candidate.mask_path), max_width)
        if invert_mask:
            mask = ~mask
        mask_rgb = np.repeat((mask.astype(np.uint8) * 255)[:, :, None], 3, axis=2)
        write_rgb(out / "mask" / candidate.path.name, mask_rgb)
    else:
        ensure_dir(out / "mask")


def main() -> None:
    args = parse_args()
    out = ensure_dir(args.out)
    for subdir in ("shadow", "shadow_free", "mask"):
        ensure_dir(out / subdir)

    candidates = collect_candidates(args)
    selected = select_balanced(candidates, args.total, args.max_per_scene)
    if not selected:
        raise RuntimeError(
            "No paired candidates found. Check that shadow, ground-truth, and mask folders exist "
            "and use matching filenames. For ISTD this usually means train_A/test_A for shadows, "
            "train_B/test_B for masks, and train_C/test_C or train_C_fixed_ours/test_C_fixed_official "
            "for shadow-free images."
        )

    for candidate in selected:
        copy_candidate(candidate, out, args.max_width, args.invert_mask)

    with (out / "categories.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "category",
                "shadow_area",
                "boundary_contrast",
                "texture_score",
                "saturation_score",
            ],
        )
        writer.writeheader()
        for candidate in selected:
            writer.writerow(
                {
                    "image": candidate.path.name,
                    "category": candidate.category,
                    "shadow_area": f"{candidate.shadow_area:.4f}",
                    "boundary_contrast": f"{candidate.boundary_contrast:.4f}",
                    "texture_score": f"{candidate.texture_score:.4f}",
                    "saturation_score": f"{candidate.saturation_score:.4f}",
                }
            )

    print(f"Selected {len(selected)} images from {len(candidates)} paired candidates into {out}")
    print("Category counts:")
    for category in QUOTAS:
        print(f"  {category}: {sum(c.category == category for c in selected)}")


if __name__ == "__main__":
    main()
