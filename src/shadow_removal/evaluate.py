from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from .io import ensure_dir, list_images, matching_path, read_mask, read_rgb, write_rgb
from .masks import MASK_GENERATORS, refine_mask
from .methods import METHODS
from .metrics import compute_metrics
from .visualize import contact_sheet, mask_to_rgb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate traditional shadow removal on a paired dataset.")
    parser.add_argument("--shadow-dir", required=True, help="Directory of shadow images.")
    parser.add_argument("--gt-dir", required=True, help="Directory of shadow-free reference images.")
    parser.add_argument("--mask-dir", help="Optional directory of binary shadow masks.")
    parser.add_argument("--auto-mask", choices=list(MASK_GENERATORS), default="basic", help="Automatic mask generator to use when --mask-dir is omitted.")
    parser.add_argument("--mask-source", choices=["provided", "auto"], default="provided", help="Use provided masks when available, or force automatic mask generation.")
    parser.add_argument("--invert-mask", action="store_true", help="Invert mask polarity after loading.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    return parser.parse_args()


def summarize(rows: list[dict[str, str | float]]) -> list[dict[str, str | float]]:
    numeric_keys = [k for k in rows[0] if k not in {"image", "method"}]
    grouped: dict[str, list[dict[str, str | float]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)

    summary = []
    for method, method_rows in grouped.items():
        out_row: dict[str, str | float] = {"method": method, "n": len(method_rows)}
        for key in numeric_keys:
            values = np.array([float(r[key]) for r in method_rows], dtype=np.float32)
            out_row[f"{key}_mean"] = float(np.nanmean(values))
            out_row[f"{key}_std"] = float(np.nanstd(values))
        summary.append(out_row)
    return summary


def write_csv(path: Path, rows: list[dict[str, str | float]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mask_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    shadow = gt
    non_shadow = ~gt
    tp = float(np.logical_and(pred, shadow).sum())
    fp = float(np.logical_and(pred, non_shadow).sum())
    fn = float(np.logical_and(~pred, shadow).sum())
    tn = float(np.logical_and(~pred, non_shadow).sum())
    shadow_total = max(tp + fn, 1.0)
    non_shadow_total = max(tn + fp, 1.0)
    union = max(tp + fp + fn, 1.0)
    shadow_error = fn / shadow_total
    non_shadow_error = fp / non_shadow_total
    return {
        "iou": tp / union,
        "precision": tp / max(tp + fp, 1.0),
        "recall": tp / shadow_total,
        "specificity": tn / non_shadow_total,
        "ber": 0.5 * (shadow_error + non_shadow_error),
        "pred_area": float(pred.mean()),
        "gt_area": float(gt.mean()),
    }


def main() -> None:
    args = parse_args()
    out = ensure_dir(args.out)
    image_out = ensure_dir(out / "images")
    sheet_out = ensure_dir(out / "contact_sheets")
    mask_out = ensure_dir(out / "masks")

    rows: list[dict[str, str | float]] = []
    mask_rows: list[dict[str, str | float]] = []
    for shadow_path in list_images(args.shadow_dir):
        gt_path = matching_path(args.gt_dir, shadow_path)
        if gt_path is None:
            print(f"Skipping {shadow_path.name}: no matching ground truth.")
            continue

        rgb = read_rgb(shadow_path)
        gt = read_rgb(gt_path)
        if gt.shape[:2] != rgb.shape[:2]:
            raise ValueError(f"Shape mismatch for {shadow_path.name}: input {rgb.shape}, gt {gt.shape}")

        mask_path = matching_path(args.mask_dir, shadow_path) if args.mask_dir else None
        gt_mask = read_mask(mask_path, rgb.shape[:2]) if mask_path else None
        if gt_mask is not None and args.mask_source == "provided":
            mask = gt_mask
        else:
            mask = MASK_GENERATORS[args.auto_mask](rgb)
        if args.invert_mask:
            mask = ~mask
            if gt_mask is not None:
                gt_mask = ~gt_mask
        mask = refine_mask(mask)
        write_rgb(mask_out / shadow_path.name, mask_to_rgb(mask))
        if gt_mask is not None and args.mask_source == "auto":
            mask_row: dict[str, str | float] = {"image": shadow_path.name, "method": args.auto_mask}
            mask_row.update(mask_metrics(mask, refine_mask(gt_mask)))
            mask_rows.append(mask_row)

        sheet_items = [("input", rgb), ("mask", mask_to_rgb(mask)), ("ground_truth", gt)]
        for method_name in args.methods:
            result = np.clip(METHODS[method_name](rgb, mask), 0, 255).astype(np.uint8)
            write_rgb(image_out / method_name / shadow_path.name, result)
            row: dict[str, str | float] = {"image": shadow_path.name, "method": method_name}
            row.update(compute_metrics(result, gt, mask))
            rows.append(row)
            sheet_items.append((method_name, result))

        contact_sheet(sheet_out / f"{shadow_path.stem}_contact.png", sheet_items)

    write_csv(out / "metrics.csv", rows)
    if rows:
        write_csv(out / "summary.csv", summarize(rows))
    write_csv(out / "mask_metrics.csv", mask_rows)
    if mask_rows:
        write_csv(out / "mask_summary.csv", summarize(mask_rows))


if __name__ == "__main__":
    main()
