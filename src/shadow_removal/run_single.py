from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .io import ensure_dir, read_mask, read_rgb, write_rgb
from .masks import MASK_GENERATORS, refine_mask
from .methods import METHODS
from .metrics import compute_metrics
from .visualize import contact_sheet, mask_to_rgb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run traditional shadow removal methods on one image.")
    parser.add_argument("--image", required=True, help="Input shadow image.")
    parser.add_argument("--mask", help="Optional binary shadow mask.")
    parser.add_argument("--auto-mask", choices=list(MASK_GENERATORS), default="basic", help="Automatic mask generator to use when --mask is omitted.")
    parser.add_argument("--mask-source", choices=["provided", "auto"], default="provided", help="Use --mask when available, or force automatic mask generation.")
    parser.add_argument("--invert-mask", action="store_true", help="Invert mask polarity after loading.")
    parser.add_argument("--gt", help="Optional shadow-free reference image.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = ensure_dir(args.out)
    rgb = read_rgb(args.image)
    gt_mask = read_mask(args.mask, rgb.shape[:2]) if args.mask else None
    if gt_mask is not None and args.mask_source == "provided":
        mask = gt_mask
    else:
        mask = MASK_GENERATORS[args.auto_mask](rgb)
    if args.invert_mask:
        mask = ~mask
        if gt_mask is not None:
            gt_mask = ~gt_mask
    mask = refine_mask(mask)
    write_rgb(out / f"{Path(args.image).stem}_mask.png", mask_to_rgb(mask))
    gt = read_rgb(args.gt) if args.gt else None

    outputs: dict[str, np.ndarray] = {}
    rows: list[dict[str, str | float]] = []
    for name in args.methods:
        result = METHODS[name](rgb, mask)
        outputs[name] = np.clip(result, 0, 255).astype(np.uint8)
        write_rgb(out / f"{Path(args.image).stem}_{name}.png", outputs[name])
        if gt is not None:
            row: dict[str, str | float] = {"image": Path(args.image).name, "method": name}
            row.update(compute_metrics(outputs[name], gt, mask))
            rows.append(row)

    sheet_items = [("input", rgb), ("mask", mask_to_rgb(mask))]
    if gt is not None:
        sheet_items.append(("ground_truth", gt))
    sheet_items.extend((name, image) for name, image in outputs.items())
    contact_sheet(out / f"{Path(args.image).stem}_contact.png", sheet_items)

    if rows:
        with (out / "metrics.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    if gt_mask is not None and args.mask_source == "auto":
        shadow = refine_mask(gt_mask)
        non_shadow = ~shadow
        tp = float(np.logical_and(mask, shadow).sum())
        fp = float(np.logical_and(mask, non_shadow).sum())
        fn = float(np.logical_and(~mask, shadow).sum())
        tn = float(np.logical_and(~mask, non_shadow).sum())
        mask_row = {
            "image": Path(args.image).name,
            "method": args.auto_mask,
            "iou": tp / max(tp + fp + fn, 1.0),
            "precision": tp / max(tp + fp, 1.0),
            "recall": tp / max(tp + fn, 1.0),
            "specificity": tn / max(tn + fp, 1.0),
            "ber": 0.5 * (fn / max(tp + fn, 1.0) + fp / max(tn + fp, 1.0)),
            "pred_area": float(mask.mean()),
            "gt_area": float(shadow.mean()),
        }
        with (out / "mask_metrics.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(mask_row.keys()))
            writer.writeheader()
            writer.writerow(mask_row)


if __name__ == "__main__":
    main()
