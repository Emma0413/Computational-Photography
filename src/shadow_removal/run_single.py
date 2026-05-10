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
    parser.add_argument("--invert-mask", action="store_true", help="Invert mask polarity after loading.")
    parser.add_argument("--gt", help="Optional shadow-free reference image.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = ensure_dir(args.out)
    rgb = read_rgb(args.image)
    mask = read_mask(args.mask, rgb.shape[:2]) if args.mask else MASK_GENERATORS[args.auto_mask](rgb)
    if args.invert_mask:
        mask = ~mask
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


if __name__ == "__main__":
    main()
