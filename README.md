# Traditional Single-Image Shadow Removal

This repository implements and evaluates traditional, non-learning shadow removal methods for paired single-image shadow removal. The project was built for a computational photography final project focused on comparing classical color-space, relighting, Retinex, and local-region methods.

The main goal is not to beat modern deep learning methods. Instead, the project studies how far traditional methods can go, where they fail, and whether more structured illumination models improve over simple global color correction.

## Features

- 11 traditional shadow removal methods.
- Paired-image evaluation on ISTD-style datasets.
- Optional automatic mask estimation when masks are unavailable.
- Metrics over full image, shadow region, non-shadow region, and shadow boundary.
- RGB and LAB RMSE reporting for comparison with shadow-removal literature.
- Contact-sheet visualizations for qualitative comparison.
- Dataset subset selection for large datasets.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset Format

Use paired shadow images, shadow-free ground truth images, and binary masks:

```text
data/selected/
  shadow/
    img_001.png
  shadow_free/
    img_001.png
  mask/
    img_001.png
```

The filenames must match across the three folders. Masks should be white for shadow pixels and black for non-shadow pixels. If your dataset uses the opposite convention, pass `--invert-mask`.

## ISTD Subset Selection

For a full ISTD-style dataset:

```text
ISTD_Dataset/
  train/
    train_A/   # shadow images
    train_B/   # masks
    train_C/   # shadow-free ground truth
```

Select a smaller diverse subset:

```bash
python -m src.shadow_removal.select_subset \
  --shadow-dir ./ISTD_Dataset/train/train_A \
  --gt-dir ./ISTD_Dataset/train/train_C \
  --mask-dir ./ISTD_Dataset/train/train_B \
  --out data/selected \
  --total 30 \
  --max-width 640 \
  --max-per-scene 2
```

The selector writes:

```text
data/selected/shadow/
data/selected/shadow_free/
data/selected/mask/
data/selected/categories.csv
```

## Run Evaluation

Evaluate all implemented methods:

```bash
python -m src.shadow_removal.evaluate \
  --shadow-dir data/selected/shadow \
  --gt-dir data/selected/shadow_free \
  --mask-dir data/selected/mask \
  --out outputs/selected_eval
```

Outputs:

```text
outputs/selected_eval/metrics.csv          # per-image, per-method metrics
outputs/selected_eval/summary.csv          # mean/std summary by method
outputs/selected_eval/images/              # restored images
outputs/selected_eval/contact_sheets/      # visual comparison grids
```

## Run One Image

```bash
python -m src.shadow_removal.run_single \
  --image data/selected/shadow/img_001.png \
  --mask data/selected/mask/img_001.png \
  --gt data/selected/shadow_free/img_001.png \
  --out outputs/single
```

## Implemented Methods

Simple baselines:

1. `rgb_ratio`: per-channel RGB gain from lit/shadow statistics.
2. `lab_l_ratio`: LAB luminance correction with chroma offset.
3. `hsv_value`: HSV value-channel relighting.
4. `ycrcb_luma`: YCrCb luma-channel relighting.
5. `mean_std_transfer`: mean/std color transfer from shadow to lit regions.
6. `linear_regression`: affine per-channel color correction.

More structured traditional methods:

7. `guo_lighting_model`: Guo/Dai/Hoiem-inspired boundary lighting-ratio model.
8. `retinex_shadow_edges`: mask-guided Retinex-style illumination normalization.
9. `anchor_optimization`: non-shadow anchor-based correction.
10. `local_patch_match`: local paired-region correction from nearby non-shadow pixels.

Study-driven hybrid:

11. `hybrid_best`: LAB luminance correction with conservative chroma transfer and feathered blending.

The paper-inspired methods are approximations implemented in a unified OpenCV pipeline, not exact reproductions of the original authors' code.

## Metrics

The evaluator reports:

- RGB MAE/RMSE.
- LAB MAE/RMSE.
- PSNR and SSIM.
- Shadow-region error.
- Non-shadow-region error.
- Boundary-ring error.

LAB RMSE is the primary metric for literature comparison. RGB metrics and contact sheets are used for visual interpretation.

## Current Results

The latest checked run is:

```text
data/selected_diverse/
outputs/selected_diverse_eval_complex/
```

Summary:

| Metric | Best Method | Value |
| --- | --- | ---: |
| LAB RMSE all | `local_patch_match` | 15.16 |
| LAB RMSE shadow | `hsv_value` | 17.48 |
| LAB boundary RMSE | `local_patch_match` | 14.82 |
| RGB RMSE all | `guo_lighting_model` | 25.14 |
| SSIM | `guo_lighting_model` | 0.907 |

These results show that local/boundary-aware traditional methods outperform simple global color-space corrections on the selected ISTD subset.

## Project Documents

- [PROJECT_STEPS.md](PROJECT_STEPS.md): experiment plan and final report checklist.
- [REPORT_OUTLINE.md](REPORT_OUTLINE.md): suggested final report structure.
- [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md): current result summary.
- [REFERENCES.md](REFERENCES.md): citation-ready references and method mapping.

## Notes For Git

Large datasets and generated outputs are ignored by `.gitignore`:

```text
ISTD_Dataset/
data/
outputs/
*.zip
```

Commit the source code, requirements, and project documentation, but do not commit the full ISTD dataset or generated output images.
