# Traditional Single-Image Shadow Removal

This project implements and evaluates traditional, non-learning shadow removal methods. It focuses on paired single-image shadow removal using color-space correction, statistical relighting, Retinex-style illumination normalization, and Guo-style boundary/material-patch relighting.

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

Filenames must match across the three folders. Masks should be white for shadow pixels and black for non-shadow pixels. If a dataset uses the opposite convention, pass `--invert-mask`.

## Select An ISTD Subset

For an ISTD-style dataset organized as:

```text
ISTD_Dataset/
  train/
    train_A/   # shadow images
    train_B/   # masks
    train_C/   # shadow-free ground truth
```

select a smaller subset with:

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

## Run Evaluation

Evaluate all implemented methods on a paired dataset:

```bash
python -m src.shadow_removal.evaluate \
  --shadow-dir data/selected/shadow \
  --gt-dir data/selected/shadow_free \
  --mask-dir data/selected/mask \
  --out outputs/selected_eval
```

The evaluator writes:

```text
outputs/selected_eval/metrics.csv
outputs/selected_eval/summary.csv
outputs/selected_eval/contact_sheets/
```

If masks are unavailable, omit `--mask-dir` and choose an automatic detector:

```bash
python -m src.shadow_removal.evaluate \
  --shadow-dir data/selected/shadow \
  --gt-dir data/selected/shadow_free \
  --auto-mask basic \
  --out outputs/selected_eval_auto
```

Available automatic masks:

- `basic`: LAB/HSV darkness cue.
- `guo`: Guo-inspired region test.
- `istd`: detector calibrated for the local selected ISTD subset.

## Run One Image

With a provided mask:

```bash
python -m src.shadow_removal.run_single \
  --image data/selected/shadow/img_001.png \
  --mask data/selected/mask/img_001.png \
  --gt data/selected/shadow_free/img_001.png \
  --out outputs/single
```

With an automatic mask:

```bash
python -m src.shadow_removal.run_single \
  --image data/Self/example.jpg \
  --auto-mask basic \
  --mask-source auto \
  --out outputs/single_self
```

## Methods

Simple color-space baselines:

1. `rgb_ratio`
2. `lab_l_ratio`
3. `hsv_value`
4. `ycrcb_luma`

Statistical and local correction methods:

5. `mean_std_transfer`
6. `linear_regression`
7. `anchor_optimization`
8. `local_patch_match`
9. `hybrid_best`
10. `material_local_hybrid`

Paper-inspired illumination methods:

11. `guo_lighting_model`
12. `guo_soft_matting`
13. `retinex_shadow_edges`

`guo_soft_matting` is a Guo-inspired approximation. It uses guided-filter soft matting and material-patch lighting-ratio voting, not the original paper's full closed-form matting and graph pipeline.

`material_local_hybrid` is the final hybrid method. It combines Guo-style material-patch relighting with local masked correction while keeping non-shadow pixels unchanged outside the mask.

## Metrics

The evaluation reports RGB and LAB MAE/RMSE, PSNR, SSIM, shadow-region error, non-shadow-region error, and boundary-ring error. Quantitative metrics are computed against paired shadow-free ground truth images.

##Results
See Drive link: https://drive.google.com/drive/folders/1kq8qq50tzCwGtLQAblMJS2iow3kyaslH?usp=drive_link
