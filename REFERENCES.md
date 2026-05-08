# References

## Citation-Ready List

1. Ruiqi Guo, Qieyun Dai, and Derek Hoiem. "Single-image shadow detection and removal using paired regions." CVPR, 2011. https://dhoiem.cs.illinois.edu/publications/cvpr11_shadow.pdf

2. Ruiqi Guo, Qieyun Dai, and Derek Hoiem. "Paired regions for shadow detection and removal." IEEE Transactions on Pattern Analysis and Machine Intelligence, 2013. https://experts.illinois.edu/en/publications/paired-regions-for-shadow-detection-and-removal

3. Graham D. Finlayson, Mark S. Drew, and Cheng Lu. "Entropy minimization for shadow removal." International Journal of Computer Vision, 2006. https://research-portal.uea.ac.uk/en/publications/entropy-minimization-for-shadow-removal/

4. Saritha Murali, V. K. Govindan, and Saidalavi Kalady. "Single image shadow removal by optimization using non-shadow anchor values." Computational Visual Media, 2019. https://www.sciopen.com/article/10.1007/s41095-019-0148-x

5. GuoLanqing. "Awesome Shadow Removal." GitHub repository. https://github.com/GuoLanqing/Awesome-Shadow-Removal

6. "A Survey of Deep Learning-based Shadow Removal Methods." ar5iv version. https://ar5iv.labs.arxiv.org/html/2407.08865v2

## Method Mapping

| Our Method | Reference / Inspiration |
| --- | --- |
| `rgb_ratio` | Simple traditional color-ratio baseline |
| `lab_l_ratio` | Color-space luminance correction baseline |
| `hsv_value` | Color-space value-channel relighting baseline |
| `ycrcb_luma` | Color-space luma relighting baseline |
| `mean_std_transfer` | Traditional color transfer baseline |
| `linear_regression` | Affine color correction baseline |
| `guo_lighting_model` | Guo, Dai, and Hoiem paired-region shadow model |
| `retinex_shadow_edges` | Finlayson/Drew/Lu illumination-invariant and Retinex-style shadow attenuation |
| `anchor_optimization` | Murali/Govindan/Kalady non-shadow anchor optimization |
| `local_patch_match` | Local paired-region/anchor correction inspired by Guo et al. and Murali et al. |
| `hybrid_best` | Our study-driven boundary-aware hybrid baseline |

## Report Wording

Use this wording to avoid overclaiming:

> The paper-inspired methods are not exact reproductions of the original implementations. They are mask-guided approximations implemented in a unified OpenCV pipeline so that traditional ideas can be compared under the same dataset, mask, and metric settings.
