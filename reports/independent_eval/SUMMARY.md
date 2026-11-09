# Independent test: summary (git commit 4d0ac63, uncommitted changes possible)

Lakes scored: 50; models: gloid_2022, ndwi_0.3, ndwi_otsu, rf_0.5, rf_0.75, sar_-15dB, sar_otsu, unet, unet_optical_only, unet_sar_only. Reference: Third Pole 2022 manual polygons. Checkpoint models/seg_fused_test.pt, threshold 0.5, 400 m buffer. Mean IoU with 95 % bootstrap CI over lakes.

## Overall
| model             | group   |   n |   mean_iou |   ci_lo |   ci_hi |   median_iou |   mean_f1 |   median_abs_area_err_pct |
|:------------------|:--------|----:|-----------:|--------:|--------:|-------------:|----------:|--------------------------:|
| gloid_2022        | all     |  15 |      0.867 |   0.819 |   0.911 |        0.906 |     0.926 |                     2.146 |
| ndwi_0.3          | all     |  50 |      0.661 |   0.568 |   0.751 |        0.807 |     0.732 |                    16.258 |
| ndwi_otsu         | all     |  50 |      0.469 |   0.371 |   0.566 |        0.409 |     0.56  |                   106.292 |
| rf_0.5            | all     |  50 |      0.814 |   0.745 |   0.869 |        0.897 |     0.872 |                     6.914 |
| rf_0.75           | all     |  50 |      0.809 |   0.735 |   0.87  |        0.893 |     0.864 |                     4.562 |
| sar_-15dB         | all     |  50 |      0.673 |   0.603 |   0.736 |        0.75  |     0.772 |                    15.468 |
| sar_otsu          | all     |  50 |      0.407 |   0.338 |   0.479 |        0.356 |     0.533 |                   180.668 |
| unet              | all     |  50 |      0.843 |   0.79  |   0.887 |        0.901 |     0.9   |                     5.265 |
| unet_optical_only | all     |  50 |      0.786 |   0.703 |   0.857 |        0.899 |     0.836 |                     5.678 |
| unet_sar_only     | all     |  50 |      0.752 |   0.674 |   0.82  |        0.834 |     0.817 |                     8.706 |

## U-Net vs each baseline (paired, per-lake IoU)
| vs                |   n_lakes |   mean_diff_iou |   unet_wins |   wilcoxon_p |
|:------------------|----------:|----------------:|------------:|-------------:|
| gloid_2022        |        15 |          -0.033 |           6 |        0.277 |
| ndwi_0.3          |        50 |           0.182 |          36 |        0     |
| ndwi_otsu         |        50 |           0.374 |          44 |        0     |
| rf_0.5            |        50 |           0.029 |          30 |        0.117 |
| rf_0.75           |        50 |           0.034 |          26 |        0.247 |
| sar_-15dB         |        50 |           0.171 |          46 |        0     |
| sar_otsu          |        50 |           0.436 |          48 |        0     |
| unet_optical_only |        50 |           0.057 |          23 |        0.368 |
| unet_sar_only     |        50 |           0.091 |          41 |        0     |

## By size_class
| model             | group   |   n |   mean_iou |   ci_lo |   ci_hi |   median_iou |   mean_f1 |   median_abs_area_err_pct |
|:------------------|:--------|----:|-----------:|--------:|--------:|-------------:|----------:|--------------------------:|
| ndwi_0.3          | L       |  10 |      0.699 |   0.492 |   0.875 |        0.868 |     0.772 |                    12.397 |
| ndwi_0.3          | M       |  10 |      0.636 |   0.396 |   0.841 |        0.799 |     0.699 |                     6.132 |
| ndwi_0.3          | S       |  13 |      0.616 |   0.419 |   0.789 |        0.805 |     0.683 |                    12.235 |
| ndwi_0.3          | XL      |   9 |      0.786 |   0.562 |   0.951 |        0.944 |     0.827 |                     3.528 |
| ndwi_0.3          | XS      |   8 |      0.579 |   0.399 |   0.702 |        0.633 |     0.695 |                    33.632 |
| rf_0.5            | L       |  10 |      0.931 |   0.911 |   0.948 |        0.938 |     0.964 |                     2.748 |
| rf_0.5            | M       |  10 |      0.865 |   0.828 |   0.9   |        0.89  |     0.927 |                     9.462 |
| rf_0.5            | S       |  13 |      0.657 |   0.487 |   0.801 |        0.718 |     0.744 |                    29.609 |
| rf_0.5            | XL      |   9 |      0.946 |   0.929 |   0.962 |        0.951 |     0.972 |                     4.337 |
| rf_0.5            | XS      |   8 |      0.711 |   0.495 |   0.849 |        0.774 |     0.783 |                    11.599 |
| sar_-15dB         | L       |  10 |      0.801 |   0.74  |   0.86  |        0.834 |     0.886 |                     5.037 |
| sar_-15dB         | M       |  10 |      0.717 |   0.628 |   0.79  |        0.742 |     0.827 |                    16.087 |
| sar_-15dB         | S       |  13 |      0.563 |   0.42  |   0.69  |        0.595 |     0.68  |                    21.778 |
| sar_-15dB         | XL      |   9 |      0.833 |   0.746 |   0.904 |        0.866 |     0.903 |                     5.432 |
| sar_-15dB         | XS      |   8 |      0.456 |   0.247 |   0.664 |        0.477 |     0.561 |                    50.936 |
| unet              | L       |  10 |      0.937 |   0.923 |   0.95  |        0.937 |     0.967 |                     1.132 |
| unet              | M       |  10 |      0.894 |   0.869 |   0.918 |        0.911 |     0.943 |                     4.588 |
| unet              | S       |  13 |      0.759 |   0.65  |   0.847 |        0.812 |     0.848 |                    19.34  |
| unet              | XL      |   9 |      0.953 |   0.93  |   0.973 |        0.972 |     0.976 |                     1.016 |
| unet              | XS      |   8 |      0.678 |   0.468 |   0.815 |        0.767 |     0.762 |                    17.787 |
| unet_optical_only | L       |  10 |      0.937 |   0.918 |   0.955 |        0.94  |     0.967 |                     2.435 |
| unet_optical_only | M       |  10 |      0.763 |   0.557 |   0.909 |        0.9   |     0.819 |                     7.261 |
| unet_optical_only | S       |  13 |      0.64  |   0.435 |   0.817 |        0.796 |     0.697 |                    20.117 |
| unet_optical_only | XL      |   9 |      0.951 |   0.918 |   0.974 |        0.961 |     0.974 |                     0.718 |
| unet_optical_only | XS      |   8 |      0.677 |   0.467 |   0.821 |        0.725 |     0.761 |                    15.262 |
| unet_sar_only     | L       |  10 |      0.874 |   0.839 |   0.905 |        0.878 |     0.932 |                     6.045 |
| unet_sar_only     | M       |  10 |      0.786 |   0.706 |   0.849 |        0.793 |     0.875 |                    11.301 |
| unet_sar_only     | S       |  13 |      0.684 |   0.501 |   0.83  |        0.817 |     0.754 |                    14.852 |
| unet_sar_only     | XL      |   9 |      0.931 |   0.912 |   0.95  |        0.932 |     0.964 |                     2.079 |
| unet_sar_only     | XS      |   8 |      0.467 |   0.204 |   0.71  |        0.64  |     0.539 |                    39.837 |

## By region_group
| model             | group     |   n |   mean_iou |   ci_lo |   ci_hi |   median_iou |   mean_f1 |   median_abs_area_err_pct |
|:------------------|:----------|----:|-----------:|--------:|--------:|-------------:|----------:|--------------------------:|
| ndwi_0.3          | core      |  40 |      0.666 |   0.561 |   0.763 |        0.813 |     0.734 |                    13.617 |
| ndwi_0.3          | geo_shift |  10 |      0.644 |   0.436 |   0.828 |        0.703 |     0.722 |                    29.486 |
| rf_0.5            | core      |  40 |      0.817 |   0.743 |   0.876 |        0.897 |     0.874 |                     6.848 |
| rf_0.5            | geo_shift |  10 |      0.804 |   0.634 |   0.911 |        0.884 |     0.863 |                     9.43  |
| sar_-15dB         | core      |  40 |      0.671 |   0.594 |   0.744 |        0.773 |     0.769 |                    15.914 |
| sar_-15dB         | geo_shift |  10 |      0.678 |   0.529 |   0.806 |        0.724 |     0.783 |                     6.097 |
| unet              | core      |  40 |      0.839 |   0.775 |   0.892 |        0.911 |     0.894 |                     5.265 |
| unet              | geo_shift |  10 |      0.861 |   0.803 |   0.91  |        0.894 |     0.923 |                     7.377 |
| unet_optical_only | core      |  40 |      0.788 |   0.693 |   0.869 |        0.9   |     0.836 |                     5.678 |
| unet_optical_only | geo_shift |  10 |      0.778 |   0.581 |   0.907 |        0.842 |     0.832 |                     6.142 |
| unet_sar_only     | core      |  40 |      0.767 |   0.687 |   0.836 |        0.854 |     0.834 |                     8.538 |
| unet_sar_only     | geo_shift |  10 |      0.695 |   0.451 |   0.878 |        0.83  |     0.749 |                     9.091 |

## By dam_type
| model             | group   |   n |   mean_iou |   ci_lo |   ci_hi |   median_iou |   mean_f1 |   median_abs_area_err_pct |
|:------------------|:--------|----:|-----------:|--------:|--------:|-------------:|----------:|--------------------------:|
| ndwi_0.3          | extra   |  17 |      0.742 |   0.61  |   0.854 |        0.809 |     0.818 |                    12.235 |
| ndwi_0.3          | icedam  |   1 |      0     | nan     | nan     |        0     |     0     |                   100     |
| ndwi_0.3          | peri    |  13 |      0.525 |   0.314 |   0.731 |        0.652 |     0.591 |                    22.186 |
| ndwi_0.3          | pro     |  16 |      0.708 |   0.546 |   0.843 |        0.845 |     0.777 |                    12.484 |
| ndwi_0.3          | supra   |   3 |      0.76  |   0.631 |   0.832 |        0.817 |     0.86  |                    17.726 |
| rf_0.5            | extra   |  17 |      0.863 |   0.821 |   0.901 |        0.887 |     0.924 |                     8.599 |
| rf_0.5            | icedam  |   1 |      0     | nan     | nan     |        0     |     0     |                   100     |
| rf_0.5            | peri    |  13 |      0.82  |   0.684 |   0.92  |        0.913 |     0.877 |                     5.867 |
| rf_0.5            | pro     |  16 |      0.82  |   0.69  |   0.916 |        0.904 |     0.87  |                     6.914 |
| rf_0.5            | supra   |   3 |      0.749 |   0.616 |   0.872 |        0.759 |     0.852 |                    15.328 |
| sar_-15dB         | extra   |  17 |      0.757 |   0.678 |   0.824 |        0.801 |     0.852 |                    10.057 |
| sar_-15dB         | icedam  |   1 |      0.025 | nan     | nan     |        0.025 |     0.048 |                   916.598 |
| sar_-15dB         | peri    |  13 |      0.713 |   0.577 |   0.808 |        0.767 |     0.802 |                     7.24  |
| sar_-15dB         | pro     |  16 |      0.607 |   0.473 |   0.732 |        0.669 |     0.715 |                    19.278 |
| sar_-15dB         | supra   |   3 |      0.589 |   0.381 |   0.793 |        0.593 |     0.727 |                    18.478 |
| unet              | extra   |  17 |      0.883 |   0.85  |   0.914 |        0.898 |     0.936 |                     6.277 |
| unet              | icedam  |   1 |      0     | nan     | nan     |        0     |     0     |                   100     |
| unet              | peri    |  13 |      0.875 |   0.821 |   0.924 |        0.921 |     0.93  |                     2.475 |
| unet              | pro     |  16 |      0.833 |   0.728 |   0.919 |        0.923 |     0.894 |                     4.502 |
| unet              | supra   |   3 |      0.816 |   0.744 |   0.86  |        0.845 |     0.898 |                    13.043 |
| unet_optical_only | extra   |  17 |      0.871 |   0.806 |   0.92  |        0.915 |     0.926 |                     4.386 |
| unet_optical_only | icedam  |   1 |      0     | nan     | nan     |        0     |     0     |                   100     |
| unet_optical_only | peri    |  13 |      0.731 |   0.538 |   0.888 |        0.907 |     0.781 |                     5.436 |
| unet_optical_only | pro     |  16 |      0.791 |   0.622 |   0.92  |        0.921 |     0.829 |                     3.68  |
| unet_optical_only | supra   |   3 |      0.781 |   0.696 |   0.851 |        0.796 |     0.876 |                    19.255 |
| unet_sar_only     | extra   |  17 |      0.795 |   0.675 |   0.88  |        0.817 |     0.86  |                     7.484 |
| unet_sar_only     | icedam  |   1 |      0     | nan     | nan     |        0     |     0     |                   100     |
| unet_sar_only     | peri    |  13 |      0.791 |   0.647 |   0.883 |        0.83  |     0.851 |                     8.36  |
| unet_sar_only     | pro     |  16 |      0.732 |   0.571 |   0.86  |        0.871 |     0.796 |                     7.467 |
| unet_sar_only     | supra   |   3 |      0.698 |   0.622 |   0.813 |        0.658 |     0.819 |                    30.769 |

## 5 worst U-Net lakes
| lake_id   | size_class   | dam_type   | region_group   |   area_ref_km2 |   iou_target |   area_err_pct |
|:----------|:-------------|:-----------|:---------------|---------------:|-------------:|---------------:|
| TP_3509   | XS           | icedam     | core           |          0.024 |        0     |       -100     |
| TP_3741   | S            | pro        | core           |          0.072 |        0.288 |        -71.229 |
| TP_4330   | S            | pro        | core           |          0.139 |        0.45  |        -54.578 |
| TP_3158   | XS           | pro        | geo_shift      |          0.034 |        0.654 |        -33.628 |
| TP_4473   | S            | peri       | core           |          0.055 |        0.669 |         41.742 |

Overlays: reports/qa/independent/. Figures: iou_vs_area.png, iou_by_size_class.png.

## Commands
```
python src/ingestion/tp_fetch.py --workers 4
python src/preprocessing/gloid_stack_tile.py --lakes-csv configs/independent_test_lakes.csv --id-col lake_id --skip-tiles
python scripts/tp_to_masks.py
python scripts/train_rf_baseline.py
python scripts/eval_independent.py
python scripts/independent_stats.py
```

## Run notes (written by hand; appended to SUMMARY.md by scripts/independent_stats.py)

Run on 2026-11-09 by Diya Manu, code at commit 4d0ac63, Windows, CPU, Python 3.12.

### What was run, and how it differs from the command list above
- All six steps were run as listed, with one exception: `scripts/train_rf_baseline.py` was **not** rerun. The RF model `models/rf_baseline.joblib` was supplied by Aparna. It was checked before scoring: `RandomForestClassifier(n_estimators=200, random_state=0)`, trained on exactly the 22 `train_pool` lakes of `kaggle_dataset/splits.json`, 400,465 balanced samples.
- No lake was replaced (`reports/independent_replacements.csv` has a header only). All 50 primary lakes downloaded on the first run (200 of 200 manifest rows `ok`), all passed the quality gate (min valid VV 1.00, min valid NDWI 0.92, so the U-Net used optical input on every lake), and all reference areas matched the CSV within 1.2 %.
- `select_independent_lakes.py` reproduced `configs/independent_test_lakes.csv` exactly (no diff).
- Reproducibility check: lake TP_3438 was deleted and rebuilt from scratch (download, stack, masks, scoring). Imagery, stack, masks, prediction and all 8 scores were bit-identical to the first run.

### Ablations (`unet_sar_only`, `unet_optical_only`)
- Trained on Kaggle (GPU T4) with `kaggle/kaggle_ablations.ipynb`: `train_segmentation.py --split test --channels sar|optical --epochs 40 --batch-size 16 --lr 3e-4`, same settings and same 22 pool lakes as `seg_fused_test.pt`. Both ran the full 40 epochs (best epochs 28 and 25). Metrics files: `models/seg_sar_test_metrics.json`, `models/seg_optical_test_metrics.json`.
- Result: fused beats SAR-only clearly (+0.091 mean IoU, 41/50 lakes, p < 0.001). Fused vs optical-only is **not** significant (+0.057 mean IoU, 23/50 lakes, p = 0.37; medians 0.901 vs 0.899). The mean gap comes from a few lakes where optical-only fails (IoU < 0.5) and fused does not: TP_3070, TP_3608, TP_4706. Describe this as "fusion avoids occasional large failures", not "fusion is better".
- All 50 lakes had clear optical composites, so this test cannot show the cloud benefit of SAR. That needs the cloud stress test (Milestone 8).

### Acceptance checks (doc section 9)
1. Distance: nearest training lake to any independent lake is 3.7 km (asserted by `eval_independent.py`).
2. Overlap with training labels: checked explicitly (not coded in the scripts). No target reference polygon intersects any of the 239 GLOID training-label polygons (30 training lakes, all years); minimum edge-to-edge distance 2.7 km.
   - Caveat: the analysis window of **TP_4431** contains part of a training-lake polygon. It is far outside the 400 m target buffer, so `iou_target` is unaffected, but that lake's whole-window score (`iou_window`) is slightly contaminated.
3. Checkpoint `seg_fused_test.pt`, split `test`, channels `fused` (asserted).
4. Every primary lake scored, none replaced or dropped (`check 4: OK`).
5. Thresholds, buffer, metrics and models were fixed in the scripts before scoring and not changed.
6. Reference areas within 5 % (`check 6: OK`, largest difference about 1.2 %).

### Observations for the write-up
- **TP_3509** (XS, ice-dammed, 0.024 km2): IoU 0 for the U-Net and every baseline; no water visible in the 2022 Aug-Oct composite, so the lake was probably drained or frozen at image time. Kept in the results, as required.
- Overlays of typical lakes show most U-Net errors are one-pixel boundary errors. Some predicted water outside the target is on other inventory lakes, but some is on small water-like patches not in the inventory (small ponds below the inventory size limit, or false positives). These count as errors where they fall inside the 400 m buffer.
- U-Net vs RF is not significant (p = 0.12 at threshold 0.5, 0.25 at 0.75); U-Net vs GLOID on its 15 lakes is not significant either (-0.033, p = 0.28).

### Reference data licence and citation
Third Pole glacial lake inventory: Zhang, T., Wang, W., An, B., Wei, L. (2023). *The inventory of glacial lakes and results of risk assessment in the Third Pole* (V1.0). Zenodo. https://doi.org/10.5281/zenodo.8369314. Licence: CC BY 4.0. Associated paper: Zhang et al. (2023), *Nature Communications*, "Enhanced glacial lake activity threatens numerous communities and infrastructure in the Third Pole" (look up the article DOI before citing).