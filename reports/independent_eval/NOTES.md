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
