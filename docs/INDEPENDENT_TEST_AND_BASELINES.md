# Independent test set (50 lakes) and baselines: step-by-step

Handoff for the person continuing this work. Read `docs/HANDOFF.md` first for project state.

## 0. Goal and what "independent" means

The segmentation model (U-Net, EfficientNet-B3, 10 input channels) was trained on **GLOID** polygons from 22 lakes. GLOID is itself a model product, so scoring against GLOID only measures agreement with another model. We need accuracy against a **different, manually delineated** reference.

- **Reference labels:** the Third Pole glacial lake inventory (manual Sentinel-2 delineation, 10 m, snapshots 2018 / 2020 / 2022). Zenodo record 8369314. Check its licence and citation before publishing.
- **Test lakes:** 50 lakes chosen from that inventory that the model never saw (`configs/independent_test_lakes.csv`, 50 `primary` rows plus 20 `backup` rows).
- **No retraining.** Train once on GLOID (already done), then only run inference and score against the inventory polygons.

Already validated (for context, do not redo): GLOID vs the inventory polygons on 16 overlapping lakes gives a median polygon IoU of 0.95 (`reports/gloid_vs_thirdpole.csv`); the model on South Lhonak vs its 2022 manual outline gives IoU 0.83 to 0.97.

## 1. What you need locally (none of this is in git)

| Item | Where it goes | How to get it |
|---|---|---|
| Third Pole inventory | `data/external/third_pole_inventory/` (keep the inner folder `Third Pole/` with the `.shp` files) | Zenodo 8369314 (about 17 MB). Aparna also has `Glacial lake inventory.zip`. |
| GLOID S2 polygons | `data/external/gloid/S2_20172024_NTB_GLOID_v1.02.gpkg` | Zenodo 10.5281/zenodo.19370146 (CC BY 4.0). Only needed for the GLOID-vs-inventory comparison (step 7d). |
| Trained weights | `models/seg_fused_test.pt` (final fused model, trained on the 22 pool lakes only) | `*.pt` is git-ignored. Ask Aparna for `results.zip`, or the Kaggle output. Share through cloud storage, not git. |
| Kaggle training tiles | `kaggle_dataset/` (needs `tiles/*.npz` and `splits.json`) | Only for the random forest baseline (step 7c). Ask Aparna for `glacierguards_kaggle.zip` (0.69 GB). Unzip it into `kaggle_dataset/`. |
| Earth Engine credentials | `.env` and `ServiceAccountKey.json` in the repo root | Ask Aparna. **Never commit them.** Copy `.env.example`. |
| Python env | `.venv` | `pip install -r requirements.txt`. Note `scikit-learn` is in the requirements but was not installed in the original env: run `pip install scikit-learn`. CPU torch is enough for this work. |

Windows notes: set `PYTHONUTF8=1`, call the venv Python explicitly (`.venv\Scripts\python.exe`), and do not run several heavy jobs at once.

## 2. Step 1: check the lake list (5 minutes)

```
python scripts/select_independent_lakes.py        # rewrites configs/independent_test_lakes.csv
```

The script is deterministic (seed 42). If the CSV changes, stop and ask why.

What the CSV contains: 50 `primary` lakes plus 20 `backup` lakes.
- **Regions:** 40 `core` (Central and Eastern Himalaya) and 10 `geo_shift` (Western Himalaya, a region no training lake covers). Be aware that the core group is mostly Eastern Himalaya (Bhutan, Sikkim, south-east Tibet); only 9 of the 50 are Central Himalaya, because lakes near training lakes were excluded.
- **Size classes (2022 area):** XS 0.02 to 0.05, S 0.05 to 0.2, M 0.2 to 0.5, L 0.5 to 1, XL over 1 km². Quotas are core 6 / 10 / 8 / 8 / 8 and geo_shift 2 / 3 / 2 / 2 / 1.
- **Dam types:** extra 17, pro 16, peri 13, supra 3, icedam 1.
- **Selection rules:** at least 3 km from every training lake and from South Lhonak; at least 3 km between selected lakes; area stable within 25 % across 2018 / 2020 / 2022; area at least 0.02 km².
- **`gloid_id_2022`:** set for the 15 primary lakes that also have a GLOID 2022 polygon. The rest, including all Western Himalaya lakes, are outside GLOID coverage.
- **`utm_epsg`:** use it for the analysis grid.
- **Backups:** replace a primary lake only when its download or data fails (rule in step 3). Never replace a lake because its score is bad.

## 3. Step 2: download imagery (about 2 to 4 hours of waiting, under 1 GB)

Reuse `src/ingestion/gloid_fetch.py`. It already does retries with exponential backoff, resumable downloads, a failure manifest (`data/raw/_gloid_manifest.csv`), the DEM fix, and the S2 / S1 composites. Do not edit its behaviour for the training lakes. Write a new wrapper `src/ingestion/tp_fetch.py` that imports from it (`RAW`, `download`, `fetch_lake_year`, `log_manifest`, `MAX_BYTES`, `BUF_M`, `RES`).

What differs from the GLOID fetch:
1. **Lake list:** read `configs/independent_test_lakes.csv` (`lake_id`, `utm_epsg`, `years`) instead of `training_lakes.csv`.
2. **AOI grid:** `aoi_grid()` takes GLOID polygons. Write your own that returns the same tuple `(crs, x0, y_top, W, H)`: take the target lake's inventory polygon (`Third_Pole_Glacial Lake_2022.shp`, field `ID_L` equals `tp_id`), reproject to `EPSG:<utm_epsg>`, buffer by 1500 m (`BUF_M`), and snap to the 10 m grid exactly as the existing function does. Include all other inventory polygons that fall inside the window in the reference mask later (step 5), not here.
3. **Folder names:** `data/raw/{lake_id}/` with the lake id `TP_<ID_L>`, so there is no collision with `GLO_*`.
4. **Year:** 2022 only (column `years`). The existing windows apply: S2 composites 1 May to 30 Nov and 1 Aug to 31 Oct, S1 median 1 Jul to 30 Aug.
5. **DEM:** copy the DEM block from `fetch_lake()` (Copernicus GLO-30 with `setDefaultProjection`, bilinear). Do not remove that fix: without it the DEM comes out constant.

Run it with `--workers 4`, and rerun until the manifest shows no `failed` rows. Rerunning is safe because existing files are skipped. Earth Engine sometimes returns a temporary 503; the wrapper retries, and a rerun covers any that still fail.

Replacement rule (decide before looking at any results): replace a primary lake with the next backup in the same `stratum` only if the manifest shows `no_scenes` / `failed` / `skipped_aoi_too_large` after three reruns, or if data quality fails (step 4). Log every replacement in `reports/independent_replacements.csv` with the reason.

## 4. Step 3: build the stacks (about 30 minutes)

`src/preprocessing/gloid_stack_tile.py` builds the 10-channel stack `[VV, VH, NDWI_blue, NDWI_green, NDVI, NDSI, NDGI, elevation, slope, aspect]` at 10 m with nodata -9999. It reads the lake list from `training_lakes.csv` and also writes 256x256 tiles. For evaluation we only need `stack.tif`. Make two small, backwards-compatible changes:
- add `--lakes-csv PATH` and `--id-col NAME` arguments (defaults keep the current behaviour);
- add `--skip-tiles` so it stops after writing `data/interim/{lake_id}/2022/stack.tif` and `stack_meta.json`.

Quality gate, same rule as training: a lake-year is usable only if `max(valid VV fraction, valid NDWI_green fraction) >= 0.5` (see `stack_meta.json` and `scripts/dataset_report.py` for how it was computed). Lakes that fail go to the replacement rule above.

## 5. Step 4: reference masks (about 1 hour)

Write `scripts/tp_to_masks.py`, modelled on `scripts/gloid_to_masks.py` (same rasterisation approach):
- Read the stack grid from `stack_meta.json` (`grid`: crs, x0, y_top, W, H, res_m), exactly as `gloid_to_masks.py` does.
- Rasterise **all** inventory 2022 polygons that intersect the window onto that grid (neighbouring lakes must not count as background). Reproject from EPSG:4326 to the stack CRS first.
- Also rasterise the **target** lake's polygon separately.
- Save `data/eval/{lake_id}/ref_all.npy` and `ref_target.npy` (uint8, same shape as the stack), and a small JSON with the target area in km².

Sanity check: for each lake, the target polygon area from the raster should be within 5 % of `area_2022_km2` in the CSV. Investigate any lake outside that.

## 6. Step 5: run the model and score (about 1 hour)

Write `scripts/eval_independent.py`. Reuse `load_model` and `predict_scene` from `src/training/predict_masks.py` (the optical-coverage rule is built in: below 80 % valid optical coverage the model runs SAR-only, matching training). Use the frozen checkpoint `models/seg_fused_test.pt` and threshold 0.5. **Fix the model, threshold and metrics before scoring; do not tune anything on these lakes.**

For each lake compute two scores:
- **Primary: target-lake score.** Restrict both prediction and reference to the target polygon buffered by **400 m** (the same definition used in the South Lhonak check): IoU, F1, precision, recall, and relative area error `(A_pred - A_ref) / A_ref`.
- **Secondary: whole-window score.** The same metrics over the full window with `ref_all` (all lakes).

Write `reports/independent_eval/per_lake.csv` with one row per lake and model: `lake_id, tp_id, region_group, size_class, dam_type, model, iou_target, f1_target, precision_target, recall_target, area_ref_km2, area_pred_km2, area_err_pct, iou_window, optical_valid_fraction, used_optical`.

## 7. Step 6: baselines on exactly the same lakes (about 3 hours)

Same stacks, same reference, same buffer, same metrics. Add each as a `model` value in `per_lake.csv`.

a. **NDWI threshold.** `NDWI_green > 0.3` (McFeeters; used by Ghimire et al.) and a per-lake Otsu threshold on valid pixels (`skimage.filters.threshold_otsu`). Optical only. Report both.

b. **SAR threshold.** VV dB threshold by Otsu per lake, and a fixed value (for example -15 dB). SAR only.

c. **Random forest** (the approach of Pathak et al., Scientific Reports 2025, with an almost identical feature set to our 10 channels).
   - Train **only on the 22 pool lakes** in `kaggle_dataset/splits.json` (`train_pool`), using the pixels and masks in `kaggle_dataset/tiles/*.npz`. Never use independent lakes or the 6 test lakes.
   - Sample about 200,000 pixels per class, balanced, and include hard negatives (pixels within about 300 m of a lake edge).
   - `RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0)`; report probability thresholds 0.5 and 0.75 (Pathak used 0.75).
   - Use the same normalisation as `src/training/seg_dataset.py: normalize` if you feed the model the same arrays.

d. **GLOID itself**, on the 15 lakes with a `gloid_id_2022`: score the GLOID 2022 S2 polygon against the inventory polygon with the same metrics. This is the ceiling of the training labels.

e. **Ablations (optional but recommended):** fused vs SAR-only vs optical-only. Only `seg_fused_test.pt` is a final model. For clean ablations, train `--channels sar --split test` and `--channels optical --split test` on Kaggle (about 30 minutes each; see `kaggle/KAGGLE_TRAINING.md`), then add them as models. Do not substitute CV fold models without saying so.

## 8. Step 7: statistics and report (about 1 hour)

From `per_lake.csv`:
- Mean IoU with a 95 % bootstrap confidence interval over lakes (10,000 resamples), overall and by `size_class`, `region_group` and `dam_type`. With 50 lakes the overall interval is about ±0.03; with 5 to 10 lakes per class it is about ±0.06, so describe per-class results as trends, not proofs.
- Paired comparison of the U-Net against each baseline on per-lake IoU (Wilcoxon signed-rank).
- Figures: IoU against lake area (log axis); a table of the 5 worst lakes with overlays saved to `reports/qa/independent/`; boxplots by size class.
- A short `reports/independent_eval/SUMMARY.md` with the table, the exact commands and the git commit hash used.

## 9. Acceptance checks (run before you report anything)

1. No independent lake is within 3 km of any training lake (guaranteed by the selection script; assert it anyway).
2. No inventory polygon used as reference overlaps a GLOID polygon that was a training label.
3. The checkpoint used is `seg_fused_test.pt` and its metrics file says `split: test`, trained on the pool only.
4. Every lake in the primary list appears in `per_lake.csv`, or in `independent_replacements.csv` with a data-failure reason. Nothing is dropped for a bad score.
5. All thresholds and model choices were fixed before scoring.
6. Reference areas from the rasterised polygons match the CSV areas within 5 %.

## 10. Caveats to state in the paper

- The inventory and GLOID are both derived from Sentinel-2, so their errors are partly correlated. The inventory is manual and independent in method, but not independent in sensor.
- The inventory polygons are a 2022 snapshot; our composite is built from Aug to Oct (or May to Nov) 2022 scenes. Small timing mismatches add noise, most for lakes under 0.05 km² (a boundary error of one pixel is a large fraction of their area).
- The core group is mostly Eastern Himalaya; only 9 of the 50 lakes are in the Central Himalaya, where the training lakes are.
- 5 to 10 lakes per size class supports trends, not strong per-class claims.
- Confirm the inventory licence and cite it correctly.

## 11. What to commit

Commit: `configs/independent_test_lakes.csv`, the new scripts, `reports/independent_eval/*.csv` and `SUMMARY.md`, and small figures.
Do not commit: `data/`, `.env`, `ServiceAccountKey.json`, `*.pt`, zips.

## 12. Time and disk

| Step | Time | Disk |
|---|---|---|
| 1 Selection | 5 min | none |
| 2 Download | 2 to 4 h wall clock, mostly waiting | about 0.5 to 1 GB in `data/raw` |
| 3 Stacks | 30 min | about 0.5 GB in `data/interim` |
| 4 Reference masks | 1 h | small |
| 5 Model scoring | 1 h on CPU | small |
| 6 Baselines | 3 h (random forest is the slowest) | small |
| 7 Report | 1 h | small |
| **Total** | **about 1 to 1.5 working days** | **under 2 GB** |

Check free disk before step 2. At the time of writing about 17 GB was free on the original machine.
