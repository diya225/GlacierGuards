# GlacierGuards handoff (written 2026-11-05, ~01:45)

Written for: the next working session (you or Claude) picking the project up at the South Lhonak anomaly review.

## 0. Update (later on 2026-11-05): read this before sections 1 to 3

Sections 1 to 3 below were written before these findings and are partly superseded.

- **The area series was measuring the wrong thing.** `area_m2` summed all predicted water in the 3 km window (other lakes and channels included). `scripts/make_lake_roi.py` now builds a fixed region around the target lake (independent manual 2022 outline plus 400 m, chosen before looking at the burst) and adds `lake_area_m2` to `area_series.csv`; `src/training/train_temporal.py` crops and masks frames to that region automatically (`data/predictions/{lake}/roi.npy`). Use `lake_area_m2`, not `area_m2`.
- **South Lhonak result with the fix.** Pre-burst lake area is 1.34 to 1.55 km2 for 2017 to 2022 (manual 2022 outline: 1.67 km2). Area is 1.63 km2 on 2023-09-26 and 0.93 km2 on 2023-10-06, a drop of 0.70 km2. Scene-to-scene changes before the burst have a standard deviation of 0.08 km2 (61 pairs, gaps of 15 days or less) and the largest earlier drop is 0.17 km2. This is a detection two days after the event (first Sentinel-1 scene after the burst), not a prediction. August to September 2023 was the highest in the record (1.62 and 1.70 km2 against 1.35 to 1.55 before); suggestive of a precursor but not statistically established. The post-burst recovery to about 1.4 km2 within 7 weeks is unexplained; SAR-only dates differ from neighbouring optical dates by up to 0.3 km2.
- **Milestone 6 result (finished 2026-11-05 12:32; this replaces the stale files mentioned in section 1).** ConvLSTM forecaster (hidden 8, 15 epochs, trained only on targets before 2023-10-04, each season scored by a model that never saw that season; `python src/training/train_temporal.py --lakes south_lhonak --cv-year --train-before 2023-10-04 --epochs 15 --hidden 8`, about 6 minutes on CPU) on the lake-specific series: 96 out-of-sample targets. **It does not beat persistence ("same as last scene")**: pre-burst MAE 0.084 vs 0.083 km2 on all pairs, 0.058 vs 0.054 km2 on pairs with a gap of 15 days or less. It flags 2023-10-06 (residual z = -9.7), but its prediction there (1.64 km2) is essentially the previous scene (1.63 km2), so it adds nothing over the simple detector. Its three pre-burst |z| > 4 values all follow long seasonal gaps (2020-06-11, 2020-08-07, 2021-06-03; early-season SAR areas are low, for example 0.69 km2 on 2020-06-11, probably ice or wet snow) and disappear when comparisons are limited to gaps of 15 days or less. **Use the simple detector as the Milestone 6 method**: `python src/training/burst_detection.py --lake south_lhonak --burst-date 2023-10-04` flags 2023-10-06 (drop 0.70 km2, 8.6 robust sigmas, 2 days after the burst) with 0 false alarms in 61 pre-burst comparisons; outputs `reports/south_lhonak_burst_detection.{csv,png}`. Report the forecaster as a negative result for a single lake; it might help only with multi-lake training data. The attention model was not run. `train_temporal.py` now supports `--tag`, uses CUDA if available, and computes z-scores from pre-burst residuals only. A Kaggle notebook exists (`kaggle/kaggle_temporal.ipynb`, data `glacierguards_temporal.zip`, built by `scripts/package_temporal_for_kaggle.py`) but is not needed for this lake.
- **Milestone 6 status and caveats (read before citing any M6 number).** Status: complete for the South Lhonak hindcast; **not** complete against the build plan's wording, which asks for one shared temporal model over all cohort lakes and an anomaly series per lake. Caveats:
  1. **One lake, one event.** The false-alarm rate (0 of 61 pre-burst comparisons) comes from one lake's quiet record. No other lake has been run. Planned extension: the four stable lakes Imja Tsho (inventory ID 4514), Tsho Rolpa (4575), Lower Barun (4474) and Thulagi (3483) as false-alarm controls and for a multi-lake forecaster (about 1 GB and 1 to 2 hours per lake). Predict Tsho Rolpa and Lower Barun with the cross-validation checkpoint that held each lake out, because the final model trained on them.
  2. **Detection, not prediction.** The burst is flagged 2 days after the configured burst date (2023-10-04), at the first Sentinel-1 pass after it; the lag cannot be shorter than the revisit time. The plan's check "elevated anomaly in the months before October 2023" is **not demonstrated**: the forecaster's 2023 pre-burst residuals range from z = -3.4 to +3.2 with a mean of -0.28, and the only value above 3 (2023-06-16) is a rise after the winter gap, probably the early-season ice or wet-snow effect.
  3. **Rules were chosen with the data in view.** The 4-sigma threshold and the 15-day gap rule (`burst_detection.py` defaults) were set after the burst and the winter-gap artefacts had been seen. They are conventional, not pre-specified; say so in any write-up, and do not tune further on this event.
  4. **The deep forecaster adds nothing here.** Out-of-sample error 0.084 vs 0.083 km2 for "same as last scene" (0.058 vs 0.054 km2 on pairs 15 days apart or less). Only the ConvLSTM was run, at a small size (hidden 8, 15 epochs); the attention model was not. Report it as a negative result for one lake.
  5. **Unexplained post-burst recovery.** Area drops to 0.93 km2 on 2023-10-06 and is back to about 1.4 km2 by late November. Not yet checked against imagery or published post-burst areas.
  6. **SAR-only noise.** Scene-to-scene change has a standard deviation of 0.083 km2 on SAR-only scenes vs 0.052 km2 on optical scenes, and SAR-only dates can sit up to 0.3 km2 below neighbouring optical dates. The Sentinel-1 scenes mix two tracks (ascending 12 with 90 scenes, descending 48 with 85); a hint of about 0.15 km2 offset between them (4 dates per track) is untested, and only 14 of the 98 stack dates matched a scene's orbit metadata, so the date-to-orbit lookup needs fixing first. Early-season SAR areas can be too low (0.69 km2 on 2020-06-11).
  7. **Possible precursor is exploratory only.** 15 Aug to 30 Sep 2023 averaged 1.68 km2 vs 1.46 +/- 0.07 km2 in five earlier years (z = 3.3, n = 5 years).
  8. **Optional noise reductions, not needed to continue:** per-track series, averaging the 5 cross-validation checkpoints plus the final model (use expected area), flagging ice and wet-snow dates, a seasonal-climatology detector, and a detection curve. Choose any of them by pre-burst noise and IoU against the manual 2022 outline, never by how well they catch the burst.
- **Independent validation exists.** `reports/gloid_vs_thirdpole.csv`: GLOID polygons vs the manual Third Pole inventory, median polygon IoU 0.95 on 16 lakes x 3 years. The model vs the manual 2022 South Lhonak outline: IoU 0.83 to 0.97 (optical dates 0.92 to 0.965, SAR-only 0.83 to 0.90).
- **Next work (no hand-digitising):** `docs/INDEPENDENT_TEST_AND_BASELINES.md` with the lake list `configs/independent_test_lakes.csv` (50 primary + 20 backup lakes from the Third Pole inventory, built by `scripts/select_independent_lakes.py`).
- **Not done:** Milestone 7 (needs crest points and weights; full instructions for Milestones 7 to 10 including the QGIS crest-point workflow are in `docs/MILESTONES_7_TO_END.md`, with the helper `scripts/crest_points_to_yaml.py`); post-burst South Lhonak check against published area values (read the sources and cite the numbers; none verified yet).
- **Housekeeping:** `kaggle_dataset/` is no longer in the project folder (not deleted by me); `glacierguards_kaggle.zip` still has its contents. A `data/external/third_pole_inventory/` folder was added.

## 1. Where things stand

| Stage | Status |
|---|---|
| Training data (28 lakes, GLOID weak labels, 877 tiles) | Done. `glacierguards_kaggle.zip` (0.66 GB), built by `scripts/build_kaggle_dataset.py`. |
| Segmentation training on Kaggle | Done. Results unpacked in `data/results_kaggle/`; final model `models/seg_fused_test.pt`. |
| South Lhonak history download | Done. 175 S1 + 101 S2 scenes, Jun-Oct 2017-2023, in `data/raw/south_lhonak/` (about 1.0 GB). |
| South Lhonak preprocessing | Done. 98 stacked dates in `data/interim/south_lhonak/stacked/`. |
| South Lhonak mask prediction | Done. 98 dates, `data/predictions/south_lhonak/` (model: `seg_fused_test.pt`). |
| **Temporal anomaly check on the long history** | **NOT DONE.** The run was stopped by the 01:35 watchdog before it finished. See section 3. |
| Milestone 7 (freeboard, risk score) | Not started. Blocked on moraine crest points and weight confirmation. |

## 2. Segmentation results (Kaggle, all on held-out lakes)

- Final model on the 6 held-out test lakes: IoU 0.919, F1 0.958.
- Fused 5-fold CV IoU: 0.888, 0.907, 0.905, 0.927, 0.926. Optical-only is about equal (about 0.92 mean). SAR-only is weaker and unstable (0.72 to 0.91, mean about 0.82).
- Weakest test lakes: `GLO_82.81138_29.30316` (IoU 0.76, recall 0.99 but precision 0.77, so it over-predicts) and `GLO_87.08223_28.12953` (0.80). Thulagi (`GLO_84.485_28.48833`) is 0.94.
- Caveat: labels are GLOID model outputs (reported F1 about 0.82 for S1, 0.92 for S2), so these scores measure agreement with GLOID, not with hand-digitised truth.
- Full metrics: `models/*_metrics.json` (per-lake numbers under `per_lake`).

## 3. The South Lhonak anomaly review: what is and is not known

**The long-history run did not finish.** I stopped it at 01:35 under the no-progress rule. The files currently in `data/predictions/south_lhonak/anomaly_series.csv` and `reports/south_lhonak_anomaly.png` are **stale**: they come from an older run on only 26 dates (16 Jun to 25 Nov 2023, 6 of them before the burst). Do not use them as the result. `models/temporal_checkpoint.pt` was saved by the stopped run after its first fit only, so treat it as unvalidated too.

Why it stalled: the ConvLSTM at 64x64 is slow on CPU (about 6 min per fit at hidden 16 / 40 epochs; about 1.5 min at hidden 8 / 15 epochs). The first versions did a refit per target (about 90 fits). I switched to blocked-by-year CV (`--cv-year`, 8 fits at the small setting), which I estimated at about 10 min but did not see complete.

What the data already show (valid regardless of the temporal model):
- Predicted lake area in 2017-2022 sits around 2.2 to 2.4 km2, with scene-to-scene swings of about 100,000 to 250,000 m2 (5 to 10%). Most dates are SAR-only because optical coverage is often low, and SAR-only was the weakest segmentation mode.
- Pre-burst 2023 values reach 2.78 km2 on 2023-09-16.
- After the 4 Oct 2023 burst, predicted area drops only to about 1.78 to 2.0 km2. If the burst drained most of the lake, that is far too small a drop. **The masks may include wet debris or flooded ground, or the model may be over-predicting. Check against imagery before trusting any area-based signal.**

What the stale 26-date run suggested (weak evidence only):
- 2023-10-06 residual z = -3.5 (observed area 0.68 km2 below the forecast). It is the only value past 3, and it falls just after the burst, with the expected sign.
- 2023-09-16 residual z = +2.2 (area up) was the largest pre-burst value. It could be a precursor or noise.
- With 6 pre-burst targets there is no usable false-alarm rate. The forecaster beat simple persistence in neither period (mean absolute residual 229k vs 173k pre-burst, 165k vs 143k post-burst), so it adds nothing over "same as last scene" yet.

**Honest current conclusion:** there is no validated evidence that the anomaly stage detects a South Lhonak pre-burst signal. That conclusion could change with the full-history run.

## 4. To resume (in this order)

1. Re-run the temporal check, running nothing else at the same time:
   ```
   PYTHONUTF8=1 .venv/Scripts/python.exe src/training/train_temporal.py --lakes south_lhonak --cv-year --train-before 2023-10-04 --epochs 15 --hidden 8
   ```
   Allow about 15 minutes, use `PYTHONUNBUFFERED=1` to see progress, and confirm `anomaly_series.csv` has about 96 rows before using it. Then `python src/training/plot_anomaly.py --lake south_lhonak`.
2. Compare against persistence, and set the alarm threshold from pre-burst residuals only.
3. Visually inspect predicted masks around 2023-09-16, 2023-10-06 and 2023-10-09 (the post-burst area looks too large). If the masks are wrong, the temporal check cannot be trusted.
4. If the signal is too noisy, options are: use only dates with optical coverage; smooth area with a rolling median; or add Landsat/older data.
5. Ask the user for the Milestone 7 inputs below.

Use `.venv\Scripts\python.exe` explicitly. A plain `python` in the shell resolved to a different runtime earlier. On Windows set `PYTHONUTF8=1`, because `run_pipeline.py` prints box characters.

## 5. Waiting on the user

- **M7-A:** moraine crest points (QGIS on the Copernicus DEM) for lakes to be scored. Without them freeboard and risk scoring cannot run. Draft automatic crest estimates from the DEM are possible but would be rough and need review.
- **M7-B:** confirm or change the risk-score weights (plan suggests area growth 0.40, freeboard 0.35, volume 0.25).
- Cleanup approval (section 6).
- Kaggle `dataset-metadata.json` still has the placeholder `YOUR_KAGGLE_USERNAME`; only matters for command-line upload.

## 6. Folder state and pending cleanup

Done this session: planning docs moved to `docs/planning/`, logs to `logs/`, Kaggle docs and notebook to `kaggle/`. README links updated.

Proposed deletions (not done, need approval):
- `results.zip` (1.08 GB): already extracted to `data/results_kaggle/`.
- `kaggle_dataset/` (658 MB): rebuildable from `scripts/build_kaggle_dataset.py`.
- `kaggle/glof_ews_kaggle.zip` (494 MB): old South Lhonak-only package.
- `models/segmentation_checkpoint.pt` and metrics (old South Lhonak model).
- `models/temporal_checkpoint.pt` (unvalidated).

Keep: `data/results_kaggle/` (holds the SAR/optical and leave-one-lake-out checkpoints not in `models/`), `glacierguards_kaggle.zip`, `data/external/`, `DATA_PROVENANCE.md`, `provenance.csv`, `.env`, `ServiceAccountKey.json` (secrets, git-ignored).

Git: nothing from this work is committed. The working tree has many new files; commit when ready. Attribution to cite: GLO/GLOID (Rawlins et al., Zenodo 10.5281/zenodo.19370146, CC BY 4.0); the DOI/year discrepancy is noted in `DATA_PROVENANCE.md`.

## 7. Code changes this session

- `src/training/train_temporal.py`: added `--train-before DATE` (fit only on targets before the date, e.g. the burst) and `--cv-year` (score each season with a model that never saw it).
- `src/ingestion/gee_fetch.py` has no retry on Earth Engine 503 errors; one crashed the 2019 download. I worked around it with a shell retry loop. Adding backoff to the fetcher would be worthwhile.
- South Lhonak stays inference-only in every split and is never used for training.

## 8. Process state

All Python processes of mine were stopped. Nothing is running.
