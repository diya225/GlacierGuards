# GlacierGuards handoff (written 2026-11-05, ~01:45)

Written for: the next working session (you or Claude) picking the project up at the South Lhonak anomaly review.

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
