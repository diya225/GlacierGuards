# Project status

Last updated: 2026-11-09 (Diya Manu). Branch `independent-test-results`, commit `ebe5179`.

This page summarises where the GLOF early-warning project stands. It covers what is finished, where the results are, and what is left, in order. Detailed instructions remain in `docs/MILESTONES_7_TO_END.md` (Milestones 7 to 10) and `docs/INDEPENDENT_TEST_AND_BASELINES.md` (50-lake test).

## At a glance

| Part | Status |
|---|---|
| M1 to M3: repo, data ingestion, preprocessing | Done |
| M4: labels (GLOID polygons as weak labels) | Done |
| M5: segmentation model (U-Net, SAR + optical) | Done |
| M6: lake area series and burst detector (6 lakes) | Done (exploratory) |
| 50-lake independent test, baselines and ablations | **Done (2026-11-09)** |
| M7: risk score (area series for 2 more lakes, crest points, risk code) | Not started |
| M8: full evaluation (hindcasts, false alarms, remaining leave-one-lake-out) | Partly done |
| M9: serving (API, Docker) | Not started |
| M10: documentation and deliverables | Not started |

`src/risk/` and `src/serving/` are still empty packages, and there is no `docker/` folder yet.

## 1. Work completed

### M1 to M4: foundations
- The repository, environment and Earth Engine access work: `src/ingestion/`, `src/preprocessing/`, `src/utils/auth.py`.
- **Labels:** GLOID annual lake polygons are used as weak labels (model-derived, CC BY 4.0). That gives 223 labelled lake-years across 30 lakes and 1,130 tiles; see `reports/dataset_report.md` and `DATA_PROVENANCE.md`.

### M5: segmentation model
- **Model:** U-Net with an EfficientNet-B3 encoder and 10 input channels (VV, VH, five optical indices, elevation, slope, aspect). It was trained on Kaggle.
- **Final model:** `models/seg_fused_test.pt`, trained on the 22 pool lakes. It scores IoU 0.919 on the 6 held-out test lakes.
- **5-fold lake-grouped CV:** fused 0.911, optical-only 0.916, SAR-only 0.819.
- **Leave-one-lake-out (6 of 22 lakes done):** mean IoU 0.906, minimum 0.856.
- Weights are not in git. Ask Aparna or Diya for the `.pt` files.

### M6: lake area series and burst detection
- **Lakes:** area series and a drop detector exist for 6 lakes: four outbursts (South Lhonak, Lower Ngole, Jinwuco, Chamai) and two quiet lakes (Imja Tsho, Thulagi).
- **Detection:** the plain detector catches 2 of 4 outbursts (South Lhonak and Lower Ngole, both 3 days after the event); the exploratory per-track mode catches 3 of 4.
- **Forecaster:** the deep forecaster was no better than "same as last scene", so the risk score will not use it.
- Details and caveats: `docs/M6_MULTI_LAKE_RESULTS.md`.

### 50-lake independent test (paper track)
This test scores the model against hand-drawn 2022 outlines from the Third Pole inventory, on 50 lakes it never saw: 40 in the Central and Eastern Himalaya and 10 in the Western Himalaya.
- **Coverage:** all 50 lakes were scored, none replaced, and all six acceptance checks pass. Rerunning one lake from scratch reproduced its results exactly.

| Model | Mean IoU [95 % CI] | Significance vs fused U-Net |
|---|---|---|
| U-Net fused | **0.843** [0.79-0.89] | n/a |
| Random forest (Pathak et al. approach) | 0.814 | not significant (p = 0.12) |
| U-Net optical-only | 0.786 | not significant (p = 0.37) |
| U-Net SAR-only | 0.752 | fused better (p < 0.001) |
| Fixed SAR / NDWI thresholds | 0.67 / 0.66 | fused better (p < 0.001) |
| GLOID polygons (15 lakes, label ceiling) | 0.867 | not significant (p = 0.28) |

- **Lake size:** accuracy falls with lake size, from IoU 0.95 for lakes over 1 km² to 0.68 for lakes under 0.05 km².
- **New region:** the Western Himalaya lakes score as well as the rest (0.86 vs 0.84).
- **Fusion:** fusion clearly beats SAR alone. It ties with optical alone on a typical lake, but avoids a few large optical failures. All test lakes had clear optical images, so this test cannot show the cloud benefit.
- **Results:**
  - `reports/independent_eval/SUMMARY.md`: tables and run notes.
  - `reports/independent_eval/per_lake.csv`: raw scores.
  - `reports/qa/independent/`: overlays.
- **Ablation weights:** `models/seg_sar_test.pt` and `models/seg_optical_test.pt`, trained with `kaggle/kaggle_ablations.ipynb`. They are not in git.

## 2. Work remaining

Listed in the order it can be done. Effort estimates come from `docs/MILESTONES_7_TO_END.md`.

### Now: finish and share the 50-lake test
1. **Share the results.** Push branch `independent-test-results` and merge it, then send the two ablation `.pt` files privately.
2. **Citation.** Look up the DOI of the associated Nature Communications article (Zhang et al. 2023) so the inventory can be cited fully.
3. **Optional follow-up.** Score the 2018 and 2020 inventory snapshots as a separately labelled extra check. The main test was fixed to 2022 before scoring.

### M7: risk score (main remaining work)
| Task | Who | Effort | Blocks |
|---|---|---|---|
| 7a Area series for **Tsho Rolpa** (use `seg_fused_cv_fold4.pt`) and **Lower Barun** (use `seg_fused_cv_fold3.pt`). These lakes were in the training pool, so use the fold model that held them out. | Diya | 1 to 2 h each, mostly waiting | Risk scores for these lakes |
| 7b **QGIS crest points**, dam height and width, and outlet flag for the 5 lakes, then `crest_points_to_yaml.py` into `lakes.yaml` | **Manual** (Aparna or Diya) | 1 to 2 h for the first lake, then 45 to 60 min each | Freeboard term of the risk score |
| **Confirm the weights** (0.40 / 0.35 / 0.25), then write and commit `docs/PREREGISTRATION.md` with the hindcast success criteria **before** any hindcast | Aparna decides | about 1 h | M8 hindcasts |
| 7c **Risk code:** `freeboard_index.py`, `trend_stats.py`, `area_volume_scaling.py`, `risk_score.py` (EWMA and CUSUM), plus unit tests | Diya | 2 to 3 days | M8, M9 |
| Weight-grid robustness check | Diya | part of 7c | n/a |

7c can start now. Until the crest points exist, the freeboard term stays empty and its weight is renormalised and flagged.

### M8: evaluation (after M7)
- **Leave-one-lake-out** for the remaining 16 pool lakes, on Kaggle (about 30 min each).
- **Cloud stress test:** hide the optical layers at random, using the new ablation models. This is the test that can show whether fusion helps.
- **Three repeat seeds** for confidence intervals.
- **Hindcasts against the preregistered criteria:**
  - South Lhonak 2023 (main);
  - Gongbatongsha 2016, after confirming its lake match (may be a miss);
  - Imja Tsho 2016 lowering.
- **False alarms** per lake-year on the quiet lakes.
- **One results table** (`src/training/evaluate.py`) that includes the 50-lake results.

### M9: serving (after M7)
- **Inference service:** `src/serving/inference_service.py`, writing to SQLite.
- **API:** FastAPI with `/health`, `/lakes/{id}/latest` and `/lakes/{id}/history`.
- **Containers:** three Dockerfiles plus `docker-compose.yml`, with no credentials or weights baked in. Test locally with Docker Desktop on WSL2.

### M10: documentation (last)
- **README:** setup, adding a lake, retraining, reading a risk score.
- **`provenance.csv`:** add the Third Pole inventory (CC BY 4.0, doi:10.5281/zenodo.8369314), GLOID and Copernicus statements.
- **Evaluation report** from the M8 tables.
- **Demo notebook.**
- **Final review by Aparna.**

### Dependency order
```
Now:            share 50-lake test   ·   7a area series (Tsho Rolpa, Lower Barun)   ·   start 7c risk code
In parallel:    7b QGIS crest points (manual)   ·   weights + PREREGISTRATION.md (Aparna)
Then:           M7 done -> M8 evaluation and M9 Docker/API -> M10 documents
```

## 3. Practical notes for whoever continues
- **Files not in git, shared privately:**
  - `.env` and `ServiceAccountKey.json`;
  - all `.pt` weights and `rf_baseline.joblib`;
  - `data/`, `kaggle_dataset/` (from `glacierguards_kaggle.zip`), and the Third Pole and GLOID files in `data/external/`.

  See section 1 of `docs/INDEPENDENT_TEST_AND_BASELINES.md` for where each goes.
- **`.pt` files on Windows:** a `.pt` file is a zip archive internally. Windows may extract it into a folder or add `.zip` to its name. Each `.pt` must be a single file of about 50 MB.
- **Running on Windows:** use `PYTHONUTF8=1` and `.venv\Scripts\python.exe`, Python 3.12, and run one heavy job at a time.
- **Environment:** `tabulate` is needed by `scripts/independent_stats.py` but is not in `requirements.txt`. Install it, or add it there.
- **Kaggle notebooks:** for single runs, use a notebook containing only the commands you need. "Run All" on `kaggle/kaggle_train.ipynb` runs about 23 trainings.
- **Reporting rules:** fix thresholds and criteria before looking at results, and report negative results.
