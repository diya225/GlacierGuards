# Milestone 7 to the end: instructions

This document covers Milestone 7 (freeboard and risk score, including the QGIS crest-point work), Milestone 8 (evaluation), Milestone 9 (serving in Docker) and Milestone 10 (documentation). The original plan is in `docs/planning/GLOF_EWS_Build_Plan.md`; where this document differs from it, this document reflects what we learned and says why.

## 0. Read this first

### Where the project stands
- Lake segmentation (M5) is trained and validated on held-out lakes. Final model `models/seg_fused_test.pt`.
- The temporal stage (M6) is done for six lakes (four outburst events, two quiet lakes): the plain drop detector finds 2 of 4 events and the exploratory per-track mode 3 of 4; the deep forecaster is no better than "same as last scene". Details, numbers and caveats: `docs/M6_MULTI_LAKE_RESULTS.md`.
- `src/risk/` and `src/serving/` are empty packages. Nothing in M7 to M10 has been built.
- The paper track (50-lake independent test and baselines) is in `docs/INDEPENDENT_TEST_AND_BASELINES.md` and runs in parallel to this document.

### Decisions already made (do not reopen without a reason)
1. **The risk score uses the lake-specific area series, not the forecaster's residuals.** The forecaster added nothing. Area is measured inside a fixed lake region (`scripts/make_lake_roi.py`), never over the whole 3 km window.
2. **The cohort is moraine-dammed lakes only.** Ice-dammed lakes fail by a different mechanism; do not build separate scoring branches.
3. **Weights come from literature and are not tuned on the hindcast.** Start values: area growth 0.40, freeboard 0.35, volume 0.25 (already in `configs/pipeline.yaml`). A weight-grid robustness check is reported alongside.
4. **Success criteria are written down before any hindcast run** (section 4.2). Both bursts were sudden-trigger events, so the score measures susceptibility, not trigger timing. Never claim the system predicted a date.
5. **Burst detection is a separate alert from the risk score.** The drop detector (`src/training/burst_detection.py`) flags a drainage that has happened; the risk score tracks susceptibility before one. Serve them as two different fields.

### Rules that apply everywhere
- Never commit `.env`, `ServiceAccountKey.json`, `*.pt`, zips or `data/`. Share weights and credentials outside git.
- Use `PYTHONUTF8=1` and `.venv\Scripts\python.exe` explicitly on Windows. Run one heavy job at a time.
- Report negative results. If a hindcast misses, say so.
- Fix a threshold, model or metric before you look at the result it will be scored on.

### Overview

| Milestone | Goal | Needs | Effort (rough) |
|---|---|---|---|
| 7a Lake series for every scored lake | One lake-specific area series per lake | Nothing | 1 to 2 h per lake, mostly waiting |
| 7b QGIS crest points | Crest points, dam height and width per lake | **Manual Task** | 1 to 2 h first lake, then 45 to 60 min each |
| 7c Risk code | Freeboard, trend, volume, risk score | Weights confirmed | 2 to 3 days |
| 8 Evaluation | Leave-one-lake-out, ablations, hindcast, false alarms | Review hindcast framing | 2 to 3 days |
| 9 Serving | SQLite, FastAPI, three Dockerfiles, compose | Test locally | 2 days |
| 10 Documentation | README, provenance, evaluation report, demo notebook | Final review | 1 day |

## 1. Choose the lakes to score

Score the five lakes that have a confirmed lake outline and a segmentation checkpoint:

| Lake (`lakes.yaml` id) | Role | Inventory ID (Third Pole, field `ID_L`) | Checkpoint to predict with |
|---|---|---|---|
| South Lhonak (`south_lhonak`) | Burst 2023-10-04 (main hindcast) | 4272 | `models/seg_fused_test.pt` (never trained on it) |
| Imja Tsho (`imja_tsho`) | Monitored; lowered 3.4 m in 2016; has an outlet channel | 4514 | `seg_fused_test.pt` (test lake) |
| Tsho Rolpa (`tsho_rolpa`) | Monitored; lowered about 3 m in 2000 | 4575 | `seg_fused_cv_fold4.pt` (this lake was in the training pool, in fold 4) |
| Lower Barun (`lower_barun`) | Monitored | 4474 | `seg_fused_cv_fold3.pt` (training pool, fold 3) |
| Thulagi (`thulagi`) | Monitored | 3483 | `seg_fused_test.pt` (test lake) |

Why the checkpoint column matters: predicting a training-pool lake with the final model is in-sample, which flatters the series. Use the fold model that held the lake out.

Optional, only after the five are done: Lumding Tsho, Hongu 2 and Gongbatongsha Co. Their matches to GLOID lakes were never confirmed, and Gongbatongsha (burst 2016-07-05, Tibet side) has sparser imagery, so it may be a miss. To find their inventory ID, run this once and read the nearest polygon (check it is the right lake on a map):

```python
import geopandas as gpd, yaml
from shapely.geometry import Point
t = gpd.read_file("data/external/third_pole_inventory/Third Pole/Third_Pole_Glacial Lake_2022.shp").to_crs(32645)
lake = [l for l in yaml.safe_load(open("configs/lakes.yaml", encoding="utf-8"))["lakes"] if l["id"] == "lumding_tsho"][0]
p = gpd.GeoSeries([Point(lake["lon"], lake["lat"])], crs=4326).to_crs(32645).iloc[0]
d = t.geometry.distance(p); i = d.idxmin(); print(t.loc[i, ["ID_L", "Area_2022", "Dam_T", "Lat", "Lon"]].to_dict(), round(d[i]), "m away")
```

(EPSG:32645 is UTM zone 45N, which covers 84 to 90 degrees east. For a lake outside that range use the matching zone: EPSG:32644 for 78 to 84 degrees east, EPSG:32646 for 90 to 96 degrees east. This snippet only needs a rough distance.)

## 2. Step 7a: build the lake-specific area series for each lake

South Lhonak is already done. For each other lake (about 1 GB and 1 to 2 hours each; check free disk first, about 17 GB were free at the last check):

1. **Download** June to October of each year 2017 to 2023, Sentinel-1 and Sentinel-2, plus the DEM. Earth Engine sometimes returns a temporary 503 error, which crashes `gee_fetch.py`; wrap each call in a retry loop. It skips files that already exist, so reruns are safe.
   ```
   for y in 2017 2018 2019 2020 2021 2022 2023; do
     for s in s1 s2; do
       for t in 1 2 3 4; do
         python src/ingestion/gee_fetch.py --lake imja_tsho --sensor $s --start $y-06-01 --end $y-11-01 && break
         sleep 20
       done
     done
   done
   python src/ingestion/gee_fetch.py --lake imja_tsho --sensor dem
   ```
2. **Stack** the scenes (writes `data/interim/{lake}/stacked/{date}.tif`):
   `PYTHONUTF8=1 .venv\Scripts\python.exe run_pipeline.py --lake imja_tsho --skip-download`
3. **Predict masks** with the checkpoint from the table:
   `python src/training/predict_masks.py --lake imja_tsho --ckpt models/seg_fused_test.pt`
4. **Build the lake region and the lake-specific area column** from the inventory outline:
   `python scripts/make_lake_roi.py --lake imja_tsho --tp-id 4514 --buffer 400`
5. **Run the burst detector** as a false-alarm check (use the lake's burst date, or `--burst-date 2100-01-01` for none):
   `python src/training/burst_detection.py --lake imja_tsho --burst-date 2100-01-01`
   Expect zero flags on a quiet lake. If it flags something, look at the dates before trusting the series.

Then look at `reports/{lake}_burst_detection.png` for each lake: the series should be roughly flat or slowly growing, with winter-gap dips from early-season ice (compare June 2020 at South Lhonak). Do not continue with a lake whose series looks wrong.

Optional improvement that also helps M7: build one series per Sentinel-1 orbit track (ascending 12 and descending 48 at South Lhonak) to remove viewing-geometry offsets. First fix the date-to-orbit lookup (only 14 of 98 stack dates matched the scene metadata).

## 3. Step 7b: moraine crest points in QGIS (done manually)

### 3.1 What you are producing and why
Freeboard is the vertical distance between the lake surface and the lowest point of the dam crest. We cannot measure it directly, so M7 computes two proxies per date from your crest points and the segmented shoreline:
- **Horizontal gap (primary):** distance from the shoreline to the nearest crest point.
- **Vertical freeboard (secondary):** lowest crest elevation minus the median DEM elevation along the shoreline.

Treat both as relative trends over time, not safety margins: the DEM is 30 m native resolution, coarse next to a crest tens to hundreds of metres wide. For each lake you also record the dam height and width and whether a visible outlet channel exists.

You need, per lake: **5 to 10 crest points (one flagged as the lowest)**, **dam height**, **dam width** and **has_outlet_channel**. Do the five lakes in section 1 first. Expect 1 to 2 hours for the first lake, then 45 to 60 minutes each.

### 3.2 Before you start
- QGIS 3.28 LTR or newer (3.34 or later is better; the built-in Elevation Profile tool needs 3.26+).
- A DEM for the lake. We already download one with the pipeline:
  - lakes fetched with `gee_fetch.py`: `data/raw/{lake}/dem/dem.tif` (South Lhonak and the other `lakes.yaml` lakes);
  - lakes fetched through GLOID: `data/raw/{GLO_ID}/dem.tif`.
  These are Copernicus GLO-30 resampled to 10 m, so they look smooth but the real detail is 30 m. If a lake has none, run `python src/ingestion/gee_fetch.py --lake {id} --sensor dem`.
- The lake outline: the manual 2022 inventory polygon (`data/external/third_pole_inventory/Third Pole/Third_Pole_Glacial Lake_2022.shp`, field `ID_L`) or the latest predicted lake mask. The inventory polygon is the easier one to load.
- Imagery for context. Use the stack's NDWI band, or add a basemap with **Layer, Add Layer, Add XYZ Layer** (for example Esri World Imagery). Use basemaps to look at, not to trace from, and check their terms of use.
- Create the output file once (make the folder `data/qgis/` first): **Layer, Create Layer, New GeoPackage Layer**, save as `data/qgis/crest_points.gpkg`, and create two layers in it with these exact names and fields (the helper script reads them):

| Layer | Geometry | Fields |
|---|---|---|
| `crest_points` | Point | `lake_id` (text), `elev_m` (decimal), `is_lowest` (integer: 1 or 0), `note` (text, optional) |
| `dam` | No geometry | `lake_id` (text), `height_m` (decimal), `width_m` (decimal), `has_outlet_channel` (integer: 1 or 0), `toe_elev_m` (decimal), `dem_source` (text), `confidence` (integer 1 to 3) |

Use one row per lake in `dam`, and the lake id exactly as in `configs/lakes.yaml` (for example `south_lhonak`).

### 3.3 Step by step, per lake
1. **New project and CRS.** Set the project CRS (bottom right) to the lake's UTM zone, for example EPSG:32645 for South Lhonak. Work in metres.
2. **Load the DEM** (Layer, Add Raster Layer). Check its CRS matches the project; if not, right-click the layer, Export, Save As, in the UTM CRS.
3. **Hillshade.** Processing Toolbox, "Hillshade" (GDAL). Keep the defaults (Z factor 1). Put it under a semi-transparent DEM for a readable relief.
4. **Contours.** Processing Toolbox, "Contour" (GDAL), interval 10 m (5 m looks busy at 30 m resolution). Style thin and grey.
5. **Load the lake outline and imagery** from section 3.2. Zoom to the lake (about 1:10,000).
6. **Find the dam.** The terminal moraine is the ridge across the lake's outlet side, where the outlet stream leaves. On the hillshade it is the arcuate ridge that holds the water back; contours run roughly parallel to it and then bend away downslope. Work out which side is downstream from the contours and the stream that leaves the lake; do not assume a compass direction. If you cannot see the dam clearly, flag `confidence = 3` for the lake and say why in `note`.
7. **Trace the crest line.** Create a temporary scratch line layer (Layer, Create Layer, New Temporary Scratch Layer, Line), toggle editing, and draw the crest along the highest ground of the moraine, from one lateral ridge to the other. Turn on snapping (Project, Snapping Options) only if it helps; do not snap to the DEM.
8. **Sample the crest elevation densely.** Processing Toolbox, "Points along geometry", distance 10 m, input the crest line. Then "Sample raster values" (Raster analysis) on the DEM. Open the result's attribute table and sort by the elevation column. The smallest value is the lowest point of the crest. Zoom to it and check it is really on the crest and not on a gully wall or in the lake.
9. **Place the 5 to 10 reference points.** Create points in the `crest_points` layer at about equal spacing along the crest (use "Points along geometry" at a larger spacing, then copy the points into `crest_points`). **Always include the lowest point**, on the exact location from step 8. Fill in:
   - `lake_id`;
   - `elev_m`: re-sample the DEM at each point with "Sample raster values" and copy the value, or type it from the identify tool;
   - `is_lowest`: 1 for the lowest point, 0 for the rest. Exactly one point per lake gets 1.
10. **Cross-check the elevations.** Load a second DEM. In Earth Engine the easy one is SRTM (`USGS/SRTMGL1_003`); download the lake window or use any other DEM you have. Sample it at the lowest point. If the two DEMs differ by more than 10 m, set `confidence = 3` and write the difference in `note`. Also check the points against a high-resolution image: the lowest point should sit on the moraine crest.
11. **Dam height and width.** Draw one profile line across the dam through the lowest point, perpendicular to the crest, from the lake shore to well downstream of the dam. Use the Elevation Profile panel (View, Panels, Elevation Profile) or the Profile Tool plugin to read the profile.
    - `crest elevation` = the elevation of the lowest point.
    - `toe elevation` (`toe_elev_m`) = the elevation where the outer (downstream) slope flattens, usually where the profile stops dropping steeply. Pick the same rule for every lake and write it down.
    - `height_m` = crest elevation minus toe elevation.
    - `width_m` = horizontal distance along the profile from the lake-side shoreline to the downstream toe.
    Record `dem_source` (for example `GLO-30 resampled to 10 m`) and enter one row in `dam`.
12. **Outlet channel.** Look at the imagery: is there a visible stream leaving the lake across or through the moraine? Yes: `has_outlet_channel = 1`. No: `0`. Imja Tsho is already marked as an outlet lake in `lakes.yaml`; check it. For outlet lakes the vertical freeboard is not meaningful (it is near zero by definition), and M7 puts that weight on the horizontal gap instead.
13. **Save** the GeoPackage. Commit nothing yet.

### 3.4 Check and convert to `lakes.yaml`
Run the helper from the repository root:

```
PYTHONUTF8=1 .venv\Scripts\python.exe scripts/crest_points_to_yaml.py data/qgis/crest_points.gpkg --out configs/crest_points_generated.yaml
```

It reads the geometry to get latitude and longitude (so there are no typing mistakes), reprojects to WGS84, and checks each lake:
- 5 to 10 points, exactly one flagged lowest, and that point really is the lowest (error otherwise);
- every point lies within the lake's `buffer_km` plus 1 km of its centre (catches a wrong CRS or the wrong lake);
- mean crest elevation within 300 m of `alt_m` in `lakes.yaml` (warning);
- dam height and width present and positive; one `dam` row per lake.

Lakes that pass are printed as YAML. **Review each block and paste it under the matching lake in `configs/lakes.yaml`, replacing `crest_points: []` and the `dam:` block.** The script never edits that file. Any lake that fails prints `ERROR:` lines and is left out; fix it in QGIS and rerun.

### 3.5 Common QGIS mistakes
- **Wrong CRS.** Points in a geographic CRS get wrong distances. Always digitise in the UTM project CRS; the helper converts to WGS84 itself.
- **Elevation typed from the wrong layer.** Always read it from the DEM at the point, not from contour labels.
- **Lowest point not on the crest.** A pit or a gully floor in the DEM can be lower than the crest. Look at it in the hillshade before accepting it.
- **Inconsistent toe rule.** Different toe definitions between lakes make heights incomparable. Use one rule.
- **The DEM predates the lake.** The Copernicus DEM was acquired about 2011 to 2015, so the elevation inside the lake shows the surface at that time. Use elevations along the shoreline and the crest, not inside the lake.

## 4. Step 7c: build the risk code

All risk parameters live in `configs/pipeline.yaml` (`risk:`, `volume_scaling:`). Nothing is hard-coded. Each module gets unit tests in `tests/` (only `test_auth.py` exists today).

### 4.1 Modules to write

**`src/models/freeboard_index.py`**
- Input: the predicted mask for one date, clipped to the fixed lake region, the lake's connected component nearest the lake centre; the crest points; the DEM.
- Shoreline: the boundary of that component. If it is empty (lake drained), return missing values, not zero.
- Horizontal gap: the minimum distance from the shoreline to any crest point, in metres (UTM).
- Vertical freeboard: lowest crest elevation minus the median DEM elevation along the shoreline.
- For `has_outlet_channel: true`: return only the horizontal gap; set the vertical value to missing.
- Return both, plus a data-quality flag (optical or radar-only mask). Radar-only masks are noisier; carry the flag through.

**`src/risk/trend_stats.py`**
- Rolling Mann-Kendall test and Sen's slope (`pymannkendall`) on the area series. Window `trend_window_days: 365`, minimum `trend_min_obs: 12`, both from config. Return missing when there are too few observations.

**`src/risk/area_volume_scaling.py`**
- Two relations, area in km², volume in km³:
  - Huggel et al. (2002): `V = 0.104 * A**1.42`
  - Wang et al. (2012): `V = 0.0354 * A**1.3724`
- Report volume as a range (low, high). Verify units against the original papers before trusting them; unit mix-ups are the usual bug.
- **Sanity check to put in a test.** With the 2022 South Lhonak area of 1.67 km² (my own arithmetic, recompute in code): Huggel gives about 0.215 km³ (215 million m³) and Wang about 0.072 km³ (72 million m³). The surveyed storage capacity is 65.8 ± 2.5 million m³. So the range 72 to 215 million m³ brackets it within about a factor of 3.3. If your numbers are off by a factor of 1000, a unit conversion is wrong.
- Known weakness: volume is a function of area, so it partly double-counts the area term. That is why its weight is the lowest. Use only relative changes.

**`src/risk/risk_score.py`** — define it exactly like this so results are reproducible:
1. **Area-growth z-score (weight 0.40).** Let the area series be a median of the last 3 scenes (reduces radar noise). Growth rate = Sen's slope of area over the trailing 90 days. Standardise it against the same lake's rates at the same time of year in **previous years only** (expanding baseline, no look-ahead): `z = (rate - mean_doy) / max(sd_doy, sd_floor)`, with `sd_floor` taken from measured pre-burst scene-to-scene noise (about 0.08 km² per scene at South Lhonak). With fewer than 3 earlier years, use the all-season spread and flag the score as low confidence.
2. **Freeboard component (weight 0.35).** The decline rate of the horizontal gap (m per year, trailing 365 days), standardised the same way. A shrinking gap raises risk. For outlet lakes use `outlet_lake_fallback` weights.
3. **Volume anomaly (weight 0.25).** Volume from both relations; standardise the relative change against the seasonal baseline as above.
4. **Composite** = weighted sum. If a component is unavailable (for example no crest points yet), renormalise the remaining weights to sum to 1 and write which components were used into the output, so scores are never silently compared across different recipes.
5. **EWMA** with `ewma_lambda: 0.2`. Scenes are irregular (6 to 12 days apart, with winter gaps), so make the smoothing time-aware: `lambda_eff = 1 - (1 - lambda) ** (dt / 6)` with `dt` in days; document this choice.
6. **CUSUM** with `k_sigma: 0.5`, `h_sigma: 4.5`: `S_t = max(0, S_{t-1} + (c_t - mu0) / sigma0 - k)`, alert when `S_t > h`, then reset. Estimate `mu0` and `sigma0` from an in-control baseline window defined before the evaluation (for example the first two years), never from the hindcast window.
7. Output per lake per date: the three z-scores, composite, EWMA, CUSUM, alert flag, the components used, and the data-quality flag. Write to `data/risk/{lake}_risk.csv`.

Unit tests to write first: a flat synthetic series gives scores near 0 and no alerts; a synthetic steady growth gives a positive area z; a missing freeboard component renormalises weights; CUSUM resets after an alert; outlet lakes use the fallback weights.

### 4.2 Confirm the weights (M7-B) and fix the success criteria
Before any hindcast run:
1. User confirms or changes the weights in `configs/pipeline.yaml`, and writes a short justification (needed for the methodology section). The starting values come from the literature.
2. Write `docs/PREREGISTRATION.md` and commit it. It must state: the lakes scored; the weights; EWMA and CUSUM parameters; the baseline windows; and the hindcast success criterion. Suggested criterion (adjust, then freeze): *South Lhonak's EWMA risk score ranks in the top two of the five scored lakes on at least 70% of dates in the six months before 2023-10-04, and the quiet lakes raise no more than one CUSUM alert per lake-year.* Report the lead time above the alert threshold, but never claim the event date was predicted.
3. Commit it before running M8's hindcast so the commit time shows the criterion was fixed first.

### 4.3 Run and check M7 (done when)
- A risk time series exists for every scored lake.
- South Lhonak's score is visibly elevated before October 2023. This is not guaranteed: the M6 work found no clear pre-burst anomaly from the forecaster, and a possible August to September 2023 rise (z about 3.3 over five earlier years) is only exploratory. If the score is not elevated, report that; do not retune.
- Quiet lakes (Imja Tsho, Tsho Rolpa, Lower Barun, Thulagi) show low, stable scores.
- Weight-grid robustness: rerun with a grid of plausible weights (for example each weight from 0.2 to 0.6 in steps of 0.1, summing to 1) and report how stable the five-lake ranking and the South Lhonak result are. Choose the final weights on literature grounds, and show the grid as a robustness check.

## 5. Milestone 8: evaluation

Goal: rigorous numbers for every pipeline component. Write `src/training/evaluate.py` to produce one results table; the training scripts already exist.

### 5.1 Segmentation
- **Leave-one-lake-out.** Six of 22 pool lakes are done (mean IoU 0.906, minimum 0.856). Run the remaining 16 on Kaggle (about 30 minutes each; see `kaggle/KAGGLE_TRAINING.md`): `... --split lake --holdout-lake GLO_... --epochs 40 --batch-size 16 --lr 3e-4`. Use `kaggle_dataset/splits.json` (key `lolo`) for the list; it is inside `glacierguards_kaggle.zip`.
- **Ablations (SAR-only, optical-only, fused).** Cross-validation versions exist (fused 0.911, optical 0.916, radar 0.819 mean IoU). The plan wants final models: train `--channels sar --split test` and `--channels optical --split test`. Report honestly: optical-only matched fused in clear conditions in our data, so the plan's expectation "fused should be best" did not hold on pooled IoU. Add a cloud stress test (hide the optical layers at random) to show where fusion matters.
- **Repeat seeds.** Three seeds for confidence intervals.
- **50-lake independent test and baselines.** See `docs/INDEPENDENT_TEST_AND_BASELINES.md`; its results go in the same evaluation report.

### 5.2 Hindcast case studies (use the criteria from `docs/PREREGISTRATION.md`)
1. **South Lhonak, 2023-10-04.** Does the risk score meet the criterion? Report the lead time above the CUSUM threshold, plus the burst detection result (flagged 6 Oct 2023, 0.70 km² drop, 0 false alarms in 61 pre-burst comparisons).
2. **Gongbatongsha Co, 2016-07-05.** Confirm its GLOID or inventory match first. Imagery is sparse (Sentinel-1A only, 12-day repeat, and Sentinel-2A from 2015). It may be a miss because the lake drained completely after a sudden debris-flow trigger; report that honestly.
3. **Imja Tsho, 2016 lowering (3.4 m).** Does the area, volume or gap proxy change as expected after the lowering? The change is small relative to the noise (about 0.08 km² scene to scene), so state what the data can and cannot resolve.
4. Tsho Rolpa (lowered about 3 m in 2000) predates Sentinel; Landsat only (`src/ingestion/landsat_fetch.py`).

### 5.3 False alarms
On the quiet lakes, report alerts per lake-year for the CUSUM trigger and for the drop detector, with the dates and what was found (scene problem, ice, or a real change).

### 5.4 Report table (the plan's format)

| Experiment | IoU | F1 | Precision | Recall |
|---|---|---|---|---|
| SAR-only | | | | |
| Optical-only | | | | |
| Fused | | | | |
| Leave-out: each lake | | | | |

Add hindcast and false-alarm tables. Do not retroactively tune weights to improve the hindcast.

## 6. Milestone 9: serving in Docker

Goal: the whole chain runs from imagery to a risk score behind a small API. Done when `docker-compose up` starts three containers with no errors and the API returns valid JSON for at least one lake. Do this only after M7 works end to end.

### 6.1 Components to write
- **`src/serving/inference_service.py`.** Given a lake id and a date: ingest scenes, stack, predict the mask (checkpoint path from an environment variable, CPU is enough), measure lake area in the fixed region, compute freeboard, update the risk series, write to SQLite at `data/glof_ews.db`. Reuse the existing scripts' functions; do not rewrite them.
- **Tables.** `lake_snapshots` (lake_id, date, area_km2, mask_source `optical` or `radar_only`, horizontal_gap_m, vertical_freeboard_m, volume_low_m3, volume_high_m3) and `risk_history` (lake_id, date, z_area, z_freeboard, z_volume, composite, ewma, cusum, alert, components_used). Keep the drop-detector alert as its own column or table so it is never mixed with the risk score.
- **`src/serving/api.py` (FastAPI).** `GET /lakes/{lake_id}/latest` (current mask summary and risk) and `GET /lakes/{lake_id}/history` (full series). Return 404 for an unknown lake and include `data_quality` and `components_used` in every response. Add `GET /health`.
- **Dockerfiles.** `docker/ingestion.Dockerfile` (runs the fetcher on a schedule), `docker/inference.Dockerfile`, `docker/api.Dockerfile` (uvicorn). Use CPU-only torch to keep images small.
- **`docker-compose.yml`.** Wires the three services, mounts the shared `data/` and `models/` folders as volumes, and sets the model path and database path through environment variables.

### 6.2 Security and practical points
- **Never bake credentials into an image.** Mount `ServiceAccountKey.json` and pass `.env` values at run time (compose `env_file` and a read-only volume). Add `.env` and the key file to `.dockerignore`.
- **Model weights are not in git.** Mount `models/seg_fused_test.pt` from a volume; document how to obtain it.
- Earth Engine rate limits and 503 errors happen: keep the retry loop in the ingestion service.
- On Windows use Docker Desktop with the WSL2 backend; if the build fails on path length or line endings, build from WSL.

### 6.3 Test it (M9-A)
```
docker-compose up --build
curl http://localhost:8000/health
curl http://localhost:8000/lakes/south_lhonak/latest
curl http://localhost:8000/lakes/south_lhonak/history
```
`latest` should return JSON with at least: `lake_id`, `date`, `area_km2`, `mask_source`, `risk` (composite, ewma, alert), `components_used`. Report any port or volume problems. Seed the database from the stored historical series (a one-off script) so `latest` has data without running a live download.

## 7. Milestone 10: documentation and deliverables

1. **README.** Setup; how to add a new lake (add it to `lakes.yaml`, download and stack, pick the checkpoint, run `make_lake_roi.py`, digitise crest points, run risk); how to retrain (Kaggle commands); how to read a risk score and its alert; the "make dataset" section already in the README.
2. **`provenance.csv`.** All datasets with dates, access route, licence and attribution text. Include the Third Pole inventory (confirm licence and citation), GLOID (confirm the DOI and year; see `DATA_PROVENANCE.md`) and Copernicus data statements.
3. **Evaluation report** from the M8 tables, with the limitations in `docs/M6_MULTI_LAKE_RESULTS.md` (sections 5 and 8).
4. **Demo notebook.** For one lake: plot the area series, the risk score and flagged alerts; for South Lhonak, mark the burst.
5. Final review by Aparna of the README and the methodology write-up before the project is called done.

## 8. Order of work and what blocks what

```
Now (no blocker):   1. area series for the 4 stable lakes (section 2)
                    2. 50-lake independent test + baselines (other doc)
Aparna (parallel):  3. QGIS crest points for the 5 lakes (section 3) -> crest_points_to_yaml.py -> lakes.yaml
                    4. confirm weights, write docs/PREREGISTRATION.md
After 1, 3, 4:      5. risk code + tests (section 4.1) -> M7 done-when (4.3)
After 5:            6. M8 evaluation (section 5)
After 5:            7. M9 Docker/API (section 6)
Last:               8. M10 documents (section 7)
```

M7 cannot finish without the crest points (step 3). Until then you can write and test the trend, volume and area-growth parts; the freeboard term stays empty and its weight is renormalised and flagged.

## 9. Honest limits to carry into every write-up
- One burst event: no false-alarm or lead-time statistics beyond one lake, until a second confirmed dated event is added.
- Detection is not prediction: the burst is seen at the first pass after it.
- Freeboard proxies use a 30 m DEM and a DEM older than the present lake; read them as trends.
- Risk-score weights and the EWMA and CUSUM values are textbook starting points, not fitted values; justify them and show the robustness grid.
- Gongbatongsha may be a miss; report it as one.
