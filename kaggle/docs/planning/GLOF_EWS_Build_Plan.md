# GLOF Early Warning System — Master Build Plan

> **Project:** Glacial Lake Outburst Flood (GLOF) Early Warning System  
> **Architecture:** Temporal, Self-Updating, Multi-Sensor CV Pipeline  
> **Stack:** Python · PyTorch · Google Earth Engine · FastAPI · Docker  

---

## What We Are Building (in one paragraph)

A fully automated pipeline that ingests Sentinel-1 SAR + Sentinel-2 optical + DEM satellite data for 8–10 Himalayan glacial lakes, segments lake boundaries at every available time step using a U-Net (EfficientNet-B3 backbone), feeds the resulting area/geometry time series into a self-updating temporal anomaly model, tracks a moraine-dam freeboard proxy, and rolls everything into a composite per-lake GLOF risk score — served via a Dockerized REST API. The novelty is chaining segmentation + temporal anomaly + self-updating risk scoring into one automated pipeline; no existing public work does all three together.

---

## Phase 0 — Pre-Build Setup (Before Any Code Is Written)

These are purely manual tasks and decisions that **must be done before certain milestones can proceed**. I will call these out at each milestone as blockers.

### 0A · Accounts & API Keys (Blocks Milestone 2)
You need to create and verify these free accounts and generate credentials:

| # | Service | URL | What to get |
|---|---|---|---|
| 1 | NASA Earthdata | https://urs.earthdata.nasa.gov/ | Username + password |
| 2 | Google Earth Engine | https://earthengine.google.com/signup/ | GEE Cloud Project ID + service account JSON key |
| 3 | Copernicus Data Space | https://dataspace.copernicus.eu/ | OAuth client ID + secret (backup to GEE) |
| 4 | USGS EarthExplorer | https://ers.cr.usgs.gov/register | Username + password (backup for Landsat) |
| 5 | GitHub (optional) | https://github.com | Repo for version control |

> [!IMPORTANT]
> **Input needed from you (0A):** Once these accounts are created, populate a local `.env` file with the real keys. You never share these keys with me — I will only ever read `.env.example` placeholders. I will create the `.env.example` template in Milestone 1.

### 0B · Lake Cohort Finalization (Blocks Milestone 2)
Based on the Section 7 document, 8 core lakes + 2 control lakes have already been proposed:

| # | Lake | Role |
|---|---|---|
| 1 | South Lhonak (Sikkim) | Main hindcast (burst 2023-10-04) |
| 2 | Gongbatongsha Co (Tibet) | Second burst case (2016-07-05) |
| 3 | Imja Tsho (Khumbu, Nepal) | Known 3.4m lowering in 2016 |
| 4 | Tsho Rolpa (Rolwaling, Nepal) | Known 3m lowering in 2000 |
| 5 | Lower Barun (Nepal) | ICIMOD critical |
| 6 | Thulagi (Nepal) | ICIMOD critical |
| 7 | Lumding Tsho (Nepal) | ICIMOD critical |
| 8 | Hongu 2 / Chamlang South (Nepal) | ICIMOD critical |
| C1 | Quiet control | No burst, ICIMOD Rank III |
| C2 | Quiet control | No burst, ICIMOD Rank III |

> [!IMPORTANT]
> **Input needed from you (0B):** For each lake (especially lakes 5–8 and C1–C2), you need to download the ICIMOD/GLIMS outline, load it in QGIS, compute the centroid coordinates, and confirm altitude from the GLO-30 DEM. The South Lhonak and Imja coordinates are verified in the doc; all others need your verification. I will build the `lakes.yaml` template, but you fill in the real coordinates.

### 0C · Compute Platform Decision (Blocks Milestones 5–6)
Training requires ~20–30 GPU-hours. Options:
- **Kaggle Notebooks** (primary): ~30 hrs/week free T4/P100
- **Google Colab** (backup): free tier, sessions cut off unpredictably
- **Colab Pro**: only if you hit free limits

> [!IMPORTANT]
> **Input needed from you (0C):** Tell me which GPU platform you plan to use so I can configure the training scripts accordingly (file paths, Drive mounting, etc.).

---

## Milestone 1 — Repository Scaffold & Environment

**Goal:** A clean, runnable repo structure with all dependencies declared. Nothing fetches data yet.  
**Blocks on:** Nothing — this can start immediately.  
**Estimated time:** ~1–2 hours of coding.

### Steps:
1. **Create full directory tree** — all folders from the spec (`src/`, `configs/`, `data/`, `docker/`, `notebooks/`, `tests/`).
2. **`requirements.txt`** — pin all core packages:
   - `earthengine-api`, `asf_search`, `hyp3_sdk`
   - `rasterio`, `geopandas`, `shapely`
   - `numpy`, `scipy`, `scikit-image`
   - `torch`, `torchvision`, `segmentation-models-pytorch`
   - `pymannkendall`
   - `fastapi`, `uvicorn`
   - `python-dotenv`, `pyyaml`
   - `sqlalchemy` (for SQLite datastore in serving)
3. **`.env.example`** — template with all 6 placeholder keys.
4. **`.gitignore`** — ignore `data/raw/`, `data/interim/`, `data/processed/`, `.env`, `*.pt`, `*.onnx`.
5. **`configs/lakes.yaml`** — template structure with South Lhonak pre-filled as an example; other lakes left as empty stubs for you to complete.
6. **`configs/pipeline.yaml`** — all tunable parameters: resolution, date ranges, model hyperparams, risk-score weights (pre-filled from Section 7 Task 6 recommendations: `area_growth: 0.40`, `freeboard: 0.35`, `volume: 0.25`), EWMA/CUSUM defaults.
7. **`README.md`** — skeleton with sections to be filled in later.
8. **`provenance.csv`** — header-only file for dataset tracking.

### Inputs needed from you at this milestone:
> [!NOTE]
> **Input M1-A:** Confirm if you want a `pyproject.toml`/`setup.py` based project structure vs. a flat `requirements.txt` approach. (I recommend flat `requirements.txt` for simplicity.)  
> **Input M1-B:** Confirm the date range you want to initially cover. I'll default to **2015–2024** (full Sentinel era) but you may want to narrow this for initial testing.

### ✅ M1 Done When:
- `pip install -r requirements.txt` succeeds in a clean venv.
- All directories exist.
- `configs/lakes.yaml` has the correct template structure.

---

## Milestone 2 — Data Ingestion

**Goal:** Scripts that pull real satellite data for at least one lake.  
**Blocks on:** Accounts created (Phase 0A), lake coordinates filled (Phase 0B).

### Steps:

#### `src/ingestion/gee_fetch.py`
1. Authenticate with GEE using service account JSON (read from `.env`).
2. Define a lake AOI polygon from `lakes.yaml` coordinates.
3. Pull **Sentinel-1 GRD** (VV + VH bands), filtered by date range and orbit.
4. Pull **Sentinel-2 L2A** (B2, B3, B4, B8, B10/B11 for index computation).
5. Pull **Copernicus DEM GLO-30** for the AOI.
6. Pull **Landsat Collection 2** (L5/L8/L9) for the pre-2015 historical baseline.
7. Export all as cloud-optimized GeoTIFFs to `data/raw/{lake_id}/{sensor}/{date}.tif`.
8. **Idempotency:** skip dates already downloaded.

#### `src/ingestion/asf_hyp3_fetch.py`
1. Authenticate with NASA Earthdata credentials.
2. Search ASF for Sentinel-1 GRD scenes over the lake AOI and date range.
3. Submit HyP3 RTC batch jobs for high-priority lakes (Tsho Rolpa, South Lhonak — steep terrain).
4. Poll for job completion and download RTC products to `data/raw/{lake_id}/s1_rtc/`.
5. Idempotent: skip already-processed granules.

#### `src/ingestion/landsat_fetch.py`
1. GEE-based pull of Landsat 5 (1984–2013), Landsat 8 (2013–2021), Landsat 9 (2021–).
2. Cloud-mask using the QA_PIXEL band.
3. Export bands needed for optical indices.

### Inputs needed from you at this milestone:
> [!IMPORTANT]
> **Input M2-A:** Your `.env` file must be populated with real API keys. I will never see these — I just need you to confirm they work by running a test script I'll provide.  
> **Input M2-B:** I'll need you to run the fetch for South Lhonak first as a test case and confirm the downloads land correctly. This is a quick manual verification step.  
> **Input M2-C:** Confirm your HyP3 credit balance (8,000 free/month). For 10 lakes × ~100 dates, HyP3 may be used selectively; I'll target only the 2–3 steepest-terrain lakes for RTC jobs.

### ✅ M2 Done When:
- `gee_fetch.py --lake south_lhonak --start 2020-01-01 --end 2023-12-31` downloads GeoTIFFs without errors.
- File structure in `data/raw/` matches the spec.

---

## Milestone 3 — Preprocessing

**Goal:** Raw rasters → co-registered, normalized, 10-channel stacked tensors.  
**Blocks on:** M2 complete (raw data available).

### Steps:

#### `src/preprocessing/sar_preprocess.py`
1. Apply Enhanced Lee speckle filter (7×7 kernel).
2. Terrain-correct using DEM if using raw GRD (not needed for HyP3 RTC products).
3. Normalize VV/VH to 8-bit via CDF/histogram equalization.
4. Preserve a nodata mask.

#### `src/preprocessing/optical_indices.py`
Compute per-pixel indices from Sentinel-2 / Landsat bands:
- `NDWI_blue = (B2 − B8) / (B2 + B8)`
- `NDWI_green = (B3 − B8) / (B3 + B8)`
- `NDVI = (B8 − B4) / (B8 + B4)`
- `NDSI = (B3 − B11) / (B3 + B11)`
- `NDGI = (B3 − B4) / (B3 + B4)`

#### `src/preprocessing/coregister_stack.py`
1. Resample all layers to common 10 m grid (bilinear interpolation).
2. Stack into a 10-channel tensor per date:
   `[VV, VH, NDWI_blue, NDWI_green, NDVI, NDSI, NDGI, elevation, slope, aspect]`
3. Slope and aspect computed from DEM using `rasterio`/`numpy` gradient.
4. Output: `data/interim/{lake_id}/{date}_stack.tif`.

#### `src/preprocessing/tile_and_pad.py`
1. Crop stack to lake AOI with a buffer.
2. Tile into 256×256 patches.
3. Zero-pad boundary tiles.
4. Output: `data/processed/{lake_id}/{date}/tile_{row}_{col}.npy`.

### Inputs needed from you at this milestone:
> [!NOTE]
> **Input M3-A:** Verify that the stacked outputs look visually correct for at least one date using the exploratory notebook I'll set up. You'll need to open `notebooks/exploratory_analysis.ipynb` and confirm the channels look right (lake water clearly visible in NDWI, SAR shows expected backscatter contrast, etc.).

### ✅ M3 Done When:
- `data/processed/south_lhonak/` contains tiled `.npy` tensors for multiple dates.
- Channel-wise stats (min/max/mean) are reasonable (no all-zero or all-NaN bands).

---

## Milestone 4 — Ground-Truth Labeling Helper

**Goal:** Convert your manually digitized QGIS shapefiles into binary raster masks.  
**Blocks on:** M3 complete (need the co-registered grid as reference), and **your manual digitizing work**.

### Steps:

#### `src/labeling/qgis_export_helper.py`
1. Read a polygon shapefile (your QGIS output).
2. Rasterize to binary mask (0/1) aligned to the same grid and resolution as the stacked tensors.
3. Write mask to `data/processed/{lake_id}/{date}/mask.npy`.
4. Run a QA check: flag masks where lake area is < 10 pixels or > 95% of the tile (likely digitizing errors).

### 🔴 Critical manual work (no code can replace this):
> [!CAUTION]
> **Input M4-A (Major):** You need to digitize ~100 lake boundary masks manually in QGIS. The plan from Section 7 is:
> - ~12 dates per core lake, ~6 per control lake
> - ~30 person-hours total (15–20 min per mask)
> - Spread across pre-monsoon (Apr–May), post-monsoon (Oct–Nov), and some winter dates
> - Protocol: pick cloud-free S2 image → pair with nearest S1 (±3 days) → auto-draft with NDWI threshold → hand-correct in QGIS → export shapefile
> - 10% of masks must be double-labeled by a second person for annotation-noise QC
> - **Winter rule decision:** you need to agree as a team whether to label full lake outline even when frozen (Section 7 Task 2 recommendation: yes, include some winter dates)
> 
> **This is the most time-consuming manual step of the entire project. M5 (model training) cannot begin without sufficient masks.**

### ✅ M4 Done When:
- `qgis_export_helper.py` converts a test shapefile to a binary mask without errors.
- At least 20–30 masks are ready to begin early model training (full ~100 masks needed for final evaluation).

---

## Milestone 5 — Segmentation Model

**Goal:** A trained U-Net that segments lake water from multi-sensor stacked imagery.  
**Blocks on:** M3 + M4 (preprocessed tiles + masks), GPU access confirmed (Phase 0C).

### Steps:

#### `src/models/segmentation_model.py`
1. U-Net architecture with EfficientNet-B3 encoder from `segmentation-models-pytorch`.
2. Adapt first convolution layer to accept 10 input channels (vs. standard 3 RGB).
3. Single-channel binary output mask.
4. Loss function: `BCE + Dice + Focal` (summed, not averaged).

#### `src/training/train_segmentation.py`
1. Dataset loader that reads stacked `.npy` tiles and corresponding `mask.npy`.
2. Train/val split logic supporting two modes from the start:
   - **Standard:** random 80/20 split (for early development)
   - **Leave-one-lake-out:** hold entire lake out as test set (for final evaluation in M8)
3. Standard training loop with:
   - Mixed precision (`torch.cuda.amp`)
   - Checkpoint saving every epoch to Drive/Kaggle output
   - Learning rate scheduling (cosine annealing)
4. Log: per-epoch IoU, F1, BCE, Dice, Focal loss values.

### Target metrics to beat:
| Metric | Target |
|---|---|
| Accuracy | ≥ 0.995 |
| IoU | ≥ 0.90 |
| F1 | ≥ 0.95 |

### Inputs needed from you at this milestone:
> [!IMPORTANT]
> **Input M5-A:** Run training on your GPU platform (Kaggle/Colab) and report back the training curves. I'll help diagnose if IoU plateaus below 0.85.  
> **Input M5-B:** Confirm the compute platform and how to mount data (Kaggle dataset, Google Drive path, etc.) so I can configure the data loading paths correctly.

### ✅ M5 Done When:
- Model achieves IoU ≥ 0.90 on held-out validation tiles.
- Weights saved to `models/segmentation_checkpoint.pt`.

---

## Milestone 6 — Temporal Anomaly Model

**Goal:** A model that ingests a sequence of past lake masks and predicts the expected next extent; the residual is the anomaly signal.  
**Blocks on:** M5 complete (need trained segmentation + predicted masks for sequences).

### Steps:

#### `src/models/temporal_model.py`
1. ConvLSTM architecture (or attention layer over U-Net bottleneck features — I'll implement both and let you evaluate).
2. Input: sequence of N past predicted lake mask embeddings (or raw masks).
3. Output: predicted next mask extent.
4. **Residual = observed − predicted** → raw anomaly signal per date.

#### `src/training/train_temporal.py`
1. Train a **single shared model across all cohort lakes** (not per-lake — too few samples per lake).
2. Sequence construction: sliding window of length N (default N=6, configurable in `pipeline.yaml`).
3. Handle irregular time spacing (Sentinel-1 revisit varies; Sentinel-1B gap Dec 2021 → 1C).
4. Save temporal model checkpoint.

### Inputs needed from you at this milestone:
> [!NOTE]
> **Input M6-A:** Review the anomaly residuals visually for South Lhonak. Does the model show elevated residual (large positive anomaly) in the months leading up to October 2023? This is a sanity check of the temporal model before building the risk score on top of it. You'll run a provided evaluation notebook and share the plot.

### ✅ M6 Done When:
- Temporal model produces anomaly time series per lake.
- South Lhonak shows elevated anomaly in 2023 pre-burst window (qualitative check).

---

## Milestone 7 — Freeboard Proxy + Risk Scoring

**Goal:** Combine area growth, freeboard decline, and volume anomaly into the composite GLOF risk score.  
**Blocks on:** M5 + M6 (segmentation + temporal outputs), moraine crest points filled (Phase 0B extended task).

### Steps:

#### `src/models/freeboard_index.py`
1. Read segmented lake boundary polygon for each date.
2. Compute **horizontal gap**: distance from shoreline to nearest crest point.
3. Compute **vertical freeboard**: lowest crest elevation − median DEM elevation along shoreline.
4. For outlet lakes (`has_outlet_channel: true`): use horizontal gap only.

#### `src/risk/trend_stats.py`
1. Rolling-window Mann-Kendall trend test on area time series (`pymannkendall`).
2. Sen's slope computation.
3. Window size configurable in `pipeline.yaml`.

#### `src/risk/area_volume_scaling.py`
Implement **two** scaling relations (as per Section 7 Task 6):
- Huggel et al. (2002): `V = 0.104 · A^1.42`
- Wang et al. (2012): `V = 0.0354 · A^1.3724`
- Report volume as a range.
- **Sanity test:** South Lhonak should give ~65 million m³ ± order of magnitude.

#### `src/risk/risk_score.py`
Composite score:
1. **Area-growth z-score** (vs. lake's own seasonal historical baseline): weight 0.40
2. **Freeboard/horizontal-gap decline rate**: weight 0.35
3. **Volume anomaly**: weight 0.25
4. EWMA (`lambda=0.2`) over the composite to smooth noise.
5. CUSUM control chart (`k=0.5σ`, `h=4.5σ`) as the alert trigger.
6. All weights and EWMA/CUSUM parameters read from `configs/pipeline.yaml` — no hardcoding.

### Inputs needed from you at this milestone:
> [!IMPORTANT]
> **Input M7-A (Major):** Moraine crest points. Using QGIS + Copernicus GLO-30 DEM, for each lake you need to:
> - Identify the terminal moraine on the downstream side
> - Trace the crest as a polyline
> - Place 5–10 points along it (always include the lowest point)
> - Record lat, lon, DEM elevation for each point
> - Record dam height and width
> - Fill `crest_points:` section in `lakes.yaml`
> 
> I will provide a QGIS workflow guide and the exact YAML template to fill.

> [!IMPORTANT]
> **Input M7-B:** Risk-score weight review. Section 7 recommends `area_growth: 0.40, freeboard: 0.35, volume: 0.25`. Review the rationale and confirm or adjust. These should reflect your judgment, not default values I invented — your methodology write-up will need to justify them.

### ✅ M7 Done When:
- Risk score time series is produced for all lakes.
- South Lhonak's risk score is visibly elevated pre-October 2023.
- Quiet control lakes show low, stable risk scores.

---

## Milestone 8 — Evaluation

**Goal:** Rigorous quantitative validation of all pipeline components.  
**Blocks on:** M5, M6, M7 complete; ~100 ground-truth masks complete (M4).

### Steps:

#### `src/training/evaluate.py`
1. **Leave-one-lake-out cross-validation:** for segmentation; re-train with each lake held out, evaluate on the held-out lake.
2. **Ablation study** (3 runs):
   - SAR-only (VV, VH channels only)
   - Optical-only (NDWI, NDVI, NDSI, NDGI channels only)
   - Fused (all 10 channels) ← should be best
3. **Hindcast case study:**
   - South Lhonak: does risk score exceed threshold before 2023-10-04? Report lead time.
   - Gongbatongsha: does it detect the 2016-07-05 event? (May be a miss — report honestly.)
   - Imja Tsho: does volume/freeboard proxy drop after 2016 lowering intervention?
4. **False alarm rate:** report on quiet control lakes.

### Report table format I'll generate:

| Experiment | IoU | F1 | Precision | Recall |
|---|---|---|---|---|
| SAR-only | | | | |
| Optical-only | | | | |
| Fused (full) | | | | |
| Leave-out: South Lhonak | | | | |
| ... (one row per lake) | | | | |

### Inputs needed from you at this milestone:
> [!NOTE]
> **Input M8-A:** Review the hindcast result plots and help frame the success criteria narrative (Section 7 Task 4 guidance: success = "risk score in top tier of cohort for months before event"). Do not retroactively tune weights to make the hindcast look better.

### ✅ M8 Done When:
- Evaluation tables generated and results documented.
- Fused model outperforms SAR-only and optical-only on IoU/F1.

---

## Milestone 9 — Serving & Docker

**Goal:** Full end-to-end inference pipeline served as a REST API inside Docker containers.  
**Blocks on:** M7 complete (full pipeline working end-to-end).

### Steps:

#### `src/serving/inference_service.py`
1. Given a lake ID and a new imagery date, run:
   - Ingestion → preprocessing → segmentation → temporal update → freeboard → risk score
2. Write results to SQLite DB (`data/glof_ews.db`):
   - Tables: `lake_snapshots` (one row per lake per date), `risk_history`

#### `src/serving/api.py` (FastAPI)
Two endpoints:
- `GET /lakes/{lake_id}/latest` — current segmentation + risk score
- `GET /lakes/{lake_id}/history` — full time series

#### Three Dockerfiles:
- `docker/ingestion.Dockerfile` — runs `gee_fetch.py` on a cron schedule
- `docker/inference.Dockerfile` — runs `inference_service.py`
- `docker/api.Dockerfile` — serves `api.py` via uvicorn

#### `docker-compose.yml`:
- Wires all three containers together
- Mounts shared `data/` volume

### Inputs needed from you at this milestone:
> [!NOTE]
> **Input M9-A:** Run `docker-compose up` locally and hit `GET /lakes/south_lhonak/latest` — confirm you get a valid JSON response with a risk score. Report any Docker/port issues.

### ✅ M9 Done When:
- `docker-compose up` starts all three containers without errors.
- API returns valid JSON for at least one lake.

---

## Milestone 10 — Documentation & Final Deliverables

**Goal:** Everything packaged for handover/presentation.

### Steps:
1. Auto-generate final `README.md` covering:
   - Setup instructions
   - How to add a new lake to the cohort
   - How to retrain models
   - How to read the risk score output
2. Polish `provenance.csv` with all datasets, dates, access routes, licenses, attribution text.
3. Generate evaluation report artifact (tables from M8).
4. Package demo notebook: plot a lake's area + risk-score time series with flagged anomaly points.

### Inputs needed from you at this milestone:
> [!NOTE]
> **Input M10-A:** Review the final README and methodology write-up for accuracy before the project is considered done.

---

## Complete Input Dependency Map

Here is every point where I need something from you, in order:

| ID | When | What I Need From You | Blocks |
|---|---|---|---|
| 0A | Before M2 | API keys created + `.env` populated | M2 ingestion |
| 0B | Before M2 | Lake coordinates verified + `lakes.yaml` filled | M2 ingestion |
| 0C | Before M5 | GPU platform chosen (Kaggle/Colab) | M5–M6 training |
| M1-A | During M1 | Confirm `requirements.txt` flat vs. `pyproject.toml` | M1 |
| M1-B | During M1 | Confirm date range (default 2015–2024) | M1 |
| M2-A | During M2 | Confirm `.env` keys work (run test script) | M2 |
| M2-B | During M2 | Verify South Lhonak download is correct | M2 |
| M2-C | During M2 | Confirm HyP3 credit balance | M2 |
| M3-A | During M3 | Visual QC of stacked tensors in notebook | M3 |
| **M4-A** | **During M4** | **~100 manually digitized QGIS masks (~30 hrs)** | **M5 (critical)** |
| M5-A | During M5 | Share training curves from GPU run | M5 |
| M5-B | During M5 | Confirm GPU platform data paths | M5 |
| M6-A | During M6 | Visual check of anomaly residuals for S. Lhonak | M6 |
| **M7-A** | **During M7** | **Moraine crest points in QGIS for all lakes** | **M7 (critical)** |
| M7-B | During M7 | Review + confirm risk-score weights | M7 |
| M8-A | During M8 | Review hindcast results + success framing | M8 |
| M9-A | During M9 | Run `docker-compose up` and test API locally | M9 |
| M10-A | During M10 | Final README + report review | Done |

> [!CAUTION]
> **Two inputs are critical path blockers with no workaround:**  
> - **M4-A** (manual QGIS digitizing, ~30 hrs) blocks all model training  
> - **M7-A** (moraine crest points in QGIS) blocks freeboard and final risk score

---

## Suggested Execution Order

```
Phase 0 (parallel, you do this while I build M1):
  ├── Create all API accounts → populate .env
  └── Start verifying lake coordinates in QGIS

M1 → Scaffold (I build, ~1 day)
  └── No inputs needed upfront

M2 → Ingestion (I build, you test, ~2–3 days)
  └── Requires: 0A + 0B complete

M3 → Preprocessing (I build, you QC, ~2 days)
  └── Requires: M2 downloads available

M4 → Labeling helper (I build the script, you do the ~30 hrs digitizing)
  └── This can run in parallel with M5 early development

M5 → Segmentation model (I build, you train on GPU, ~1 week including GPU runs)
  └── Requires: M3 complete + at least 20–30 early masks from M4

M6 → Temporal model (I build, you train, ~3–5 days)
  └── Requires: M5 complete

M7 → Risk scoring (I build, requires crest points from you)
  └── Requires: M6 + M7-A crest points

M8 → Evaluation (I build + run, ~2–3 days)
  └── Requires: M7 + all 100 masks from M4

M9 → Docker + API (I build, you test locally, ~2 days)
  └── Requires: M7 working end-to-end

M10 → Docs (I write, you review, ~1 day)
```

---

## What I Will Build vs. What You Must Do

| Category | Me (Agent) | You (Manual) |
|---|---|---|
| All Python code | ✅ | |
| Dockerfiles + compose | ✅ | |
| Config templates | ✅ (templates) | ✅ (fill real values) |
| API accounts | | ✅ |
| `.env` population | | ✅ |
| Lake coordinate verification | | ✅ (QGIS) |
| QGIS mask digitizing (~100 masks) | | ✅ (~30 hrs) |
| Moraine crest points | | ✅ (QGIS) |
| GPU training runs | | ✅ (Kaggle/Colab) |
| Training curve review | ✅ (I diagnose) | ✅ (you report) |
| Hindcast interpretation | ✅ (I generate plots) | ✅ (you review narrative) |
| Licensing/attribution text | ✅ (template) | ✅ (verify each source) |
| Final report sign-off | ✅ (I draft) | ✅ (you review) |
