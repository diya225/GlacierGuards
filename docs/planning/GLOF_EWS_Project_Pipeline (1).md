# Glacial Lake Outburst Flood (GLOF) Early Warning System
### Temporal, Self-Updating, Multi-Sensor CV Pipeline — Build Spec

This document is the single source of truth for building the project. It is the master instruction set, plus a checklist of things.

---

## 1. Project Summary

Build a pipeline that:
1. Ingests time-series Sentinel-1 SAR + Sentinel-2 optical + DEM data for a cohort of Himalayan glacial lakes.
2. Segments lake boundaries per time step with a U-Net (EfficientNet-B3 backbone).
3. Feeds the resulting area/geometry time series into a temporal anomaly model that self-updates its baseline (rather than a fixed threshold).
4. Tracks a moraine-dam "freeboard" proxy from DEM + lake boundary geometry.
5. Combines these into a composite, continuously updating **per-lake GLOF risk score**.
6. Serves results through a small automated (Dockerized) pipeline with a REST endpoint.

**Novelty claim to preserve throughout implementation:** existing public work does *static* single-epoch classification (Nature paper: Random Forest, single time step) or *temporal segmentation without a risk layer* (arXiv SAR U-Net paper), or *retrospective manual trend statistics* (EGUsphere PDGL paper). This project is the first to chain all three into one automated, self-updating pipeline. Don't let implementation drift back into "just a classifier" — the temporal anomaly + risk-scoring stage is the actual contribution.

---

## 2. Datasets — ALL FREE, NO PAID LICENSES

Every dataset below is free to access with a free account. **PlanetScope is deliberately excluded** from the core pipeline — it requires a commercial or research license through Planet Labs and is not guaranteed free access; it will not be used. (If you personally have an institutional Planet license later, it can be added as an optional Level-II enhancement, but the pipeline must fully function without it.)

| Dataset | Purpose | Access | Cost |
|---|---|---|---|
| **Sentinel-1 GRD (SAR)** | All-weather lake boundary detection | Copernicus Data Space Ecosystem (CDSE) or NASA Alaska Satellite Facility (ASF) | Free |
| **Sentinel-2 L2A (optical)** | Spectral indices (NDWI/NDSI/NDVI/NDGI) | Copernicus Data Space Ecosystem or Google Earth Engine (GEE) | Free |
| **Landsat 5/8/9** | Long historical baseline (pre-2015), fills gaps before Sentinel-2 existed | USGS EarthExplorer or Google Earth Engine | Free (you already have experience here) |
| **Copernicus DEM GLO-30** | Terrain correction for SAR, slope/aspect/elevation features, moraine-crest geometry | Copernicus Data Space Ecosystem or GEE | Free |
| **SRTM DEM (30m)** | Backup/alternate DEM, cross-check | USGS / GEE | Free |
| **NASA HyP3** | On-demand RTC (radiometric terrain correction) processing of Sentinel-1 GRD → analysis-ready rasters | Via ASF, tied to your NASA Earthdata login | Free (8,000 credits/month) |

**Recommendation: use Google Earth Engine (GEE) as the primary access + preprocessing layer.** It hosts Sentinel-1 GRD, Sentinel-2 L2A, Landsat Collection 2, and both SRTM and Copernicus DEM natively, with cloud-side compute so you don't need to download and preprocess terabytes locally. Use ASF/HyP3 only for the SAR RTC products where you need Enhanced-Lee-filtered, terrain-corrected outputs beyond what GEE's built-in filters provide.

---

## 3. Accounts & API Keys You Need to Create (Free)


1. **Google Earth Engine account** — https://earthengine.google.com/signup/ (free, needs a Google account; may need to register a Cloud Project, still free tier).
   - Output needed: a GEE **Cloud Project ID** and an authenticated service account JSON key (for non-interactive/server use) OR your personal OAuth token for interactive use.
2. **NASA Earthdata Login** — https://urs.earthdata.nasa.gov/ (free). Required for ASF Search API and HyP3.
   - Output needed: username + password (used as env vars; ASF's `asf_search` SDK and `hyp3_sdk` both authenticate with this).
3. **Copernicus Data Space Ecosystem account** — https://dataspace.copernicus.eu/ (free). Alternative/backup to ASF for Sentinel-1/2.
   - Output needed: client ID + client secret for their OAuth2 API if you use their API instead of GEE.
4. **USGS EarthExplorer account** — https://ers.cr.usgs.gov/register (free). Only needed if pulling Landsat outside GEE.
5. **(Optional) GitHub account** — for version control of the pipeline and Docker images.
6. **(Optional) Docker Hub account** — free tier, only if you want to push built images somewhere other than local/GitHub Container Registry.

Store all secrets in a local `.env` file (never commit it — add to `.gitignore`). Codex should generate a `.env.example` with placeholder keys.

---

## 4. Repository Structure 

```
glof-ews/
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
├── docker-compose.yml
├── data/
│   ├── raw/                  # untouched downloads (gitignored)
│   ├── interim/              # co-registered, tiled rasters
│   └── processed/            # final stacked tensors + ground-truth masks
├── configs/
│   ├── lakes.yaml            # cohort: name, lat, lon, altitude, moraine-crest reference points
│   └── pipeline.yaml         # dates, resolution, thresholds, model hyperparams
├── src/
│   ├── ingestion/
│   │   ├── gee_fetch.py           # pull S1/S2/Landsat/DEM via Earth Engine
│   │   ├── asf_hyp3_fetch.py      # SAR RTC via ASF + HyP3 for high-quality S1
│   │   └── landsat_fetch.py       # long-baseline historical pull
│   ├── preprocessing/
│   │   ├── sar_preprocess.py      # speckle filter, terrain correction, normalize to 8-bit
│   │   ├── optical_indices.py     # NDWI(blue/green), NDSI, NDVI, NDGI
│   │   ├── coregister_stack.py    # resample all layers to common grid, stack channels
│   │   └── tile_and_pad.py        # 256x256 tiling, zero-padding
│   ├── labeling/
│   │   └── qgis_export_helper.py  # converts manually digitized QGIS shapefiles to raster masks
│   ├── models/
│   │   ├── segmentation_model.py  # U-Net + EfficientNet-B3, composite BCE+Dice+Focal loss
│   │   ├── temporal_model.py      # ConvLSTM / temporal-attention anomaly head
│   │   └── freeboard_index.py     # DEM-vs-lake-boundary dam proxy computation
│   ├── risk/
│   │   ├── trend_stats.py         # Mann-Kendall + Sen's slope, rolling window
│   │   ├── area_volume_scaling.py # empirical area→volume conversion
│   │   └── risk_score.py          # composite EWMA/CUSUM self-updating score
│   ├── training/
│   │   ├── train_segmentation.py
│   │   ├── train_temporal.py
│   │   └── evaluate.py            # IoU, F1, precision, recall, leave-one-lake-out CV
│   └── serving/
│       ├── inference_service.py   # runs segmentation + temporal + risk on new imagery
│       └── api.py                 # FastAPI REST endpoint
├── docker/
│   ├── ingestion.Dockerfile
│   ├── inference.Dockerfile
│   └── api.Dockerfile
├── notebooks/
│   └── exploratory_analysis.ipynb
└── tests/
    └── ...
```

---

## 5. Build in This Order

These are the milestones to be achieved. Each milestone should end in a runnable, testable artifact before moving to the next — don't let it try to build everything at once.

### Milestone 1 — Environment & scaffolding
- Create the repo structure above.
- `requirements.txt` should include at minimum: `earthengine-api`, `asf_search`, `hyp3_sdk`, `rasterio`, `geopandas`, `numpy`, `torch`, `segmentation-models-pytorch` (has U-Net + EfficientNet-B3 out of the box), `scikit-image`, `pymannkendall` (for Mann-Kendall/Sen's slope), `fastapi`, `uvicorn`, `docker`, `python-dotenv`.
- Write `.env.example` with placeholders: `GEE_PROJECT_ID`, `GEE_SERVICE_ACCOUNT_JSON`, `EARTHDATA_USERNAME`, `EARTHDATA_PASSWORD`, `CDSE_CLIENT_ID`, `CDSE_CLIENT_SECRET`.
- Set up `configs/lakes.yaml` as an empty template — YOU fill in the actual lakes (see Section 7, Manual Task #1).

### Milestone 2 — Ingestion
- `gee_fetch.py`: authenticate with GEE, pull Sentinel-1 GRD (VV/VH), Sentinel-2 L2A (bands needed for NDWI/NDSI/NDVI/NDGI), Copernicus DEM, and Landsat Collection 2 for a given lake polygon + date range, export as GeoTIFFs to `data/raw/`.
- `asf_hyp3_fetch.py`: for lakes needing higher-quality RTC SAR (steep terrain, e.g. Rolwaling Valley-type sites), submit HyP3 batch RTC jobs via `asf_search` + `hyp3_sdk`, poll for completion, download.
- Both scripts must be resumable/idempotent (skip already-downloaded dates).

### Milestone 3 — Preprocessing
- `sar_preprocess.py`: Enhanced Lee speckle filter (7x7 window), normalize to 8-bit via histogram equalization/CDF mapping, preserve a nodata mask.
- `optical_indices.py`: implement
  - NDWI_blue = (B2 − B8) / (B2 + B8)
  - NDWI_green = (B3 − B8) / (B3 + B8)
  - NDVI = (B8 − B4) / (B8 + B4)
  - NDSI = (B3 − B10 or B11 depending on sensor) / (B3 + B10)
  - NDGI = (B3 − B4) / (B3 + B4)
- `coregister_stack.py`: resample everything to a common 10 m grid (bilinear), stack into a multi-channel tensor per date: [VV, VH, NDWI_blue, NDWI_green, NDVI, NDSI, NDGI, elevation, slope, aspect].
- `tile_and_pad.py`: crop to lake AOI, tile to 256×256, zero-pad boundary tiles.

### Milestone 4 — Ground truth labeling helper
- `qgis_export_helper.py`: takes hand-digitized polygon shapefiles (you make these in QGIS — see Manual Task #2) and rasterizes them to binary masks aligned to the same grid as the stacked tensors.

### Milestone 5 — Segmentation model
- `segmentation_model.py`: U-Net with EfficientNet-B3 encoder (ImageNet pretrained), input channels = however many are in your fused stack, output = single-channel binary mask.
- Loss = BCE + Dice + Focal, summed (not averaged) as three components.
- `train_segmentation.py`: standard train/val split at first, but Milestone 8 requires leave-one-lake-out CV — write the split logic to support both modes from the start.
- Target metrics to beat as a sanity check: Accuracy ~0.995+, IoU ~0.90+, F1 ~0.95+ on held-out data (these are the numbers the reference SAR paper achieved; your fused multi-sensor model should be at or above this).

### Milestone 6 — Temporal anomaly model
- `temporal_model.py`: a ConvLSTM (or attention layer over U-Net bottleneck features) that ingests the sequence of past N masks/embeddings and predicts the next expected extent.
- Output the **residual** (observed − predicted) as the raw anomaly signal per date.
- `train_temporal.py`: train on the historical sequence per lake; this is inherently a small-N-per-lake problem, so consider training one shared temporal model across all cohort lakes rather than per-lake models.

### Milestone 7 — Freeboard / dam-proxy + risk scoring
- `freeboard_index.py`: compute vertical/horizontal gap between segmented lake boundary and DEM-derived moraine-crest reference points (you provide these crest points manually — Manual Task #3).
- `trend_stats.py`: rolling-window Mann-Kendall trend test + Sen's slope on the area time series (use `pymannkendall`).
- `area_volume_scaling.py`: implement an empirical area→volume power-law scaling relationship (standard published glacial-lake scaling coefficients; cite the source you use in code comments).
- `risk_score.py`: composite score combining (a) z-score of area growth rate vs. that lake's own historical seasonal baseline, (b) freeboard decline rate, (c) volume anomaly — with an EWMA or CUSUM control-chart mechanism so the baseline itself updates over time rather than staying fixed. Weight parameters go in `configs/pipeline.yaml` so you can tune them without touching code.

### Milestone 8 — Evaluation
- `evaluate.py`: leave-one-lake-out cross-validation for segmentation; ablation runs (SAR-only vs optical-only vs fused) to quantify the fusion gain; a hindcast test against any lake in your cohort with a documented precursor/near-miss period (Manual Task #4 tells you where to find candidate events).

### Milestone 9 — Serving & Docker
- `inference_service.py`: given new imagery for a lake, run the full chain (preprocess → segment → update temporal state → freeboard → risk score) and write results to a small local datastore (SQLite is fine at this scale).
- `api.py`: FastAPI app exposing `GET /lakes/{lake_id}/latest` and `GET /lakes/{lake_id}/history`.
- Three Dockerfiles + a `docker-compose.yml` wiring: ingestion (cron-scheduled), inference, API — mirroring the three-microservice pattern from the reference SAR paper.

### Milestone 10 — Documentation
- Auto-generate a `README.md` covering setup, how to add a new lake to the cohort, how to retrain, and how to read the risk score output.

---

## 6. Deliverables Checklist (what "done" looks like)

At project completion you should be able to hand over / present:

1. **Codebase** — the full repo above, version-controlled, with passing tests.
2. **Trained model weights** — segmentation model (`.pt`/`.onnx`) and temporal model checkpoint.
3. **Ground-truth dataset** — your manually digitized masks + the raw imagery tiles they correspond to (document license/provenance for each source).
4. **Evaluation report** — IoU/F1/precision/recall table, leave-one-lake-out results, ablation table (SAR-only vs optical-only vs fused), and the hindcast case study writeup.
5. **Risk score time series** — CSV/DB export of the composite risk score per lake per date, plus the underlying area/volume/freeboard series.
6. **Running Docker stack** — `docker-compose up` should bring up ingestion + inference + API locally.
7. **API documentation** — endpoint list, example responses.
8. **Methodology write-up** — the accompanying document (already drafted separately) formatted as a report/paper section.
9. **A short demo** — either a notebook or a simple dashboard plotting a lake's area/risk-score time series with flagged anomaly points.
10. **Data provenance appendix** — list of exact datasets, dates, and access method used, for reproducibility.

---

## 7. Manual Tasks 

1. **Finalize your lake cohort** — pick 6–10 lakes, get their lat/lon/altitude, and fill in `configs/lakes.yaml`. Prioritize lakes with existing risk documentation (ICIMOD/national hazard inventories) so your hindcast validation has something to test against.
2. **Manually digitize ground-truth lake boundaries in QGIS** — for at least a representative subset of dates per lake, trace the lake outline against cloud-free optical imagery to create training/validation masks. This is unavoidable manual GIS work; there is no free public glacial-lake mask dataset with the temporal density this project needs.
3. **Identify moraine-crest reference points per lake** — using DEM contours or visual inspection in QGIS, mark the approximate crest line of each lake's containing moraine dam so `freeboard_index.py` has something to measure against.
4. **Research documented precursor/near-miss events** — for your hindcast validation, find at least one lake in your cohort (or a comparable one) with published records of pre-outburst swelling, so you can test whether your risk score would have flagged it in advance. National/ICIMOD glacial-lake hazard reports are a good starting point.
5. **Create the free accounts and generate the keys listed in Section 3**, and populate your real `.env` file (do not share this with Codex or commit it to version control).
6. **Decide and justify the risk-score weighting** — the composite formula in `risk_score.py` has tunable weights; these should reflect your judgment (or a literature-based justification) about which precursor signal matters most for moraine-dammed vs. ice-dammed lakes, not something an agent should invent unsupervised.
7. **Review and sign off on licensing/attribution** for every dataset used (all listed sources are free for research use, but each has its own attribution requirement — Sentinel data is EU Copernicus data, Landsat is USGS public domain, DEM sources have their own citation requirements). Keep the citations in your final report.
8. **Compute budget** — training the segmentation + temporal models needs a GPU. If you don't have local access, Google Colab's free tier is sufficient to start; upgrade to Colab Pro or a cloud GPU instance only if you hit time limits.

---

## 8. Notes on Fidelity to the Source Papers

- Segmentation architecture, loss function, and preprocessing steps mirror the temporal-first Sentinel-1 U-Net approach validated in the arXiv paper (IoU 0.913).
- Multi-sensor fusion (SAR + optical + DEM) and the specific NDWI/NDSI/NDVI/NDGI index formulas mirror the Himachal Pradesh Random Forest paper, adapted from point-based classification to full-tile segmentation input channels.
- Trend statistics (Mann-Kendall, Sen's slope) and area-volume scaling mirror the Nepal PDGL multi-decadal study, adapted from a retrospective annual analysis to a rolling, automated, self-updating computation.
