# GLOF Early Warning System

> **Temporal, Self-Updating, Multi-Sensor CV Pipeline for Himalayan Glacial Lakes**

A fully automated pipeline that:
1. Ingests Sentinel-1 SAR + Sentinel-2 optical + DEM data for a cohort of Himalayan glacial lakes
2. Segments lake boundaries at each time step with a U-Net (EfficientNet-B3 backbone)
3. Feeds the resulting area/geometry time series into a temporal anomaly model that self-updates its baseline
4. Tracks a moraine-dam freeboard proxy from DEM + lake boundary geometry
5. Combines these into a composite, continuously updating **per-lake GLOF risk score**
6. Serves results via a Dockerised REST API

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure credentials
Copy `.env.example` to `.env` and fill in your API keys:
```bash
cp .env.example .env
# Edit .env with your real credentials
```

### 3. Register your GEE service account
Place `ServiceAccountKey.json` in the project root (downloaded from Google Cloud Console).

### 4. Test authentication
```bash
python tests/test_auth.py
```

### 5. Download data for a lake
```bash
python src/ingestion/gee_fetch.py --lake south_lhonak --start 2022-01-01 --end 2023-12-31
```

### 6. Preprocess
```bash
python src/preprocessing/sar_preprocess.py --lake south_lhonak
python src/preprocessing/optical_indices.py --lake south_lhonak
python src/preprocessing/coregister_stack.py --lake south_lhonak
python src/preprocessing/tile_and_pad.py --lake south_lhonak
```

---

## How to Add a New Lake

1. Add an entry to `configs/lakes.yaml` following the existing template.
2. Verify coordinates using QGIS and the ICIMOD/GLIMS outlines.
3. Run `gee_fetch.py --lake <new_lake_id>` to download data.
4. Follow the labeling workflow: `python src/labeling/qgis_export_helper.py --help-qgis`
5. Add moraine crest points to the `crest_points:` field in `lakes.yaml`.

---

## Repository Structure

```
glacier_guards/
├── configs/
│   ├── lakes.yaml          # Lake cohort: coordinates, dam type, crest points
│   └── pipeline.yaml       # All tunable hyperparameters
├── src/
│   ├── ingestion/          # GEE + ASF/HyP3 + Landsat data download
│   ├── preprocessing/      # SAR filter, indices, stacking, tiling
│   ├── labeling/           # QGIS shapefile → binary mask converter
│   ├── models/             # U-Net segmentation + ConvLSTM temporal model
│   ├── risk/               # Mann-Kendall, area-volume, EWMA/CUSUM risk score
│   ├── training/           # Training loops + evaluation scripts
│   ├── serving/            # FastAPI endpoint + inference service
│   └── utils/              # Auth, config loaders
├── docker/                 # Dockerfiles for ingestion/inference/API
├── data/
│   ├── raw/                # Downloaded satellite data (gitignored)
│   ├── interim/            # Co-registered stacked tensors
│   └── processed/          # Tiles + ground-truth masks
└── tests/
```

---

## Risk Score

The composite GLOF risk score combines:
- **Area-growth anomaly** (z-score vs lake's own seasonal baseline) — weight 0.40
- **Freeboard / horizontal-gap decline rate** — weight 0.35
- **Volume anomaly** (area→volume power law) — weight 0.25

Smoothed via EWMA (λ=0.2) with CUSUM alert trigger (k=0.5σ, h=4.5σ).
All weights are configurable in `configs/pipeline.yaml`.

---

## Data Sources

| Dataset | Purpose | License |
|---|---|---|
| Sentinel-1 GRD | SAR lake boundary detection | Copernicus open |
| Sentinel-2 L2A | Optical indices | Copernicus open |
| Copernicus DEM GLO-30 | Terrain correction + freeboard | Copernicus open |
| Landsat 5/8/9 | Pre-2015 historical baseline | USGS public domain |
| NASA HyP3 | High-quality SAR RTC for steep terrain | Free with Earthdata |

---

*See `provenance.csv` for full data provenance and attribution.*

## make dataset (GLOID training set for Kaggle)
Requires `.env` GEE credentials and the GLO/GLOID gpkg files in `data/external/gloid/` (Zenodo 10.5281/zenodo.19370146).
```
python scripts/list_candidate_lakes.py --n 30            # select lakes -> configs/training_lakes.csv
python src/ingestion/gloid_fetch.py                      # download composites (resumable; failures in data/raw/_gloid_manifest.csv)
python src/preprocessing/gloid_stack_tile.py             # stacks (data/interim) + tiles (data/processed)
python scripts/gloid_to_masks.py                         # rasterize GLOID polygons -> mask.npy, reports/mask_manifest.csv
python scripts/dataset_report.py                         # QA numbers, fails on misalignment/NaN
python scripts/qa_overlays.py                            # overlays -> reports/qa/
python scripts/build_kaggle_dataset.py                   # splits + glacierguards_kaggle.zip
python scripts/verify_package.py                         # clean unzip + 2 training steps
```
Training instructions: `kaggle/KAGGLE_TRAINING.md` and `kaggle/kaggle_train.ipynb`.
