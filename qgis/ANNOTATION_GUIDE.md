# Moraine Dam Annotation Guide (QGIS Workflow)

> [!WARNING]
> **Remote-Sensing Interpretation Disclaimer:** All annotations and DEM-derived elevations produced by this workflow are manual interpretations of satellite imagery and remotely sensed elevation models (e.g., ~30 m native Copernicus GLO-30 DEM). They are **NOT** field-surveyed measurements and must not be treated as engineering safety margins or absolute freeboard measurements.

This guide explains how to manually digitize the moraine dam for **South Lhonak Lake** (and any other lake in `configs/lakes.yaml`) in QGIS, extract DEM elevations, validate the GeoPackage, and update the lake configuration for Milestone 7.

---

## 1. Prepare and Open the QGIS Project

1. Ensure your Python environment has the project dependencies installed (`pip install -r requirements.txt`) and that satellite/DEM data for the lake has been fetched (`data/raw/{lake_id}/dem/dem.tif`, `data/raw/{lake_id}/s2/`, and/or `data/interim/{lake_id}/stacked/`).
2. Run the preparation script from the repository root:
   ```bash
   python src/labeling/qgis_moraine_prep.py --lake south_lhonak
   ```
   *(To regenerate and repair layers while keeping an automatic timestamped backup of any previous annotations, pass `--force-recreate`)*:
   ```bash
   python src/labeling/qgis_moraine_prep.py --lake south_lhonak --force-recreate
   ```
   This script:
   - Inspects the lake's rasters to determine the actual CRS (e.g. `EPSG:32645` for South Lhonak).
   - Generates a DEM hillshade at `data/labels/manual/south_lhonak/working/south_lhonak_hillshade.tif` (when `dem.tif` is present).
   - Creates a 3-layer spatial GeoPackage at `data/labels/manual/south_lhonak/south_lhonak_moraine.gpkg` with concrete geometry types (`dam_footprint`: Polygon, `crest_line`: LineString, `crest_points`: Point) in `EPSG:32645` (backing up any existing file with `backup_<timestamp>.gpkg` first).
   - Validates that the layers have concrete spatial types (not `NoGeometry`), correct CRS, and capability to add features.
   - Builds/updates `qgis/south_lhonak_annotation.qgz` with preconfigured symbology, `wkbType` tags, and attribute forms.
3. Open `qgis/south_lhonak_annotation.qgz` in **QGIS 3.28+**.
   - If you also want a high-resolution basemap for visual context only, add one via **Layer → Add Layer → Add XYZ Layer** (do not trace directly from third-party basemaps without checking terms of use).
   - Optionally generate 10 m contours from the DEM (**Raster → Extraction → Contour**, interval `10.0`) to help read ridge lines.

---

## 2. Identify and Digitize the Moraine Dam Footprint (`dam_footprint`)

1. Locate the terminal moraine ridge on the outlet side of the lake using the DEM, hillshade, and Sentinel-2 imagery. Contours typically run parallel to the ridge and bend downslope on the downstream face.
2. Select the **`south_lhonak_dam_footprint`** layer (Polygon) and click **Toggle Editing** (pencil icon).
3. Click **Add Polygon Feature** and digitize the approximate spatial extent of the moraine dam (from the lake-side shoreline contact to the downstream toe and between the lateral moraine flanks).
4. Fill in the attributes:
   - `annotation_id`: e.g. `sl_dam_01`
   - `lake_id`: `south_lhonak` (pre-filled default)
   - `confidence`: `1` (High), `2` (Medium), or `3` (Low)
   - `notes`: brief description of visible outlet/toe criteria used.

---

## 3. Digitize the Moraine Crest Line (`crest_line`)

1. Select **`south_lhonak_crest_line`** (LineString) and click **Toggle Editing**.
2. Click **Add Line Feature** and trace along the highest continuous ridge line of the terminal moraine crest, from one lateral abutment across the outlet sector to the opposite abutment.
3. Do not snap vertices to DEM pixel centres; follow the geomorphological crest visible across the hillshade, contours, and imagery.
4. Fill in:
   - `annotation_id`: e.g. `sl_crest_01`
   - `lake_id`: `south_lhonak`
   - `confidence`: `1`, `2`, or `3`
   - `notes`: any ambiguity (e.g. braided outlet, snow cover).

---

## 4. Place Crest Points (`crest_points`)

Place points along the digitized crest line (typically **5–10 points total**: endpoints, lowest saddle, any local depressions, and 3–5 representative crest points).

1. Select **`south_lhonak_crest_points`** (Point) and enable **Snapping** to `crest_line` if helpful so points sit cleanly on the crest.
2. Click **Toggle Editing** → **Add Point Feature**.
3. For each point, fill in:
   - `point_id`: unique ID, e.g. `sl_cp_01`, `sl_cp_02`, …
   - `lake_id`: `south_lhonak`
   - `point_type`: select from dropdown:
     - `crest_endpoint` — start/end abutment of the crest line
     - `lowest_saddle` — the single lowest point along the crest (potential overtopping/breach saddle)
     - `local_depression` — secondary low notch or swale along the crest
     - `representative_crest` — evenly spaced points characterizing typical crest height
   - `confidence`: `1`, `2`, or `3`
   - `elevation_m`: leave blank during digitizing (it will be sampled reproducibly from the DEM in Step 7).
   - `notes`: optional observation.

---

## 5. Identify the Lowest Saddle (`lowest_saddle`)

1. Use the QGIS **Elevation Profile** tool (**View → Panels → Elevation Profile**) along `crest_line` on the DEM, or visually inspect the DEM + hillshade + 10 m contours along the crest.
2. Verify that the lowest elevation along the crest is genuinely on the moraine crest (not a DEM pit inside the lake or down a steep downstream gully).
3. Place **exactly one** point with `point_type = lowest_saddle` at that location. It will display as a red star in QGIS.

---

## 6. Save Edits

For **each** of the three layers (`dam_footprint`, `crest_line`, `crest_points`):
1. Click **Save Layer Edits** (disk icon).
2. Click **Toggle Editing** off so changes are flushed to `data/labels/manual/south_lhonak/south_lhonak_moraine.gpkg`.
3. Save the QGIS project (`Ctrl+S`) if you adjusted view settings.

---

## 7. Run the DEM Extraction, Validation, and Export Workflow

From the repository root, run `qgis_export_helper.py` with `--extract-dem` and `--save-extracted-gpkg`:

```bash
python src/labeling/qgis_export_helper.py ^
    --lake south_lhonak ^
    --moraine-gpkg data/labels/manual/south_lhonak/south_lhonak_moraine.gpkg ^
    --dem data/raw/south_lhonak/dem/dem.tif ^
    --extract-dem ^
    --save-extracted-gpkg
```

This command:
1. Transforms `crest_points` coordinates into the DEM's CRS, samples elevations in metres (respecting NoData and reporting any out-of-bounds/invalid samples instead of silently assigning `0`), and preserves the original point geometries.
2. Validates all three layers (non-empty, CRS defined, expected geometry types, geometry validity without silent modification, required attributes, unique `point_id`, valid `elevation_m`, and exactly one `lowest_saddle`).
3. Writes:
   - `data/labels/manual/south_lhonak/south_lhonak_annotation_report.json`
   - `data/labels/manual/south_lhonak/south_lhonak_annotation_metadata.json`
   - `data/labels/manual/south_lhonak/south_lhonak_lakes_patch.yaml`

---

## 8. Interpret the Generated Report

Open `data/labels/manual/south_lhonak/south_lhonak_annotation_report.json`:
- **`valid`**: must be `true` before using the annotations in M7.
- **`errors` / `warnings`**: fix any listed geometry, attribute, duplicate ID, or missing `lowest_saddle` issues in QGIS and re-run Step 7.
- **`metrics`**:
  - `dam_footprint_area_m2`: total digitized dam footprint area in m² (computed in UTM).
  - `crest_line_length_m`: total digitized crest length in m (computed in UTM).
  - `crest_elevation_min_m`, `crest_elevation_max_m`, `crest_elevation_mean_m`: summary statistics of DEM-sampled elevations across all crest points.
  - `lowest_saddle_elevation_m`: DEM-sampled elevation of the `lowest_saddle` point (kept separate from the mean so M7 can track vertical freeboard relative to the saddle independently).
  - `num_crest_points`: total annotated crest points.

---

## 9. Update the Lake Configuration (`configs/lakes.yaml`)

Once `valid: true`, review `data/labels/manual/south_lhonak/south_lhonak_lakes_patch.yaml` and either:
- Paste the `crest_points:` and `moraine_annotation:` blocks under `south_lhonak` in `configs/lakes.yaml`, **or**
- Pass `--apply-to-lakes-yaml` to `qgis_export_helper.py` to surgically update only `south_lhonak` while preserving all comments and other lakes:
  ```bash
  python src/labeling/qgis_export_helper.py ^
      --lake south_lhonak ^
      --moraine-gpkg data/labels/manual/south_lhonak/south_lhonak_moraine.gpkg ^
      --apply-to-lakes-yaml
  ```
