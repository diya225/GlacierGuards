"""
qgis_export_helper.py — Convert hand-digitized QGIS shapefiles → binary raster masks.

After you digitize lake boundaries in QGIS and export them as shapefiles (.shp),
run this script to rasterize them onto the same grid as the stacked tensors.
The output is a binary mask (0 = not-lake, 1 = lake) saved as a NumPy array
alongside the tile patches, ready for training.

Usage
-----
    # Convert ONE shapefile for a specific lake+date:
    python src/labeling/qgis_export_helper.py \\
        --lake south_lhonak \\
        --date 2023-06-15 \\
        --shapefile path/to/south_lhonak_2023-06-15.shp

    # Batch-convert a whole folder of shapefiles:
    python src/labeling/qgis_export_helper.py \\
        --lake south_lhonak \\
        --shapefile-dir path/to/shapefiles/

    # Run QA check on all masks for a lake:
    python src/labeling/qgis_export_helper.py --lake south_lhonak --qa-only

QGIS workflow (step-by-step for non-GIS users)
----------------------------------------------
See the printed instructions below when you run --help-qgis.

Output
------
    data/processed/{lake_id}/{YYYY-MM-DD}/mask.npy    Binary uint8 (0/1)
    data/processed/{lake_id}/{YYYY-MM-DD}/mask_qa.json QA stats
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import geopandas as gpd
from shapely.geometry import mapping
from tqdm import tqdm

from src.utils.config_loader import get_lake_by_id, get_active_lakes, load_pipeline_config

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# QA thresholds
QA_MIN_LAKE_PIXELS = 10          # Flag mask if lake is fewer than this many pixels
QA_MAX_LAKE_FRACTION = 0.95      # Flag mask if lake covers > 95% of the tile


QGIS_INSTRUCTIONS = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 HOW TO DIGITIZE LAKE BOUNDARIES IN QGIS (STEP-BY-STEP)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

What you need installed:
  - QGIS (free) from https://qgis.org/

For each lake + date you want to label:

STEP 1 — Open QGIS and load your base imagery
  a) In QGIS menu: Layer → Add Layer → Add Raster Layer
  b) Browse to the stacked .tif file:
       data/interim/{lake_id}/stacked/{YYYY-MM-DD}.tif
  c) In the Layer Styling panel (F7), set the RGB to:
       Red  = Band 3 (NDWI_blue)
       Green= Band 4 (NDWI_green)
       Blue = Band 7 (NDGI)
     This makes water appear bright and easy to trace.

STEP 2 — Generate a rough draft polygon with NDWI threshold
  a) Go to: Raster → Raster Calculator
  b) Formula: ("stacked@3" > 0.1)   [where 0.1 is the NDWI threshold]
  c) This creates a binary raster. Convert it to a polygon:
     Raster → Conversion → Polygonize (Raster to Vector)
  d) Use the result as a starting shape — you will hand-correct it next.

STEP 3 — Create a new vector layer for the lake boundary
  a) Layer → Create Layer → New Shapefile Layer
  b) File name: {lake_id}_{YYYY-MM-DD}.shp
  c) Geometry type: Polygon
  d) CRS: Same as the stacked raster (check bottom-right of QGIS window)
  e) Add a field: Name="lake_id", Type=Text

STEP 4 — Digitize / hand-correct the polygon
  a) Select your new shapefile layer
  b) Toggle Editing: pencil icon or Layer → Toggle Editing
  c) Add Feature: the green pencil with a polygon icon
  d) Trace around the visible lake boundary by clicking points.
     Right-click when done to close the polygon.
  e) Correct the NDWI draft if you used one:
     - Use the "Vertex Editor" tool (right-click feature → Vertex Editor)
     - Drag vertices to better match the lake edge
  f) Save edits: disk icon or Layer → Save Layer Edits
  g) Turn off editing: pencil icon again

STEP 5 — Export the shapefile
  a) Right-click the layer → Export → Save Features As
  b) Format: ESRI Shapefile
  c) File name: data/labels/raw/{lake_id}_{YYYY-MM-DD}.shp
  d) CRS: Keep as-is (same as raster)
  e) Click OK

STEP 6 — Run this script to convert to binary mask
  python src/labeling/qgis_export_helper.py \\
      --lake {lake_id} \\
      --date {YYYY-MM-DD} \\
      --shapefile data/labels/raw/{lake_id}_{YYYY-MM-DD}.shp

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 TIPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 • Winter dates: trace the full lake outline even if frozen or snow-covered
 • When Sentinel-2 is cloudy, load the paired SAR file for visual guidance
   (data/raw/{lake_id}/s1/{S1-date}.tif) — water appears dark in SAR
 • Aim for 12 dates per core lake spread across seasons and years
 • 10% of masks need a second person to independently re-digitize (for QA)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


# ─────────────────────────────────────────────────────────────────────────────
# Core rasterization function
# ─────────────────────────────────────────────────────────────────────────────

def shapefile_to_mask(
    shp_path: Path,
    reference_tif: Path,
) -> np.ndarray:
    """
    Rasterize a polygon shapefile onto the same grid as a reference GeoTIFF.

    Args:
        shp_path:      Path to the digitized shapefile (.shp).
        reference_tif: Path to the reference raster (stacked .tif) whose
                       transform/CRS/shape the mask will match.

    Returns:
        Binary uint8 mask, shape (H, W), values 0 (not-lake) or 1 (lake).

    Raises:
        FileNotFoundError: if either file is missing.
        ValueError:        if the shapefile has no features.
    """
    if not shp_path.exists():
        raise FileNotFoundError(f"Shapefile not found: {shp_path}")
    if not reference_tif.exists():
        raise FileNotFoundError(
            f"Reference raster not found: {reference_tif}\n"
            f"Run coregister_stack.py for this lake+date first."
        )

    gdf = gpd.read_file(shp_path)
    if gdf.empty:
        raise ValueError(f"Shapefile has no features: {shp_path}")

    with rasterio.open(reference_tif) as src:
        ref_crs = src.crs
        ref_transform = src.transform
        ref_shape = (src.height, src.width)

    # Reproject shapefile to match raster CRS if needed
    if gdf.crs is None:
        logger.warning(
            "Shapefile has no CRS defined. Assuming it matches the raster CRS: %s",
            ref_crs,
        )
        gdf = gdf.set_crs(ref_crs)
    elif gdf.crs != ref_crs:
        logger.info(
            "Reprojecting shapefile from %s to %s …", gdf.crs, ref_crs
        )
        gdf = gdf.to_crs(ref_crs)

    # Rasterize — burn 1 for lake, 0 elsewhere
    shapes = [(geom, 1) for geom in gdf.geometry if geom is not None]
    if not shapes:
        raise ValueError("No valid geometries found in the shapefile.")

    mask = rasterize(
        shapes=shapes,
        out_shape=ref_shape,
        transform=ref_transform,
        fill=0,       # background
        dtype=np.uint8,
        all_touched=False,  # only pixels whose centre falls inside the polygon
    )
    return mask


def save_mask(
    mask: np.ndarray,
    lake_id: str,
    date_str: str,
    shp_path: Path,
) -> dict:
    """
    Save a binary mask as .npy and write QA stats JSON.

    Returns:
        QA stats dict.
    """
    out_dir = DATA_PROCESSED / lake_id / date_str
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_path = out_dir / "mask.npy"
    np.save(mask_path, mask)

    # ── QA statistics ─────────────────────────────────────────────────────────
    h, w = mask.shape
    lake_pixels = int(np.sum(mask == 1))
    total_pixels = h * w
    lake_fraction = lake_pixels / total_pixels

    qa = {
        "lake_id": lake_id,
        "date": date_str,
        "source_shapefile": str(shp_path),
        "mask_shape": [h, w],
        "lake_pixels": lake_pixels,
        "total_pixels": total_pixels,
        "lake_fraction": round(lake_fraction, 4),
        "qa_warnings": [],
    }

    if lake_pixels < QA_MIN_LAKE_PIXELS:
        msg = (
            f"Lake covers only {lake_pixels} pixels (< {QA_MIN_LAKE_PIXELS} threshold). "
            "Likely a digitizing error or very small/frozen lake."
        )
        qa["qa_warnings"].append(msg)
        logger.warning("[%s %s] QA WARNING: %s", lake_id, date_str, msg)

    if lake_fraction > QA_MAX_LAKE_FRACTION:
        msg = (
            f"Lake covers {lake_fraction:.1%} of the tile (> {QA_MAX_LAKE_FRACTION:.0%} threshold). "
            "Check if the AOI buffer is too small."
        )
        qa["qa_warnings"].append(msg)
        logger.warning("[%s %s] QA WARNING: %s", lake_id, date_str, msg)

    if not qa["qa_warnings"]:
        logger.info(
            "[%s %s] Mask saved ✓ — %d lake pixels (%.1f%% of tile)",
            lake_id, date_str, lake_pixels, lake_fraction * 100,
        )

    qa_path = out_dir / "mask_qa.json"
    with open(qa_path, "w") as fh:
        json.dump(qa, fh, indent=2)

    return qa


# ─────────────────────────────────────────────────────────────────────────────
# Batch processing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_date_from_stem(stem: str) -> str | None:
    """Try to extract YYYY-MM-DD from a filename stem."""
    match = re.search(r"\d{4}-\d{2}-\d{2}", stem)
    return match.group(0) if match else None


def batch_convert_dir(lake_id: str, shp_dir: Path) -> None:
    """
    Convert all shapefiles in shp_dir that match the pattern
    {lake_id}_{YYYY-MM-DD}.shp (or just {YYYY-MM-DD}.shp).
    """
    shp_files = sorted(shp_dir.glob("*.shp"))
    if not shp_files:
        logger.warning("No .shp files found in %s", shp_dir)
        return

    logger.info(
        "[%s] Batch converting %d shapefiles from %s …",
        lake_id, len(shp_files), shp_dir,
    )

    for shp in tqdm(shp_files, desc=f"Masks {lake_id}", unit="file"):
        date_str = _parse_date_from_stem(shp.stem)
        if not date_str:
            logger.warning(
                "Cannot extract date from filename '%s' — skipping. "
                "Filename should contain YYYY-MM-DD.", shp.stem,
            )
            continue

        reference_tif = DATA_INTERIM / lake_id / "stacked" / f"{date_str}.tif"
        try:
            mask = shapefile_to_mask(shp, reference_tif)
            save_mask(mask, lake_id, date_str, shp)
        except Exception as exc:
            logger.error("Error converting %s: %s", shp.name, exc)


def qa_report(lake_id: str) -> None:
    """Print a QA summary table for all masks in a lake."""
    processed_dir = DATA_PROCESSED / lake_id
    if not processed_dir.exists():
        logger.warning("No processed data for lake '%s'.", lake_id)
        return

    qa_files = sorted(processed_dir.rglob("mask_qa.json"))
    if not qa_files:
        logger.warning("No mask_qa.json files found for lake '%s'.", lake_id)
        return

    print(f"\n{'━'*70}")
    print(f"  QA Report — Lake: {lake_id}   ({len(qa_files)} masks)")
    print(f"{'━'*70}")
    print(f"  {'Date':<14} {'Lake px':>10} {'Fraction':>10} {'Warnings'}")
    print(f"  {'─'*66}")

    total_warnings = 0
    for qa_path in qa_files:
        qa = json.loads(qa_path.read_text())
        w = len(qa["qa_warnings"])
        total_warnings += w
        warn_str = f"⚠️  {w}" if w else "✓"
        print(
            f"  {qa['date']:<14} {qa['lake_pixels']:>10,d} "
            f"{qa['lake_fraction']:>9.1%}  {warn_str}"
        )

    print(f"{'━'*70}")
    print(f"  Total: {len(qa_files)} masks, {total_warnings} with warnings\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert hand-digitized QGIS shapefiles to binary lake masks "
            "aligned to the stacked tensor grid."
        )
    )
    parser.add_argument("--lake", required=True, help="Lake ID (e.g. 'south_lhonak').")
    parser.add_argument(
        "--date",
        default=None,
        help="Date YYYY-MM-DD for single-file conversion. Required with --shapefile.",
    )
    parser.add_argument(
        "--shapefile",
        default=None,
        help="Path to a single .shp file for conversion.",
    )
    parser.add_argument(
        "--shapefile-dir",
        default=None,
        help=(
            "Path to a directory containing multiple .shp files to batch-convert. "
            "Filenames must include YYYY-MM-DD."
        ),
    )
    parser.add_argument(
        "--qa-only",
        action="store_true",
        help="Print QA report for existing masks without converting anything new.",
    )
    parser.add_argument(
        "--help-qgis",
        action="store_true",
        help="Print step-by-step QGIS digitizing instructions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.help_qgis:
        print(QGIS_INSTRUCTIONS)
        return

    if args.qa_only:
        qa_report(args.lake)
        return

    lake = get_lake_by_id(args.lake)

    if args.shapefile_dir:
        batch_convert_dir(lake["id"], Path(args.shapefile_dir))

    elif args.shapefile:
        if not args.date:
            raise ValueError("--date YYYY-MM-DD is required when using --shapefile.")
        shp_path = Path(args.shapefile)
        reference_tif = DATA_INTERIM / lake["id"] / "stacked" / f"{args.date}.tif"
        mask = shapefile_to_mask(shp_path, reference_tif)
        save_mask(mask, lake["id"], args.date, shp_path)

    else:
        raise ValueError(
            "Specify one of: --shapefile <path>, --shapefile-dir <dir>, "
            "--qa-only, or --help-qgis."
        )


if __name__ == "__main__":
    main()
