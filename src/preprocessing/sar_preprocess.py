"""
sar_preprocess.py — SAR speckle filtering and normalisation.

Applies Enhanced Lee speckle filter (7×7 window) to Sentinel-1 GRD VV/VH bands
and normalises to 8-bit [0, 255] via CDF histogram equalization.
Preserves a nodata mask so downstream steps can ignore invalid pixels.

Usage
-----
    # Preprocess all SAR files for South Lhonak:
    python src/preprocessing/sar_preprocess.py --lake south_lhonak

    # Preprocess all verified lakes:
    python src/preprocessing/sar_preprocess.py --lake all

Output
------
    data/interim/{lake_id}/s1/{YYYY-MM-DD}_filtered.tif   (float32, dB-space)
    data/interim/{lake_id}/s1/{YYYY-MM-DD}_norm8.tif      (uint8, 0–255)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from scipy.ndimage import uniform_filter, variance as ndimage_variance
from tqdm import tqdm

from src.utils.config_loader import get_lake_by_id, get_active_lakes, load_pipeline_config

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"

NODATA_VALUE = -9999.0


# ─────────────────────────────────────────────────────────────────────────────
# Enhanced Lee speckle filter
# ─────────────────────────────────────────────────────────────────────────────

def enhanced_lee_filter(band: np.ndarray, window: int = 7, input_is_db: bool = False) -> np.ndarray:
    """
    Apply the Enhanced Lee adaptive speckle filter to a single SAR band.

    The filter adapts between a mean filter in homogeneous regions and an
    identity (no filtering) in point targets, based on the local coefficient
    of variation. Works on linear-scale (not dB) backscatter.

    Args:
        band:   2-D float32 array of linear-scale SAR backscatter.
        window: Filter window size (default 7 → 7×7 kernel).

    Returns:
        Filtered 2-D float32 array with the same shape as `band`.
    """
    # GEE COPERNICUS/S1_GRD stores backscatter in dB. Convert to linear power
    # before the Lee filter; negative dB values are valid water/land returns.
    valid = np.isfinite(band) & (band != NODATA_VALUE)
    linear = np.full(band.shape, np.nan, dtype=np.float64)
    if input_is_db:
        linear[valid] = np.power(10.0, band[valid].astype(np.float64) / 10.0)
    else:
        valid &= band > 0
        linear[valid] = band[valid]
    band = linear

    local_mean = uniform_filter(np.nan_to_num(band), size=window)
    local_sq_mean = uniform_filter(np.nan_to_num(band) ** 2, size=window)
    local_var = local_sq_mean - local_mean ** 2
    local_var = np.maximum(local_var, 1e-10)  # avoid division by zero

    # Equivalent number of looks (ENL) — assumed 4.9 for IW GRD
    enl = 4.9
    noise_var = local_mean ** 2 / enl

    # Weighting factor W: 0 in homogeneous regions → 1 at point targets
    weights = 1.0 - (noise_var / local_var)
    weights = np.clip(weights, 0.0, 1.0)

    filtered = local_mean + weights * (np.nan_to_num(band) - local_mean)
    # Re-apply nan mask
    filtered = np.where(np.isnan(band), np.nan, filtered)
    return filtered.astype(np.float32)


def linear_to_db(band: np.ndarray) -> np.ndarray:
    """Convert linear SAR backscatter to dB: 10 * log10(x)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10.0 * np.log10(np.where(band > 0, band, np.nan))
    return db.astype(np.float32)


def normalise_to_8bit(band: np.ndarray, db_min: float = -30.0, db_max: float = 0.0) -> np.ndarray:
    """
    Normalise a float band to uint8 [0, 255] using percentile stretch (CDF mapping).
    Pixels that are NaN are set to 0 (nodata in uint8).

    Args:
        band:   2-D float array.
        p_low:  Lower percentile to clip at (default 2%).
        p_high: Upper percentile to clip at (default 98%).

    Returns:
        uint8 array, same shape as `band`.
    """
    valid = np.isfinite(band) & (band != NODATA_VALUE)
    clipped = np.clip(band, db_min, db_max)
    normed = (clipped - db_min) / (db_max - db_min) * 255.0
    normed = np.where(valid, normed, 0.0)
    return normed.astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Per-file processing
# ─────────────────────────────────────────────────────────────────────────────

def process_s1_file(
    in_path: Path,
    out_filtered_path: Path,
    out_norm_path: Path,
    window: int = 7,
) -> None:
    """
    Read a raw S1 GeoTIFF (2 bands: VV, VH), apply Enhanced Lee filter,
    convert to dB, and save two outputs:
        1. Float32 dB-space filtered raster.
        2. Uint8 normalised raster (for visualisation and stacking).
    """
    with rasterio.open(in_path) as src:
        profile = src.profile.copy()
        vv = src.read(1).astype(np.float32)
        vh = src.read(2).astype(np.float32)
        nodata = src.nodata

    # Mark nodata pixels as NaN
    if nodata is not None:
        vv = np.where(vv == nodata, np.nan, vv)
        vh = np.where(vh == nodata, np.nan, vh)

    # Filter in linear space
    # Earth Engine's S1_GRD VV/VH bands are already in dB.
    vv_filt = enhanced_lee_filter(vv, window, input_is_db=True)
    vh_filt = enhanced_lee_filter(vh, window, input_is_db=True)

    # Convert to dB
    vv_db = linear_to_db(vv_filt)
    vh_db = linear_to_db(vh_filt)

    # ── Save filtered dB raster ───────────────────────────────────────────
    filt_profile = profile.copy()
    filt_profile.update(dtype="float32", count=2, nodata=np.nan)
    out_filtered_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_filtered_path, "w", **filt_profile) as dst:
        dst.write(vv_db, 1)
        dst.write(vh_db, 2)
        dst.update_tags(1, name="VV_dB_filtered")
        dst.update_tags(2, name="VH_dB_filtered")

    # ── Save 8-bit normalised raster ──────────────────────────────────────
    vv_u8 = normalise_to_8bit(vv_db)
    vh_u8 = normalise_to_8bit(vh_db)

    norm_profile = profile.copy()
    norm_profile.update(dtype="uint8", count=2, nodata=0)
    out_norm_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_norm_path, "w", **norm_profile) as dst:
        dst.write(vv_u8, 1)
        dst.write(vh_u8, 2)
        dst.update_tags(1, name="VV_norm8")
        dst.update_tags(2, name="VH_norm8")


# ─────────────────────────────────────────────────────────────────────────────
# Lake-level batch processing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_lake_sar(lake: dict, cfg: dict) -> None:
    """Process all raw S1 GeoTIFFs for one lake."""
    window = cfg["preprocessing"]["sar"]["filter_size"]

    raw_dir = DATA_RAW / lake["id"] / "s1"
    if not raw_dir.exists():
        logger.warning(
            "[%s] Raw S1 directory not found: %s — run gee_fetch.py first.",
            lake["id"], raw_dir,
        )
        return

    tif_files = sorted(raw_dir.glob("*.tif"))
    if not tif_files:
        logger.warning("[%s] No .tif files found in %s.", lake["id"], raw_dir)
        return

    out_dir = DATA_INTERIM / lake["id"] / "s1"

    logger.info(
        "[%s] Processing %d SAR files (Enhanced Lee %dx%d) …",
        lake["id"], len(tif_files), window, window,
    )

    for tif_path in tqdm(tif_files, desc=f"SAR {lake['id']}", unit="file"):
        date_str = tif_path.stem  # includes orbit/relative orbit for S1 scenes
        out_filtered = out_dir / f"{date_str}_filtered.tif"
        out_norm = out_dir / f"{date_str}_norm8.tif"

        if out_filtered.exists() and out_norm.exists():
            logger.debug("SAR %s %s already processed — skipping.", lake["id"], date_str)
            continue

        try:
            process_s1_file(tif_path, out_filtered, out_norm, window=window)
        except Exception as exc:
            logger.error(
                "Error processing %s: %s — skipping this file.", tif_path.name, exc
            )

    logger.info("[%s] SAR preprocessing complete.", lake["id"])


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply Enhanced Lee speckle filter and normalise Sentinel-1 SAR rasters."
    )
    parser.add_argument(
        "--lake",
        required=True,
        help="Lake ID (e.g. 'south_lhonak') or 'all'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_pipeline_config()

    if args.lake == "all":
        lakes = get_active_lakes(exclude_placeholder_coords=True)
    else:
        lakes = [get_lake_by_id(args.lake)]

    for lake in lakes:
        logger.info("━━━ Lake: %s (%s) ━━━", lake["name"], lake["id"])
        preprocess_lake_sar(lake, cfg)

    logger.info("SAR preprocessing complete for all lakes.")


if __name__ == "__main__":
    main()
