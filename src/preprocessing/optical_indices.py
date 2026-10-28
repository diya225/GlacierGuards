"""
optical_indices.py — Compute spectral water/snow/vegetation indices from S2 and Landsat.

Index formulas (from the project spec / reference papers):
    NDWI_blue  = (B2 − B8)  / (B2 + B8)   [McFeeters, blue-NIR water index]
    NDWI_green = (B3 − B8)  / (B3 + B8)   [Gao, green-NIR water index]
    NDVI       = (B8 − B4)  / (B8 + B4)   [vegetation]
    NDSI       = (B3 − B11) / (B3 + B11)  [snow/ice; B11 = SWIR1 for S2]
    NDGI       = (B3 − B4)  / (B3 + B4)   [turbid/glacial water, green-red]

All indices are clipped to [−1, 1] and stored as float32.

Usage
-----
    python src/preprocessing/optical_indices.py --lake south_lhonak
    python src/preprocessing/optical_indices.py --lake all

Output
------
    data/interim/{lake_id}/s2/{YYYY-MM-DD}_indices.tif   5-band float32
    Band order: [NDWI_blue, NDWI_green, NDVI, NDSI, NDGI]
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

# Band positions in the downloaded S2 GeoTIFF (0-indexed)
# File bands: B2(blue), B3(green), B4(red), B8(NIR), B11(SWIR)
S2_BAND_IDX = {"blue": 0, "green": 1, "red": 2, "nir": 3, "swir": 4}

NODATA_OUT = -9999.0


# ─────────────────────────────────────────────────────────────────────────────
# Index computation
# ─────────────────────────────────────────────────────────────────────────────

def _safe_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute (a − b) / (a + b) safely, returning NaN where the denominator
    is zero or where either input is NaN.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            (a + b) == 0,
            np.nan,
            (a.astype(np.float32) - b.astype(np.float32))
            / (a.astype(np.float32) + b.astype(np.float32)),
        )
    return np.clip(ratio, -1.0, 1.0).astype(np.float32)


def compute_indices_from_bands(
    blue: np.ndarray,
    green: np.ndarray,
    red: np.ndarray,
    nir: np.ndarray,
    swir: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Compute all five spectral indices from individual band arrays.

    All inputs should be float32 reflectance values (S2 L2A is in [0, 10000]
    before scaling; pass raw values — the ratio normalises them automatically).

    Returns:
        Dict mapping index name → 2-D float32 array, clipped to [−1, 1].
    """
    return {
        "NDWI_blue": _safe_ratio(blue, nir),     # (B2 − B8) / (B2 + B8)
        "NDWI_green": _safe_ratio(green, nir),    # (B3 − B8) / (B3 + B8)
        "NDVI": _safe_ratio(nir, red),            # (B8 − B4) / (B8 + B4)
        "NDSI": _safe_ratio(green, swir),         # (B3 − B11) / (B3 + B11)
        "NDGI": _safe_ratio(green, red),          # (B3 − B4) / (B3 + B4)
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-file processing
# ─────────────────────────────────────────────────────────────────────────────

def process_s2_file(in_path: Path, out_path: Path) -> None:
    """
    Read a raw S2 GeoTIFF (bands: B2 B3 B4 B8 B11) and write a 5-band
    float32 GeoTIFF containing the computed spectral indices.
    """
    with rasterio.open(in_path) as src:
        profile = src.profile.copy()
        bands = src.read().astype(np.float32)  # shape: (5, H, W)
        nodata = src.nodata

    # Build nodata mask
    if nodata is not None:
        mask = np.any(bands == nodata, axis=0)  # (H, W)
    else:
        mask = np.zeros(bands.shape[1:], dtype=bool)

    blue  = np.where(mask, np.nan, bands[S2_BAND_IDX["blue"]])
    green = np.where(mask, np.nan, bands[S2_BAND_IDX["green"]])
    red   = np.where(mask, np.nan, bands[S2_BAND_IDX["red"]])
    nir   = np.where(mask, np.nan, bands[S2_BAND_IDX["nir"]])
    swir  = np.where(mask, np.nan, bands[S2_BAND_IDX["swir"]])

    indices = compute_indices_from_bands(blue, green, red, nir, swir)
    index_names = ["NDWI_blue", "NDWI_green", "NDVI", "NDSI", "NDGI"]

    out_profile = profile.copy()
    out_profile.update(
        count=5,
        dtype="float32",
        nodata=NODATA_OUT,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **out_profile) as dst:
        for band_idx, name in enumerate(index_names, start=1):
            arr = indices[name]
            arr_out = np.where(np.isfinite(arr), arr, NODATA_OUT)
            dst.write(arr_out, band_idx)
            dst.update_tags(band_idx, name=name)

    logger.debug("Indices written → %s", out_path)


def process_landsat_file(in_path: Path, out_path: Path) -> None:
    """
    Compute indices from a Landsat file (bands: blue green red nir swir).
    Band mapping is the same as S2 thanks to our standardised naming in landsat_fetch.py.
    """
    # Reuse S2 logic — same band order
    process_s2_file(in_path, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Lake-level batch processing
# ─────────────────────────────────────────────────────────────────────────────

def compute_lake_indices(lake: dict) -> None:
    """Compute spectral indices for all S2 and Landsat files for one lake."""
    # ── Sentinel-2 ───────────────────────────────────────────────────────────
    s2_raw_dir = DATA_RAW / lake["id"] / "s2"
    if s2_raw_dir.exists():
        s2_out_dir = DATA_INTERIM / lake["id"] / "s2"
        s2_tifs = sorted(s2_raw_dir.glob("*.tif"))
        logger.info("[%s] Computing indices for %d S2 images …", lake["id"], len(s2_tifs))
        for tif in tqdm(s2_tifs, desc=f"S2 indices {lake['id']}", unit="file"):
            date_str = tif.stem
            out_path = s2_out_dir / f"{date_str}_indices.tif"
            if out_path.exists():
                continue
            try:
                process_s2_file(tif, out_path)
            except Exception as exc:
                logger.error("S2 index error %s: %s — skipping.", tif.name, exc)
    else:
        logger.warning("[%s] No raw S2 directory found — run gee_fetch.py first.", lake["id"])

    # ── Landsat (all missions) ────────────────────────────────────────────────
    for mission in ["L5", "L8", "L9"]:
        ls_raw_dir = DATA_RAW / lake["id"] / "landsat" / mission
        if not ls_raw_dir.exists():
            continue
        ls_out_dir = DATA_INTERIM / lake["id"] / "landsat" / mission
        ls_tifs = sorted(ls_raw_dir.glob("*.tif"))
        logger.info(
            "[%s] Computing Landsat %s indices for %d images …",
            lake["id"], mission, len(ls_tifs),
        )
        for tif in tqdm(ls_tifs, desc=f"L{mission} indices {lake['id']}", unit="file"):
            date_str = tif.stem
            out_path = ls_out_dir / f"{date_str}_indices.tif"
            if out_path.exists():
                continue
            try:
                process_landsat_file(tif, out_path)
            except Exception as exc:
                logger.error("Landsat index error %s: %s — skipping.", tif.name, exc)

    logger.info("[%s] Spectral index computation complete.", lake["id"])


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute NDWI/NDWI_green/NDVI/NDSI/NDGI spectral indices from S2 and Landsat."
    )
    parser.add_argument(
        "--lake", required=True, help="Lake ID or 'all'."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.lake == "all":
        lakes = get_active_lakes(exclude_placeholder_coords=True)
    else:
        lakes = [get_lake_by_id(args.lake)]

    for lake in lakes:
        logger.info("━━━ Lake: %s (%s) ━━━", lake["name"], lake["id"])
        compute_lake_indices(lake)

    logger.info("Optical index computation done.")


if __name__ == "__main__":
    main()
