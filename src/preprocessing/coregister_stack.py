"""
coregister_stack.py — Co-register and stack all layers into 10-channel tensors.

For each date where both SAR (filtered) and S2 indices exist, this script:
  1. Reprojects all layers to a common 10 m UTM grid.
  2. Computes slope and aspect from the DEM.
  3. Stacks all 10 channels in the canonical order:
         [VV, VH, NDWI_blue, NDWI_green, NDVI, NDSI, NDGI, elevation, slope, aspect]
  4. Writes the result to data/interim/{lake_id}/stacked/{date}.tif

Usage
-----
    python src/preprocessing/coregister_stack.py --lake south_lhonak
    python src/preprocessing/coregister_stack.py --lake all

Output
------
    data/interim/{lake_id}/stacked/{YYYY-MM-DD}.tif   10-band float32 GeoTIFF
    data/interim/{lake_id}/stacked/{YYYY-MM-DD}_meta.json  channel metadata
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from rasterio.crs import CRS
from scipy.ndimage import sobel
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

# Canonical channel order (10 channels total)
CHANNEL_NAMES = [
    "VV",          # 0 — SAR VV (filtered, dB-space)
    "VH",          # 1 — SAR VH (filtered, dB-space)
    "NDWI_blue",   # 2 — (B2-B8)/(B2+B8)
    "NDWI_green",  # 3 — (B3-B8)/(B3+B8)
    "NDVI",        # 4 — (B8-B4)/(B8+B4)
    "NDSI",        # 5 — (B3-B11)/(B3+B11)
    "NDGI",        # 6 — (B3-B4)/(B3+B4)
    "elevation",   # 7 — Copernicus GLO-30 DEM, metres
    "slope",       # 8 — degrees, computed from DEM
    "aspect",      # 9 — degrees 0–360, north=0
]


# ─────────────────────────────────────────────────────────────────────────────
# DEM-derived terrain layers
# ─────────────────────────────────────────────────────────────────────────────

def compute_slope_aspect(dem: np.ndarray, cell_size_m: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute slope (degrees) and aspect (degrees, north=0 clockwise) from a DEM.

    Args:
        dem:         2-D float32 elevation array in metres.
        cell_size_m: Grid cell size in metres (default 10 m after resampling).

    Returns:
        (slope_deg, aspect_deg) — both float32 arrays, same shape as `dem`.
    """
    dy = sobel(dem, axis=0) / (8.0 * cell_size_m)
    dx = sobel(dem, axis=1) / (8.0 * cell_size_m)

    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.degrees(slope_rad).astype(np.float32)

    # Aspect: north=0, clockwise
    aspect_rad = np.arctan2(-dy, dx)
    aspect_deg = (90.0 - np.degrees(aspect_rad)) % 360.0
    aspect_deg = aspect_deg.astype(np.float32)

    return slope_deg, aspect_deg


# ─────────────────────────────────────────────────────────────────────────────
# Resampling / reprojection
# ─────────────────────────────────────────────────────────────────────────────

def reproject_to_grid(
    src_path: Path,
    dst_transform,
    dst_crs: CRS,
    dst_shape: tuple[int, int],
    band_indices: list[int] | None = None,
    resampling: Resampling = Resampling.bilinear,
) -> np.ndarray:
    """
    Reproject a raster to a target grid.

    Args:
        src_path:     Source GeoTIFF path.
        dst_transform: Target Affine transform.
        dst_crs:       Target CRS.
        dst_shape:     (height, width) of the target grid.
        band_indices:  Which bands to read (1-indexed). If None, read all.
        resampling:    Rasterio resampling method.

    Returns:
        Float32 array of shape (n_bands, height, width).
    """
    with rasterio.open(src_path) as src:
        all_bands = list(range(1, src.count + 1))
        selected = band_indices if band_indices else all_bands
        output = np.full(
            (len(selected), *dst_shape), NODATA_VALUE, dtype=np.float32
        )
        for out_idx, src_band in enumerate(selected):
            reproject(
                source=rasterio.band(src, src_band),
                destination=output[out_idx],
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=resampling,
                dst_nodata=NODATA_VALUE,
            )
    return output


def _get_utm_crs_for_point(lat: float, lon: float) -> CRS:
    """Return the best UTM CRS for a given lat/lon point."""
    zone = int((lon + 180) / 6) + 1
    hemisphere = "north" if lat >= 0 else "south"
    epsg = 32600 + zone if hemisphere == "north" else 32700 + zone
    return CRS.from_epsg(epsg)


# ─────────────────────────────────────────────────────────────────────────────
# Reference grid construction
# ─────────────────────────────────────────────────────────────────────────────

def build_reference_grid(
    dem_path: Path,
    lake: dict,
    target_resolution_m: int = 10,
) -> tuple:
    """
    Compute a UTM reference grid from the DEM that all other layers will be
    resampled to.

    Returns:
        (dst_crs, dst_transform, dst_height, dst_width)
    """
    dst_crs = _get_utm_crs_for_point(lake["lat"], lake["lon"])
    with rasterio.open(dem_path) as src:
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src.crs,
            dst_crs,
            src.width,
            src.height,
            *src.bounds,
            resolution=target_resolution_m,
        )
    return dst_crs, dst_transform, dst_height, dst_width


# ─────────────────────────────────────────────────────────────────────────────
# Date matching (find S2 nearest to S1)
# ─────────────────────────────────────────────────────────────────────────────

def _find_nearest_s1(s2_date_str: str, s1_dates: list[str], max_gap_days: int = 5) -> str | None:
    """
    Find the nearest S1 date to a given S2 date within max_gap_days.
    Returns None if no suitable S1 date exists.
    """
    from datetime import datetime
    s2_dt = datetime.strptime(s2_date_str, "%Y-%m-%d")
    best = None
    best_gap = float("inf")
    for s1_str in s1_dates:
        s1_dt = datetime.strptime(s1_str, "%Y-%m-%d")
        gap = abs((s2_dt - s1_dt).days)
        if gap <= max_gap_days and gap < best_gap:
            best_gap = gap
            best = s1_str
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Main stacking function
# ─────────────────────────────────────────────────────────────────────────────

def stack_lake(lake: dict, cfg: dict) -> None:
    """
    Build 10-channel stacked tensors for all available paired S1+S2 dates.
    """
    resolution = cfg["data"]["resolution_m"]
    max_gap = cfg["data"].get("s1_max_temporal_gap_days", 5)

    # ── Locate source files ───────────────────────────────────────────────────
    dem_path = DATA_RAW / lake["id"] / "dem" / "dem.tif"
    if not dem_path.exists():
        logger.warning(
            "[%s] DEM not found at %s — run gee_fetch.py --sensor dem first.",
            lake["id"], dem_path,
        )
        return

    s1_interim_dir = DATA_INTERIM / lake["id"] / "s1"
    s2_interim_dir = DATA_INTERIM / lake["id"] / "s2"

    if not s1_interim_dir.exists() or not s2_interim_dir.exists():
        logger.warning(
            "[%s] Interim SAR or S2 directory missing. "
            "Run sar_preprocess.py and optical_indices.py first.",
            lake["id"],
        )
        return

    s1_dates = [f.stem.replace("_filtered", "") for f in sorted(s1_interim_dir.glob("*_filtered.tif"))]
    s2_dates = [f.stem.replace("_indices", "") for f in sorted(s2_interim_dir.glob("*_indices.tif"))]

    if not s1_dates or not s2_dates:
        logger.warning("[%s] No processed S1 or S2 files found.", lake["id"])
        return

    # ── Build the reference UTM grid (once per lake) ─────────────────────────
    dst_crs, dst_transform, dst_h, dst_w = build_reference_grid(
        dem_path, lake, target_resolution_m=resolution
    )
    logger.info(
        "[%s] Reference grid: %dx%d px @ %dm (UTM zone from %.3f°, %.3f°)",
        lake["id"], dst_h, dst_w, resolution, lake["lat"], lake["lon"],
    )

    # ── Precompute DEM terrain layers (static per lake) ───────────────────────
    dem_reprojected = reproject_to_grid(
        dem_path, dst_transform, dst_crs, (dst_h, dst_w),
        band_indices=[1],
    )[0]
    slope_deg, aspect_deg = compute_slope_aspect(dem_reprojected, cell_size_m=resolution)

    # ── Loop over S2 dates, find matching S1, build stack ────────────────────
    out_dir = DATA_INTERIM / lake["id"] / "stacked"
    out_dir.mkdir(parents=True, exist_ok=True)

    paired = []
    for s2_date in s2_dates:
        s1_date = _find_nearest_s1(s2_date, s1_dates, max_gap_days=max_gap)
        if s1_date:
            paired.append((s2_date, s1_date))

    logger.info(
        "[%s] Found %d S2 dates, %d paired with S1 (max gap %d days).",
        lake["id"], len(s2_dates), len(paired), max_gap,
    )

    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 10,
        "height": dst_h,
        "width": dst_w,
        "crs": dst_crs,
        "transform": dst_transform,
        "nodata": NODATA_VALUE,
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }

    for s2_date, s1_date in tqdm(paired, desc=f"Stack {lake['id']}", unit="date"):
        out_path = out_dir / f"{s2_date}.tif"
        meta_path = out_dir / f"{s2_date}_meta.json"

        if out_path.exists():
            logger.debug("Stack %s %s already exists — skipping.", lake["id"], s2_date)
            continue

        try:
            # SAR channels (VV=band1, VH=band2 from filtered file)
            s1_path = s1_interim_dir / f"{s1_date}_filtered.tif"
            sar_bands = reproject_to_grid(
                s1_path, dst_transform, dst_crs, (dst_h, dst_w),
                band_indices=[1, 2],
            )  # (2, H, W)

            # S2 index channels (5 bands)
            s2_path = s2_interim_dir / f"{s2_date}_indices.tif"
            s2_bands = reproject_to_grid(
                s2_path, dst_transform, dst_crs, (dst_h, dst_w),
                band_indices=[1, 2, 3, 4, 5],
            )  # (5, H, W)

            # Stack into 10-channel tensor
            stack = np.stack(
                [
                    sar_bands[0],        # VV
                    sar_bands[1],        # VH
                    s2_bands[0],         # NDWI_blue
                    s2_bands[1],         # NDWI_green
                    s2_bands[2],         # NDVI
                    s2_bands[3],         # NDSI
                    s2_bands[4],         # NDGI
                    dem_reprojected,     # elevation
                    slope_deg,           # slope
                    aspect_deg,          # aspect
                ],
                axis=0,
            )  # (10, H, W)

            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(stack)
                for i, name in enumerate(CHANNEL_NAMES, start=1):
                    dst.update_tags(i, name=name)

            with open(meta_path, "w") as fh:
                json.dump(
                    {
                        "date": s2_date,
                        "s1_date": s1_date,
                        "s1_gap_days": abs(
                            (
                                __import__("datetime").datetime.strptime(s2_date, "%Y-%m-%d")
                                - __import__("datetime").datetime.strptime(s1_date, "%Y-%m-%d")
                            ).days
                        ),
                        "lake_id": lake["id"],
                        "channels": CHANNEL_NAMES,
                        "shape": [10, dst_h, dst_w],
                        "resolution_m": resolution,
                        "crs": str(dst_crs),
                    },
                    fh,
                    indent=2,
                )

        except Exception as exc:
            logger.error(
                "Stack error %s %s: %s — skipping.", lake["id"], s2_date, exc
            )

    logger.info("[%s] Stacking complete → %s", lake["id"], out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Co-register and stack SAR + optical + DEM into 10-channel tensors "
            "aligned to a common UTM grid."
        )
    )
    parser.add_argument("--lake", required=True, help="Lake ID or 'all'.")
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
        stack_lake(lake, cfg)

    logger.info("Co-registration and stacking complete.")


if __name__ == "__main__":
    main()
