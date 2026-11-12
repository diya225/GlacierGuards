"""
gee_fetch.py — Download Sentinel-1, Sentinel-2, and Copernicus DEM via Google Earth Engine.

Usage
-----
    # Download all available data for South Lhonak, 2020–2023:
    python src/ingestion/gee_fetch.py --lake south_lhonak --start 2020-01-01 --end 2023-12-31

    # Download S1 only for all verified lakes:
    python src/ingestion/gee_fetch.py --lake all --sensor s1 --start 2015-01-01 --end 2024-12-31

    # Download the DEM once for a specific lake:
    python src/ingestion/gee_fetch.py --lake south_lhonak --sensor dem

Output
------
    data/raw/{lake_id}/s1/{YYYY-MM-DD}.tif          Sentinel-1 VV+VH
    data/raw/{lake_id}/s2/{YYYY-MM-DD}.tif          Sentinel-2 B2,B3,B4,B8,B11
    data/raw/{lake_id}/dem/dem.tif                   Copernicus GLO-30 DEM
    data/raw/{lake_id}/s1/{YYYY-MM-DD}_meta.json    Per-image metadata
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import ee
import requests
from tqdm import tqdm

# Add project root to path so sibling modules are importable
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.auth import initialize_gee
from src.utils.config_loader import get_lake_by_id, get_active_lakes, load_pipeline_config

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

DATA_RAW = PROJECT_ROOT / "data" / "raw"


# ─────────────────────────────────────────────────────────────────────────────
# GEE geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def lake_to_aoi(lake: dict, buffer_km: float | None = None) -> ee.Geometry:
    """Create a square AOI around the lake centroid with a given buffer."""
    km = buffer_km or lake.get("buffer_km", 2.0)
    point = ee.Geometry.Point([lake["lon"], lake["lat"]])
    return point.buffer(km * 1000).bounds()  # square bbox in metres


# ─────────────────────────────────────────────────────────────────────────────
# Sentinel-1 fetch
# ─────────────────────────────────────────────────────────────────────────────

def _s1_collection(aoi: ee.Geometry, start: str, end: str, cfg: dict) -> ee.ImageCollection:
    s1_cfg = cfg["sentinel1"]
    return (
        ee.ImageCollection(s1_cfg["collection"])
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("instrumentMode", s1_cfg["beam_mode"]))
        .select(["VV", "VH"])
    )


def fetch_sentinel1(
    lake: dict,
    start_date: str,
    end_date: str,
    cfg: dict,
    scale: int = 10,
) -> None:
    """Download all Sentinel-1 GRD passes for a lake and date range."""
    aoi = lake_to_aoi(lake)
    out_dir = DATA_RAW / lake["id"] / "s1"
    out_dir.mkdir(parents=True, exist_ok=True)

    collection = _s1_collection(aoi, start_date, end_date, cfg)
    image_list = collection.toList(collection.size()).getInfo()

    logger.info(
        "[%s] Found %d Sentinel-1 images (%s → %s)",
        lake["id"], len(image_list), start_date, end_date,
    )

    for img_info in tqdm(image_list, desc=f"S1 {lake['id']}", unit="scene"):
        img_id = img_info["id"]
        # Use the acquisition date as the filename key
        acq_date = img_info["properties"].get(
            "system:time_start", 0
        )
        date_str = datetime.utcfromtimestamp(acq_date / 1000).strftime("%Y-%m-%d")

        out_path = out_dir / f"{date_str}.tif"
        meta_path = out_dir / f"{date_str}_meta.json"

        if out_path.exists():
            logger.debug("S1 %s %s already downloaded — skipping.", lake["id"], date_str)
            continue

        image = ee.Image(img_id).clip(aoi)
        _download_image_to_file(image, aoi, scale, ["VV", "VH"], out_path)

        # Save metadata alongside
        with open(meta_path, "w") as fh:
            json.dump(
                {
                    "image_id": img_id,
                    "date": date_str,
                    "lake_id": lake["id"],
                    "sensor": "Sentinel-1 GRD",
                    "bands": ["VV", "VH"],
                    "scale_m": scale,
                    "properties": img_info["properties"],
                },
                fh,
                indent=2,
            )

    logger.info("[%s] Sentinel-1 download complete.", lake["id"])


# ─────────────────────────────────────────────────────────────────────────────
# Sentinel-2 fetch
# ─────────────────────────────────────────────────────────────────────────────

def _s2_collection(aoi: ee.Geometry, start: str, end: str, cfg: dict) -> ee.ImageCollection:
    s2_cfg = cfg["sentinel2"]
    data_cfg = cfg["data"]
    return (
        ee.ImageCollection(s2_cfg["collection"])
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(
            ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", data_cfg["cloud_cover_max_pct"])
        )
        .select(s2_cfg["bands"])
    )


def fetch_sentinel2(
    lake: dict,
    start_date: str,
    end_date: str,
    cfg: dict,
    scale: int = 10,
) -> None:
    """Download cloud-free Sentinel-2 L2A images for a lake and date range."""
    aoi = lake_to_aoi(lake)
    out_dir = DATA_RAW / lake["id"] / "s2"
    out_dir.mkdir(parents=True, exist_ok=True)

    s2_cfg = cfg["sentinel2"]
    collection = _s2_collection(aoi, start_date, end_date, cfg)
    image_list = collection.toList(collection.size()).getInfo()

    logger.info(
        "[%s] Found %d Sentinel-2 images (cloud<%.0f%%, %s→%s)",
        lake["id"], len(image_list),
        cfg["data"]["cloud_cover_max_pct"],
        start_date, end_date,
    )

    for img_info in tqdm(image_list, desc=f"S2 {lake['id']}", unit="scene"):
        img_id = img_info["id"]
        acq_date = img_info["properties"].get("system:time_start", 0)
        date_str = datetime.utcfromtimestamp(acq_date / 1000).strftime("%Y-%m-%d")

        out_path = out_dir / f"{date_str}.tif"
        meta_path = out_dir / f"{date_str}_meta.json"

        if out_path.exists():
            logger.debug("S2 %s %s already downloaded — skipping.", lake["id"], date_str)
            continue

        image = ee.Image(img_id).clip(aoi)
        _download_image_to_file(image, aoi, scale, s2_cfg["bands"], out_path)

        with open(meta_path, "w") as fh:
            json.dump(
                {
                    "image_id": img_id,
                    "date": date_str,
                    "lake_id": lake["id"],
                    "sensor": "Sentinel-2 L2A",
                    "bands": s2_cfg["bands"],
                    "scale_m": scale,
                    "cloud_pct": img_info["properties"].get("CLOUDY_PIXEL_PERCENTAGE"),
                    "properties": img_info["properties"],
                },
                fh,
                indent=2,
            )

    logger.info("[%s] Sentinel-2 download complete.", lake["id"])


# ─────────────────────────────────────────────────────────────────────────────
# DEM fetch (one-time static layer)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_dem(lake: dict, cfg: dict, scale: int = 30) -> None:
    """Download Copernicus GLO-30 DEM for a lake AOI (run once, static)."""
    aoi = lake_to_aoi(lake)
    out_dir = DATA_RAW / lake["id"] / "dem"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "dem.tif"
    if out_path.exists():
        logger.info("[%s] DEM already downloaded — skipping.", lake["id"])
        return

    dem_cfg = cfg["dem"]
    dem_image = (
        ee.ImageCollection(dem_cfg["collection"])
        .filterBounds(aoi)
        .mosaic()
        .select(dem_cfg["band"])
        .clip(aoi)
    )

    logger.info("[%s] Downloading Copernicus GLO-30 DEM (30 m) …", lake["id"])
    _download_image_to_file(dem_image, aoi, scale, [dem_cfg["band"]], out_path)
    logger.info("[%s] DEM saved → %s", lake["id"], out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Download helper
# ─────────────────────────────────────────────────────────────────────────────

def _download_image_to_file(
    image: ee.Image,
    aoi: ee.Geometry,
    scale: int,
    bands: list[str],
    out_path: Path,
    max_retries: int = 3,
) -> None:
    """
    Generate a GEE download URL and stream the GeoTIFF to out_path.
    Retries on network errors with exponential back-off.
    Raises RuntimeError if all retries are exhausted.
    """
    for attempt in range(1, max_retries + 1):
        try:
            url = image.getDownloadURL(
                {
                    "scale": scale,
                    "region": aoi,
                    "format": "GEO_TIFF",
                    "bands": bands,
                    "filePerBand": False,  # all bands in one file
                }
            )
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()

            tmp_path = out_path.with_suffix(".tmp")
            with open(tmp_path, "wb") as fh:
                for chunk in response.iter_content(chunk_size=1024 * 64):
                    fh.write(chunk)
            tmp_path.rename(out_path)  # atomic rename on success
            return

        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else 0
            if status in (429, 500, 502, 503, 504): # i added a few more error codes
                wait = 2**attempt * 30  # back-off: 60s, 120s, 240s
                logger.warning(
                    "GEE API error %s. Waiting %ds before retry %d/%d …",
                    status, wait, attempt, max_retries,
                )
                time.sleep(wait)
            else:
                raise
        except (requests.ConnectionError, requests.Timeout) as exc:
            wait = 2**attempt * 10
            logger.warning(
                "Network error (%s). Retrying in %ds (%d/%d) …",
                exc, wait, attempt, max_retries,
            )
            time.sleep(wait)

    raise RuntimeError(
        f"Failed to download {out_path.name} after {max_retries} attempts."
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch satellite data from Google Earth Engine for GLOF lake monitoring."
    )
    parser.add_argument(
        "--lake",
        required=True,
        help=(
            "Lake ID to process (e.g. 'south_lhonak'), "
            "or 'all' to process all verified lakes in lakes.yaml."
        ),
    )
    parser.add_argument(
        "--sensor",
        choices=["s1", "s2", "dem", "all"],
        default="all",
        help="Which sensor to download (default: all).",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Start date YYYY-MM-DD (default: from pipeline.yaml).",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="End date YYYY-MM-DD (default: from pipeline.yaml).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Initialise GEE ───────────────────────────────────────────────────────
    logger.info("Initialising Google Earth Engine …")
    initialize_gee()

    # ── Load configs ─────────────────────────────────────────────────────────
    cfg = load_pipeline_config()
    start_date = args.start or cfg["data"]["start_date"]
    end_date = args.end or cfg["data"]["end_date"]
    scale = cfg["data"]["resolution_m"]

    # ── Resolve lake list ────────────────────────────────────────────────────
    if args.lake == "all":
        lakes = get_active_lakes(exclude_placeholder_coords=True)
        logger.info("Processing %d verified lakes.", len(lakes))
    else:
        lakes = [get_lake_by_id(args.lake)]

    # ── Download ─────────────────────────────────────────────────────────────
    for lake in lakes:
        logger.info("━━━ Lake: %s (%s) ━━━", lake["name"], lake["id"])

        if args.sensor in ("s1", "all"):
            fetch_sentinel1(lake, start_date, end_date, cfg, scale)

        if args.sensor in ("s2", "all"):
            fetch_sentinel2(lake, start_date, end_date, cfg, scale)

        if args.sensor in ("dem", "all"):
            fetch_dem(lake, cfg, scale=30)  # DEM always 30 m

    logger.info("All downloads finished.")


if __name__ == "__main__":
    main()
