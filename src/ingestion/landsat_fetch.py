"""
landsat_fetch.py — Download Landsat 5/8/9 imagery via Google Earth Engine.

Provides the long historical baseline (1990–2015) before Sentinel-1/2 existed.
Useful for reconstructing multi-decadal lake area trends.

Usage
-----
    # Download Landsat for South Lhonak, full historical baseline:
    python src/ingestion/landsat_fetch.py --lake south_lhonak

    # Custom date range:
    python src/ingestion/landsat_fetch.py --lake south_lhonak --start 2000-01-01 --end 2015-12-31

Output
------
    data/raw/{lake_id}/landsat/{mission}/{YYYY-MM-DD}.tif
    e.g. data/raw/south_lhonak/landsat/L8/2019-06-15.tif
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import ee
from tqdm import tqdm

from src.utils.auth import initialize_gee
from src.utils.config_loader import get_lake_by_id, get_active_lakes, load_pipeline_config
from src.ingestion.gee_fetch import lake_to_aoi, _download_image_to_file

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

DATA_RAW = PROJECT_ROOT / "data" / "raw"

# Landsat mission short labels for folder naming
MISSION_LABELS = {
    "LANDSAT/LT05/C02/T1_L2": "L5",
    "LANDSAT/LC08/C02/T1_L2": "L8",
    "LANDSAT/LC09/C02/T1_L2": "L9",
}


def _cloud_mask_landsat(image: ee.Image, qa_band: str = "QA_PIXEL") -> ee.Image:
    """
    Apply the Landsat Collection 2 cloud + cloud-shadow bitmask.
    Bits 3 (cloud shadow) and 4 (cloud) must both be 0 for a pixel to be clear.
    """
    qa = image.select(qa_band)
    cloud_shadow = qa.bitwiseAnd(1 << 3).eq(0)
    cloud = qa.bitwiseAnd(1 << 4).eq(0)
    return image.updateMask(cloud_shadow.And(cloud))


def fetch_landsat_collection(
    lake: dict,
    collection_cfg: dict,
    start_date: str,
    end_date: str,
    cloud_max_pct: int = 20,
    scale: int = 30,
) -> None:
    """
    Download cloud-masked Landsat scenes for one mission (L5/L8/L9).
    Outputs bands: [blue, green, red, nir, swir] renamed to standard names.
    """
    aoi = lake_to_aoi(lake)
    mission_label = MISSION_LABELS.get(collection_cfg["id"], "LS")
    out_dir = DATA_RAW / lake["id"] / "landsat" / mission_label
    out_dir.mkdir(parents=True, exist_ok=True)

    # Restrict to the sensor's operational period
    period_start = max(start_date, collection_cfg["period"][0])
    period_end = min(end_date, collection_cfg["period"][1])
    if period_start >= period_end:
        logger.debug(
            "%s: date range %s–%s outside sensor operational period — skipping.",
            mission_label, start_date, end_date,
        )
        return

    # Source band names for this mission
    src_bands = [
        collection_cfg["blue"],
        collection_cfg["green"],
        collection_cfg["red"],
        collection_cfg["nir"],
        collection_cfg["swir"],
    ]
    target_bands = ["blue", "green", "red", "nir", "swir"]

    collection = (
        ee.ImageCollection(collection_cfg["id"])
        .filterBounds(aoi)
        .filterDate(period_start, period_end)
        .filter(ee.Filter.lt("CLOUD_COVER", cloud_max_pct))
        .map(lambda img: _cloud_mask_landsat(img, collection_cfg["qa"]))
        .select(src_bands, target_bands)
    )

    image_list = collection.toList(collection.size()).getInfo()
    logger.info(
        "[%s] %s: Found %d images (%s → %s)",
        lake["id"], mission_label, len(image_list), period_start, period_end,
    )

    for img_info in tqdm(
        image_list, desc=f"Landsat {mission_label} {lake['id']}", unit="scene"
    ):
        img_id = img_info["id"]
        acq_date = img_info["properties"].get("system:time_start", 0)
        date_str = datetime.utcfromtimestamp(acq_date / 1000).strftime("%Y-%m-%d")

        out_path = out_dir / f"{date_str}.tif"
        meta_path = out_dir / f"{date_str}_meta.json"

        if out_path.exists():
            logger.debug("%s %s %s already downloaded — skipping.", mission_label, lake["id"], date_str)
            continue

        image = ee.Image(img_id).clip(aoi)
        _download_image_to_file(image, aoi, scale, target_bands, out_path)

        with open(meta_path, "w") as fh:
            json.dump(
                {
                    "image_id": img_id,
                    "date": date_str,
                    "lake_id": lake["id"],
                    "sensor": f"Landsat {mission_label}",
                    "collection": collection_cfg["id"],
                    "bands": target_bands,
                    "source_bands": src_bands,
                    "scale_m": scale,
                    "cloud_pct": img_info["properties"].get("CLOUD_COVER"),
                },
                fh,
                indent=2,
            )

    logger.info("[%s] %s download complete.", lake["id"], mission_label)


def fetch_all_landsat(
    lake: dict,
    cfg: dict,
    start_date: str,
    end_date: str,
) -> None:
    """Fetch all applicable Landsat missions (L5/L8/L9) for a lake and date range."""
    for coll_cfg in cfg["landsat"]["collections"]:
        fetch_landsat_collection(lake, coll_cfg, start_date, end_date)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Landsat 5/8/9 imagery via GEE for long-baseline lake monitoring."
    )
    parser.add_argument(
        "--lake",
        required=True,
        help="Lake ID (e.g. 'south_lhonak') or 'all' for all verified lakes.",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Start date YYYY-MM-DD (default: from pipeline.yaml historical start).",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="End date YYYY-MM-DD (default: from pipeline.yaml end_date).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("Initialising GEE …")
    initialize_gee()

    cfg = load_pipeline_config()
    start_date = args.start or cfg["data"]["landsat_start_date"]
    end_date = args.end or cfg["data"]["end_date"]

    if args.lake == "all":
        lakes = get_active_lakes(exclude_placeholder_coords=True)
    else:
        lakes = [get_lake_by_id(args.lake)]

    for lake in lakes:
        logger.info("━━━ Lake: %s (%s) ━━━", lake["name"], lake["id"])
        fetch_all_landsat(lake, cfg, start_date, end_date)

    logger.info("Landsat fetch complete.")


if __name__ == "__main__":
    main()
