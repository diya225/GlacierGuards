"""Download 2022 composites for the independent test lakes (Third Pole inventory) via GEE.

Thin wrapper around gloid_fetch.py (same retries, resumable files, manifest, DEM fix, S2/S1 windows).
Differences: lake list = configs/independent_test_lakes.csv; AOI = inventory 2022 polygon of the lake
(field ID_L == tp_id) + 1500 m buffer on the 10 m grid; folder data/raw/TP_<ID_L>/.

    python src/ingestion/tp_fetch.py --limit-lakes 1          # smoke test
    python src/ingestion/tp_fetch.py --workers 4              # all primary lakes
    python src/ingestion/tp_fetch.py --lakes TP_3630 TP_3652  # chosen lakes (e.g. backups)
Rerun until data/raw/_gloid_manifest.csv has no 'failed' rows for these lakes (existing files are skipped).
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import ee
import geopandas as gpd
import pandas as pd

from src.ingestion.gloid_fetch import BUF_M, MAX_BYTES, RAW, RES, download, fetch_lake_year, log_manifest
from src.utils.auth import initialize_gee

log = logging.getLogger("tp_fetch")
TP = ROOT / "data" / "external" / "third_pole_inventory" / "Third Pole" / "Third_Pole_Glacial Lake_2022.shp"


def aoi_grid(poly_4326, epsg):
    """Same tuple as gloid_fetch.aoi_grid: (crs, x0, y_top, W, H), polygon bounds + BUF_M, snapped to 10 m."""
    crs = f"EPSG:{int(epsg)}"
    minx, miny, maxx, maxy = gpd.GeoSeries([poly_4326], crs=4326).to_crs(crs).total_bounds
    x0 = math.floor((minx - BUF_M) / RES) * RES
    y1 = math.ceil((maxy + BUF_M) / RES) * RES
    W = int(math.ceil((maxx + BUF_M - x0) / RES))
    H = int(math.ceil((y1 - (miny - BUF_M)) / RES))
    return crs, x0, y1, W, H


def fetch_lake(row, tp):
    gid = row.lake_id
    try:
        hit = tp[tp.ID_L == int(row.tp_id)]
        if hit.empty:
            log_manifest(lake=gid, year="", product="all", status="failed", detail=f"tp_id {row.tp_id} not in inventory")
            return
        grid = aoi_grid(hit.geometry.iloc[0], row.utm_epsg)
        crs, x0, y1, W, H = grid
        if W * H * 10 > MAX_BYTES or W * H * 8 > MAX_BYTES:
            log_manifest(lake=gid, year="", product="all", status="skipped_aoi_too_large", detail=f"{W}x{H}")
            return
        aoi = ee.Geometry.Rectangle([x0, y1 - H * RES, x0 + W * RES, y1], crs, geodesic=False)
        dem = RAW / gid / "dem.tif"
        if not dem.exists():
            try:
                col = ee.ImageCollection("COPERNICUS/DEM/GLO30_2024_1").filterBounds(aoi).select("DEM")
                # mosaic() drops the native projection (-> constant output); restore it before resampling
                img = col.mosaic().setDefaultProjection(col.first().projection()).resample("bilinear").toFloat()
                download(img, grid, dem)
                log_manifest(lake=gid, year="", product="dem", status="ok")
            except Exception as ex:
                log_manifest(lake=gid, year="", product="dem", status="failed", detail=str(ex)[:200])
        for y in sorted(set(map(int, str(row.years).split()))):
            fetch_lake_year(gid, y, grid, aoi)
        log.info("lake %s done", gid)
    except Exception as ex:
        log_manifest(lake=gid, year="", product="all", status="failed", detail=str(ex)[:200])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "configs" / "independent_test_lakes.csv"))
    ap.add_argument("--roles", nargs="+", default=["primary"], help="rows to fetch (primary, backup)")
    ap.add_argument("--lakes", nargs="+", help="explicit lake_ids (overrides --roles)")
    ap.add_argument("--limit-lakes", type=int)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    initialize_gee()
    lakes = pd.read_csv(a.csv)
    lakes = lakes[lakes.lake_id.isin(a.lakes)] if a.lakes else lakes[lakes.role.isin(a.roles)]
    if a.limit_lakes:
        lakes = lakes.head(a.limit_lakes)
    tp = gpd.read_file(TP)
    tp = tp[tp.ID_L.isin(lakes.tp_id.astype(int))]
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(lambda r: fetch_lake(r, tp), lakes.itertuples()))


if __name__ == "__main__":
    main()
