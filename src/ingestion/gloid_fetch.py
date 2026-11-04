"""Phase 1a: download label-window composites for the GLOID training lakes via GEE.

Per lake (AOI = union of GLOID polygons + 1.5 km buffer, snapped to a 10 m UTM grid):
  data/raw/{GLO_ID}/dem.tif                      elevation, bilinear, 10 m
  data/raw/{GLO_ID}/{year}/s2_maynov.tif         S2 L2A SCL-masked median, 1 May-30 Nov  (B2,B3,B4,B8,B11 uint16, 0 = no data)
  data/raw/{GLO_ID}/{year}/s2_augoct.tif         same, 1 Aug-31 Oct (cleaner candidate)
  data/raw/{GLO_ID}/{year}/s1_julaug.tif         S1 IW VV/VH median dB, 1 Jul-30 Aug     (float32, -9999 = no data)
  data/raw/{GLO_ID}/{year}/meta.json             windows, scene counts, CRS, transform
Resumable: existing files are skipped; failures go to data/raw/_gloid_manifest.csv, not exceptions.

    python src/ingestion/gloid_fetch.py --limit-lakes 1      # smoke test
    python src/ingestion/gloid_fetch.py                      # all lakes in configs/training_lakes.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import ee
import geopandas as gpd
import pandas as pd
import rasterio
import requests
from rasterio.io import MemoryFile

from src.utils.auth import initialize_gee

log = logging.getLogger("gloid_fetch")
RAW = ROOT / "data" / "raw"
GL = ROOT / "data" / "external" / "gloid"
MANIFEST = RAW / "_gloid_manifest.csv"
BUF_M, RES = 1500, 10
MAX_BYTES = 30e6                      # getDownloadURL request cap is 32 MB
S2_BANDS = ["B2", "B3", "B4", "B8", "B11"]
WINDOWS = {
    "s2_maynov": ("{y}-05-01", "{y}-12-01"),
    "s2_augoct": ("{y}-08-01", "{y}-11-01"),
    "s1_julaug": ("{y}-07-01", "{y}-08-31"),
}
_lock = threading.Lock()


def log_manifest(**row):
    with _lock:
        RAW.mkdir(parents=True, exist_ok=True)
        new = not MANIFEST.exists()
        with open(MANIFEST, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["time", "lake", "year", "product", "status", "n_scenes", "detail"])
            if new:
                w.writeheader()
            w.writerow({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "n_scenes": "", "detail": "", **row})


def aoi_grid(gid, polys):
    """UTM grid (crs, x0, y_top, W, H) covering all-year polygons + buffer, snapped to 10 m."""
    g = polys[polys.GLO_ID == gid]
    c = g.to_crs(4326).geometry.union_all().centroid
    zone = int((c.x + 180) // 6) + 1
    crs = f"EPSG:{(32600 if c.y >= 0 else 32700) + zone}"
    minx, miny, maxx, maxy = g.to_crs(crs).total_bounds
    x0 = math.floor((minx - BUF_M) / RES) * RES
    y1 = math.ceil((maxy + BUF_M) / RES) * RES
    W = int(math.ceil((maxx + BUF_M - x0) / RES))
    H = int(math.ceil((y1 - (miny - BUF_M)) / RES))
    return crs, x0, y1, W, H


def request_params(grid):
    crs, x0, y1, W, H = grid
    return {"crs": crs, "crs_transform": [RES, 0, x0, 0, -RES, y1], "dimensions": f"{W}x{H}", "format": "GEO_TIFF"}


def download(image, grid, out: Path, retries=5):
    out.parent.mkdir(parents=True, exist_ok=True)
    for a in range(1, retries + 1):
        try:
            url = image.getDownloadURL({**request_params(grid), "filePerBand": False})
            r = requests.get(url, timeout=300)
            r.raise_for_status()
            data = r.content
            if len(data) < 200 or data[:1] == b"{":      # error JSON instead of a GeoTIFF
                raise RuntimeError(data[:200].decode(errors="ignore"))
            with MemoryFile(data) as mf, mf.open() as src:
                prof, arr = src.profile.copy(), src.read()
            prof.update(compress="deflate", zlevel=6, predictor=3 if arr.dtype.kind == "f" else 2,
                        tiled=True, blockxsize=256, blockysize=256)
            tmp = out.with_suffix(".tmp")
            with rasterio.open(tmp, "w", **prof) as dst:
                dst.write(arr)
            tmp.replace(out)
            return
        except Exception as e:                            # 429 / 5xx / network: exponential back-off
            wait = min(2 ** a * 5, 120)
            log.warning("%s attempt %d/%d failed (%s); sleeping %ds", out.name, a, retries, str(e)[:120], wait)
            time.sleep(wait)
    raise RuntimeError(f"gave up on {out}")


def s2_image(aoi, start, end):
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(aoi).filterDate(start, end)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60)))

    def mask(im):
        scl = im.select("SCL")
        bad = scl.eq(1).Or(scl.eq(3)).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10))  # saturated, shadow, cloud, cirrus (snow kept)
        return im.select(S2_BANDS).updateMask(bad.Not())

    return col.map(mask).median().unmask(0).toUint16(), col.size()


def s1_image(aoi, start, end):
    col = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(aoi).filterDate(start, end)
           .filter(ee.Filter.eq("instrumentMode", "IW"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")).select(["VV", "VH"]))
    return col.median().unmask(-9999).toFloat(), col.size()


def fetch_lake_year(gid, year, grid, aoi):
    ydir = RAW / gid / str(year)
    ydir.mkdir(parents=True, exist_ok=True)
    meta_p = ydir / "meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {
        "lake": gid, "year": year,
        "grid": {"crs": grid[0], "x0": grid[1], "y_top": grid[2], "W": grid[3], "H": grid[4], "res_m": RES},
        "products": {}}
    for prod, (s, e) in WINDOWS.items():
        out = ydir / f"{prod}.tif"
        if out.exists():
            continue
        win = [s.format(y=year), e.format(y=year)]
        try:
            img, n = (s2_image if prod.startswith("s2") else s1_image)(aoi, *win)
            n = int(n.getInfo())
            meta["products"][prod] = {"window": win, "n_scenes": n}
            if n == 0:
                log_manifest(lake=gid, year=year, product=prod, status="no_scenes", n_scenes=0)
                continue
            download(img, grid, out)
            log_manifest(lake=gid, year=year, product=prod, status="ok", n_scenes=n)
        except Exception as ex:
            log_manifest(lake=gid, year=year, product=prod, status="failed", detail=str(ex)[:200])
    meta_p.write_text(json.dumps(meta, indent=2))


def fetch_lake(row, polys):
    gid = row.GLO_ID
    try:
        grid = aoi_grid(gid, polys)
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
        s1y = set(map(int, str(row.s1_years).split())) if isinstance(row.s1_years, str) else set()
        for y in sorted(set(map(int, str(row.years).split())) | s1y):
            fetch_lake_year(gid, y, grid, aoi)
        log.info("lake %s done", gid)
    except Exception as ex:
        log_manifest(lake=gid, year="", product="all", status="failed", detail=str(ex)[:200])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-lakes", type=int)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    initialize_gee()
    lakes = pd.read_csv(ROOT / "configs" / "training_lakes.csv")
    if a.limit_lakes:
        lakes = lakes.head(a.limit_lakes)
    polys = gpd.read_file(GL / "S2_20172024_NTB_GLOID_v1.02.gpkg")
    polys = polys[polys.GLO_ID.isin(lakes.GLO_ID)]
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(lambda r: fetch_lake(r, polys), lakes.itertuples()))


if __name__ == "__main__":
    main()
