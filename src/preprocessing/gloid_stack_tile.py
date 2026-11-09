"""Phase 1b: raw composites -> 10-channel stacks -> 256x256 tiles for the GLOID training lakes.

Reuses existing functions: enhanced_lee_filter / linear_to_db (sar_preprocess), compute_indices_from_bands
(optical_indices), compute_slope_aspect (coregister_stack), tile_image (tile_and_pad).
Same channel order as the Milestone-3 stacks: [VV, VH, NDWI_blue, NDWI_green, NDVI, NDSI, NDGI, elevation, slope, aspect].

  data/interim/{GLO_ID}/{year}/stack.tif + stack_meta.json   (float32, deflate, nodata -9999)
  data/processed/{GLO_ID}/{year}/tiles.npz + tile_index.json
      tiles.npz: x (n,9,256,256) float16 [all channels except elevation; invalid -> 0],
                 elev (n,256,256) float32, valid (n,10,256,256) bool
Which S2 composite is used: Aug-Oct if >=90% valid pixels, else whichever of the two has more valid pixels.
Resumable: lake-years with tiles.npz are skipped.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import rasterio

from src.preprocessing.coregister_stack import compute_slope_aspect
from src.preprocessing.optical_indices import compute_indices_from_bands
from src.preprocessing.sar_preprocess import enhanced_lee_filter, linear_to_db
from src.preprocessing.tile_and_pad import tile_image

RAW, INTERIM, PROC = (ROOT / "data" / p for p in ("raw", "interim", "processed"))
NODATA = -9999.0
OVERLAP = 32
CHANNELS = ["VV", "VH", "NDWI_blue", "NDWI_green", "NDVI", "NDSI", "NDGI", "elevation", "slope", "aspect"]


def read(p):
    with rasterio.open(p) as r:
        return r.read().astype(np.float32), r.profile


def s2_valid(a):
    return (a > 0).all(0)


def build_stack(gid, year):
    ydir = RAW / gid / str(year)
    meta = json.loads((ydir / "meta.json").read_text())
    dem_p = RAW / gid / "dem.tif"
    cands = {k: ydir / f"{k}.tif" for k in ("s2_augoct", "s2_maynov") if (ydir / f"{k}.tif").exists()}
    if not cands or not dem_p.exists():
        return None, "missing s2 or dem"
    vf = {}
    for k, p in cands.items():
        a, prof = read(p)
        vf[k] = (float(s2_valid(a).mean()), a)
    pick = "s2_augoct" if ("s2_augoct" in vf and vf["s2_augoct"][0] >= 0.9) else max(vf, key=lambda k: vf[k][0])
    s2 = vf[pick][1]
    valid_s2 = s2_valid(s2)
    s2 = np.where(valid_s2, s2, np.nan)
    idx = compute_indices_from_bands(s2[0], s2[1], s2[2], s2[3], s2[4])   # blue, green, red, nir, swir
    s1_p = ydir / "s1_julaug.tif"
    if s1_p.exists():
        s1, _ = read(s1_p)
        s1 = np.where(s1 == NODATA, np.nan, s1)
        vv = linear_to_db(enhanced_lee_filter(s1[0], 7, input_is_db=True))   # same sequence as process_s1_file
        vh = linear_to_db(enhanced_lee_filter(s1[1], 7, input_is_db=True))
    else:
        vv = vh = np.full(s2.shape[1:], np.nan, np.float32)
    dem, prof = read(dem_p)
    dem = dem[0]
    slope, aspect = compute_slope_aspect(dem, 10.0)
    stack = np.stack([vv, vh, idx["NDWI_blue"], idx["NDWI_green"], idx["NDVI"], idx["NDSI"], idx["NDGI"], dem, slope, aspect]).astype(np.float32)
    stack = np.where(np.isfinite(stack), stack, NODATA)
    info = {"lake": gid, "year": year, "s2_used": pick, "s2_valid_fraction": {k: round(v[0], 4) for k, v in vf.items()},
            "s1_present": s1_p.exists(), "s2_scenes": {k: meta["products"].get(k, {}).get("n_scenes") for k in cands},
            "s1_scenes": meta["products"].get("s1_julaug", {}).get("n_scenes"), "s2_windows": {k: meta["products"].get(k, {}).get("window") for k in cands},
            "grid": meta["grid"], "channels": CHANNELS,
            "vv_valid_fraction": round(float((stack[0] != NODATA).mean()), 4),
            "ndwi_green_valid_fraction": round(float((stack[3] != NODATA).mean()), 4)}
    return (stack, prof, info), None


def process(gid, year, skip_tiles=False):
    out_t = PROC / gid / str(year)
    if (INTERIM / gid / str(year) / "stack.tif").exists() if skip_tiles else (out_t / "tiles.npz").exists():
        return "skip"
    res, err = build_stack(gid, year)
    if res is None:
        return err
    stack, prof, info = res
    idir = INTERIM / gid / str(year)
    idir.mkdir(parents=True, exist_ok=True)
    p = prof.copy()
    p.update(count=10, dtype="float32", nodata=NODATA, compress="deflate", zlevel=6, predictor=3, tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(idir / "stack.tif", "w", **p) as dst:
        dst.write(stack)
        dst.descriptions = tuple(CHANNELS)
    (idir / "stack_meta.json").write_text(json.dumps(info, indent=2))
    if skip_tiles:
        return "ok"
    tiles = tile_image(stack, 256, OVERLAP)
    arr = np.stack([t for t, _, _ in tiles])
    valid = np.isfinite(arr) & (arr != NODATA)
    arr = np.where(valid, arr, 0).astype(np.float32)
    out_t.mkdir(parents=True, exist_ok=True)
    keep = [i for i in range(10) if i != 7]
    np.savez_compressed(out_t / "tiles.npz", x=arr[:, keep].astype(np.float16), elev=arr[:, 7], valid=valid)
    (out_t / "tile_index.json").write_text(json.dumps({
        "lake_id": gid, "year": year, "tile_size": 256, "overlap": OVERLAP, "shape": list(stack.shape), "channels": CHANNELS,
        "x_channels": [CHANNELS[i] for i in keep], "tiles": [{"row_offset": r, "col_offset": c} for _, r, c in tiles]}, indent=1))
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-lakes", type=int)
    ap.add_argument("--lakes-csv", default=str(ROOT / "configs" / "training_lakes.csv"))
    ap.add_argument("--id-col", default="GLO_ID")
    ap.add_argument("--skip-tiles", action="store_true", help="stop after stack.tif + stack_meta.json (evaluation lakes)")
    a = ap.parse_args()
    lakes = pd.read_csv(a.lakes_csv)[a.id_col].tolist()[: a.limit_lakes]
    for gid in lakes:
        for ydir in sorted(p for p in (RAW / gid).glob("20*") if (p / "meta.json").exists()):
            st = process(gid, int(ydir.name), a.skip_tiles)
            print(gid, ydir.name, st, flush=True)


if __name__ == "__main__":
    main()
