"""Stack one 10-channel image per SENTINEL-1 date (not per Sentinel-2 date).

coregister_stack.py builds a stack for every clear Sentinel-2 date, which leaves a lake in a monsoon-cloudy region
with only a handful of dates over many years (Imja Tsho: 11 dates in 7 seasons, but 178 Sentinel-1 scenes). This
script builds one stack per Sentinel-1 date instead. The optical channels (NDWI blue/green, NDVI, NDSI, NDGI) come from
the nearest Sentinel-2 index image within --max-gap days when one exists, otherwise they are left as no-data (-9999),
and predict_masks.py then runs the model in radar-only mode for that date (as it does for cloudy dates).

Same grid, same channel order, same no-data value and file format as coregister_stack.py.
    python src/preprocessing/stack_by_s1.py --lake imja_tsho
Output: data/interim/{lake}/stacked_s1date/{YYYY-MM-DD}.tif and {date}_meta.json (records the S2 date used, if any)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import rasterio
from tqdm import tqdm

from src.preprocessing.coregister_stack import (
    CHANNEL_NAMES, DATA_INTERIM, DATA_RAW, NODATA_VALUE, build_reference_grid, compute_slope_aspect, reproject_to_grid,
)
from src.utils.config_loader import get_lake_by_id, load_pipeline_config

OUT_SUBDIR = "stacked_s1date"


def nearest_s2(d: str, s2_dates: list[str], max_gap: int):
    t = datetime.strptime(d, "%Y-%m-%d")
    best, gap = None, 10 ** 9
    for s in s2_dates:
        g = abs((datetime.strptime(s, "%Y-%m-%d") - t).days)
        if g <= max_gap and g < gap:
            best, gap = s, g
    return best, gap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lake", required=True)
    ap.add_argument("--max-gap", type=int, default=5)
    a = ap.parse_args()
    cfg = load_pipeline_config()
    lake = get_lake_by_id(a.lake)
    res = cfg["data"]["resolution_m"]
    dem_path = DATA_RAW / lake["id"] / "dem" / "dem.tif"
    s1_dir, s2_dir = DATA_INTERIM / lake["id"] / "s1", DATA_INTERIM / lake["id"] / "s2"
    s1_dates = [f.stem.replace("_filtered", "") for f in sorted(s1_dir.glob("*_filtered.tif"))]
    s2_dates = [f.stem.replace("_indices", "") for f in sorted(s2_dir.glob("*_indices.tif"))] if s2_dir.exists() else []
    assert dem_path.exists() and s1_dates, f"need the DEM and filtered Sentinel-1 files for {a.lake}"

    dst_crs, tr, h, w = build_reference_grid(dem_path, lake, target_resolution_m=res)
    dem = reproject_to_grid(dem_path, tr, dst_crs, (h, w), band_indices=[1])[0]
    slope, aspect = compute_slope_aspect(dem, cell_size_m=res)
    out = DATA_INTERIM / lake["id"] / OUT_SUBDIR
    out.mkdir(parents=True, exist_ok=True)
    profile = {"driver": "GTiff", "dtype": "float32", "count": 10, "height": h, "width": w, "crs": dst_crs, "transform": tr,
               "nodata": NODATA_VALUE, "compress": "lzw", "tiled": True, "blockxsize": 256, "blockysize": 256}
    n_opt = 0
    for d in tqdm(s1_dates, desc=f"stack {a.lake}", unit="date"):
        p = out / f"{d}.tif"
        if p.exists():
            continue
        sar = reproject_to_grid(s1_dir / f"{d}_filtered.tif", tr, dst_crs, (h, w), band_indices=[1, 2])
        s2d, gap = nearest_s2(d, s2_dates, a.max_gap)
        if s2d:
            opt = reproject_to_grid(s2_dir / f"{s2d}_indices.tif", tr, dst_crs, (h, w), band_indices=[1, 2, 3, 4, 5])
            n_opt += 1
        else:
            opt = np.full((5, h, w), NODATA_VALUE, np.float32)
        stack = np.concatenate([sar, opt, dem[None], slope[None], aspect[None]], axis=0).astype(np.float32)
        with rasterio.open(p, "w", **profile) as dst:
            dst.write(stack)
            for i, name in enumerate(CHANNEL_NAMES, start=1):
                dst.update_tags(i, name=name)
        (out / f"{d}_meta.json").write_text(json.dumps({"date": d, "s1_date": d, "s2_date": s2d, "s2_gap_days": gap if s2d else None,
                                                        "lake_id": lake["id"], "channels": CHANNEL_NAMES, "shape": [10, h, w]}, indent=1))
    print(f"{a.lake}: {len(list(out.glob('*.tif')))} stacks in {out} ({n_opt} with an optical scene within {a.max_gap} days)")


if __name__ == "__main__":
    main()
