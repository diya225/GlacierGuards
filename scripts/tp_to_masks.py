"""Reference masks for the independent test lakes from the Third Pole 2022 inventory (manual polygons).

For every lake in configs/independent_test_lakes.csv that has data/interim/{lake}/2022/stack_meta.json:
  data/eval/{lake}/ref_all.npy      uint8, ALL inventory 2022 polygons intersecting the window (neighbours are not background)
  data/eval/{lake}/ref_target.npy   uint8, the target polygon only (ID_L == tp_id)
  data/eval/{lake}/ref.json         raster area vs CSV area (km2), % difference, flag if > 5 %
Summary: reports/independent_eval/reference_check.csv

    python scripts/tp_to_masks.py
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
TP = ROOT / "data" / "external" / "third_pole_inventory" / "Third Pole" / "Third_Pole_Glacial Lake_2022.shp"
INTERIM, EVAL = ROOT / "data" / "interim", ROOT / "data" / "eval"
TOL_PCT = 5.0


def rast(geoms, shape, transform):
    geoms = [g for g in geoms if g is not None and not g.is_empty]
    if not geoms:
        return np.zeros(shape, np.uint8)
    return rasterize([(g, 1) for g in geoms], out_shape=shape, transform=transform, fill=0, dtype="uint8", all_touched=False)


def main():
    lakes = pd.read_csv(ROOT / "configs" / "independent_test_lakes.csv")
    tp = gpd.read_file(TP)
    rows = []
    for r in lakes.itertuples():
        meta_p = INTERIM / r.lake_id / "2022" / "stack_meta.json"
        if not meta_p.exists():
            continue
        gr = json.loads(meta_p.read_text())["grid"]
        H, W, res = gr["H"], gr["W"], gr["res_m"]
        tr = from_origin(gr["x0"], gr["y_top"], res, res)
        win = box(gr["x0"], gr["y_top"] - H * res, gr["x0"] + W * res, gr["y_top"])
        g = tp.to_crs(gr["crs"])
        near = g[g.intersects(win)]
        tgt = near[near.ID_L == int(r.tp_id)]
        assert len(tgt) == 1, f"{r.lake_id}: target polygon not found in window"
        ref_all, ref_t = rast(near.geometry, (H, W), tr), rast(tgt.geometry, (H, W), tr)
        out = EVAL / r.lake_id
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "ref_all.npy", ref_all)
        np.save(out / "ref_target.npy", ref_t)
        a_ras = float(ref_t.sum() * res * res / 1e6)
        diff = 100 * (a_ras - r.area_2022_km2) / r.area_2022_km2
        info = {"lake_id": r.lake_id, "tp_id": int(r.tp_id), "area_raster_km2": round(a_ras, 4), "area_csv_km2": r.area_2022_km2,
                "diff_pct": round(diff, 2), "n_polygons_in_window": int(len(near)), "flag_over_5pct": bool(abs(diff) > TOL_PCT)}
        (out / "ref.json").write_text(json.dumps(info, indent=2))
        rows.append(info)
        print(info, flush=True)
    outd = ROOT / "reports" / "independent_eval"
    outd.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(outd / "reference_check.csv", index=False)
    if len(df):
        print(f"{len(df)} lakes; {int(df.flag_over_5pct.sum())} with area difference > {TOL_PCT}% (investigate): "
              f"{df[df.flag_over_5pct].lake_id.tolist()}")


if __name__ == "__main__":
    main()
