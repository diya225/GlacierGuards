"""Phase 2: rasterize GLOID polygons onto each lake-year stack grid -> mask.npy (full grid, aligned with tile offsets).

Label rules
  * target lake polygon must be valid: TS_OUTLIER != TRUE and relative uncertainty <= sensor 75th percentile
  * primary label = S2 polygon (imagery is S2-led); if only S1 is valid, the S1 polygon is used
  * if both S1 and S2 polygons exist and IoU < 0.5 -> lake-year dropped and logged
  * every other GLOID lake polygon of that year that falls inside the AOI is also rasterized (any validity),
    so neighbouring lakes are never labelled as background
Outputs: data/processed/{GLO_ID}/{year}/mask.npy (uint8), mask_qa.json; reports/mask_manifest.csv (kept/dropped + reason)
    python scripts/gloid_to_masks.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import box

GL = ROOT / "data" / "external" / "gloid"
PROC, INTERIM = ROOT / "data" / "processed", ROOT / "data" / "interim"
IOU_MIN = 0.5


def load(sensor):
    g = gpd.read_file(GL / f"{sensor}_20172024_NTB_GLOID_v1.02.gpkg")
    g["rel_unc"] = g.AREA_UNCERTAINTY / g.AREA
    thr = g.rel_unc.quantile(0.75)
    g["valid"] = (g.TS_OUTLIER != "TRUE") & (g.rel_unc <= thr)
    return g


def rast(geoms, shape, transform):
    geoms = [g for g in geoms if g is not None and not g.is_empty]
    if not geoms:
        return np.zeros(shape, np.uint8)
    return rasterize([(g, 1) for g in geoms], out_shape=shape, transform=transform, fill=0, dtype="uint8", all_touched=False)


def main():
    lakes = pd.read_csv(ROOT / "configs" / "training_lakes.csv").GLO_ID.tolist()
    assert "south_lhonak" not in "".join(lakes).lower()
    S = {"S2": load("S2"), "S1": load("S1")}
    rows = []
    for gid in lakes:
        for ydir in sorted((INTERIM / gid).glob("20*")) if (INTERIM / gid).exists() else []:
            year = int(ydir.name)
            info = json.loads((ydir / "stack_meta.json").read_text())
            gr = info["grid"]
            H, W = gr["H"], gr["W"]
            tr = from_origin(gr["x0"], gr["y_top"], gr["res_m"], gr["res_m"])
            row = {"lake": gid, "year": year, "status": "", "reason": "", "iou_s1_s2": "", "label_source": "", "uncertainty": "", "water_px": 0}
            aoi = gpd.GeoSeries([box(gr["x0"], gr["y_top"] - H * gr["res_m"], gr["x0"] + W * gr["res_m"], gr["y_top"])], crs=gr["crs"])
            per = {}
            for s, g in S.items():
                gy = g[g.AREA_YEAR == year].to_crs(gr["crs"])
                nearby = gy[gy.intersects(aoi.iloc[0])]
                tgt = nearby[(nearby.GLO_ID == gid)]
                tv = tgt[tgt.valid]
                if tv.empty:
                    per[s] = None
                    continue
                m = rast(nearby.geometry, (H, W), tr)
                per[s] = {"mask": m, "unc": float(tv.AREA_UNCERTAINTY.iloc[0] / tv.AREA.iloc[0]), "tgt_px": int(rast(tv.geometry, (H, W), tr).sum())}
            if per["S2"] is None and per["S1"] is None:
                row.update(status="dropped", reason="no valid S1/S2 polygon this year")
            else:
                if per["S1"] is not None and per["S2"] is not None:
                    a, b = per["S1"]["mask"].astype(bool), per["S2"]["mask"].astype(bool)
                    iou = float((a & b).sum() / max((a | b).sum(), 1))
                    (PROC / gid / str(year)).mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(PROC / gid / str(year) / "masks_s1s2.npz", s1=a, s2=b)   # for QA of disagreements
                    row["iou_s1_s2"] = round(iou, 3)
                    if iou < IOU_MIN:
                        row.update(status="dropped", reason=f"S1/S2 polygon IoU {iou:.2f} < {IOU_MIN}")
                if not row["status"]:
                    src = "S2" if per["S2"] is not None else "S1"
                    mask = per[src]["mask"]
                    if per[src]["tgt_px"] < 5:
                        row.update(status="dropped", reason="target polygon < 5 px at 10 m")
                    else:
                        out = PROC / gid / str(year)
                        np.save(out / "mask.npy", mask)
                        qa = {"label_source": f"gloid_annual_{src}", "uncertainty_rel": per[src]["unc"], "iou_s1_s2": row["iou_s1_s2"],
                              "water_px": int(mask.sum()), "target_px": per[src]["tgt_px"], "shape": [H, W], "crs": gr["crs"]}
                        (out / "mask_qa.json").write_text(json.dumps(qa, indent=2))
                        row.update(status="kept", label_source=qa["label_source"], uncertainty=round(per[src]["unc"], 3), water_px=int(mask.sum()))
            rows.append(row)
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "mask_manifest.csv", index=False)
    print(df.status.value_counts().to_dict())
    print(df[df.status == "dropped"].reason.str.replace(r"[\d.]+", "#", regex=True).value_counts().to_dict())


if __name__ == "__main__":
    main()
