"""Phase 0: filter GLOID per-year polygons, pick N stratified training lakes, print the download plan.

    python scripts/list_candidate_lakes.py --n 60 --seed 42
Filters (documented choices):
  * drop TS_OUTLIER == TRUE
  * drop polygons whose relative uncertainty (AREA_UNCERTAINTY/AREA) is above the sensor's 75th percentile
  * lake needs >= 4 valid years in S2 file, mean valid area >= 0.05 km2
Outputs: configs/training_lakes.csv
"""
from __future__ import annotations
import argparse, math, random
from pathlib import Path
import geopandas as gpd, numpy as np, pandas as pd, yaml
from pyproj import Transformer

R = Path(__file__).resolve().parents[1]
GL = R / "data" / "external" / "gloid"
BUF_KM, MIN_AREA, MIN_YEARS = 1.5, 0.05, 4
SIZE = [(0.05, 0.2, "S<0.2"), (0.2, 0.5, "M0.2-0.5"), (0.5, 1.0, "L0.5-1"), (1.0, 99, "XL>1")]
ELEV = [(0, 4500, "<4500"), (4500, 5000, "4500-5000"), (5000, 5400, "5000-5400"), (5400, 9999, ">5400")]


def band(v, bins):
    return next(n for a, b, n in bins if a <= v < b)


def load(sensor):
    g = gpd.read_file(GL / f"{sensor}_20172024_NTB_GLOID_v1.02.gpkg")
    g["rel_unc"] = g.AREA_UNCERTAINTY / g.AREA
    thr = g.rel_unc.quantile(0.75)
    g["valid"] = (g.TS_OUTLIER != "TRUE") & (g.rel_unc <= thr)
    return g, thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cohort-ids", nargs="*", default=["imja_tsho", "tsho_rolpa", "lower_barun", "thulagi"],
                    help="cohort lakes with confirmed GLOID matches (the others need manual verification)")
    a = ap.parse_args()
    s2, t2 = load("S2"); s1, t1 = load("S1")
    print(f"rel-uncertainty thresholds (75th pct): S2 {t2:.3f}, S1 {t1:.3f}")

    v2 = s2[s2.valid]
    agg = v2.groupby("GLO_ID").agg(years=("AREA_YEAR", lambda x: sorted(int(i) for i in x)), area=("AREA", "mean"),
                                    elev=("ELEVATION_MEAN", "first"), conn=("CONNECTIVITY", "first"), country=("COUNTRY", "first"),
                                    basin=("BASIN", "first"), lon=("LONGITUDE", "first"), lat=("LATITUDE", "first"))
    agg["n_years"] = agg.years.map(len)
    agg["s1_years"] = s1[s1.valid].groupby("GLO_ID").AREA_YEAR.apply(lambda x: sorted(int(i) for i in x))
    agg["s1_years"] = agg.s1_years.apply(lambda v: v if isinstance(v, list) else [])
    cand = agg[(agg.n_years >= MIN_YEARS) & (agg.area >= MIN_AREA)].copy()
    print(f"S2 lakes {s2.GLO_ID.nunique()} -> valid candidates {len(cand)} (>= {MIN_YEARS} valid years, area >= {MIN_AREA} km2)")

    # bbox (EPSG:4326) of the union of each lake's polygons -> AOI size estimate
    bb = s2[s2.GLO_ID.isin(cand.index)].to_crs(4326).dissolve("GLO_ID").bounds
    cand = cand.join(bb)
    cand["size"] = cand.area.map(lambda v: band(v, SIZE)); cand["elev_band"] = cand.elev.map(lambda v: band(v, ELEV))
    cand["stratum"] = cand["size"] + "|" + cand.elev_band + "|" + cand.conn

    # cohort lakes: nearest valid candidate (area >= 0.05) within 2 km of the yaml coordinate
    cfg = yaml.safe_load(open(R / "configs" / "lakes.yaml"))["lakes"]
    cohort, notes = {}, []
    for lk in cfg:
        d = np.hypot((cand.lon - lk["lon"]) * 98, (cand.lat - lk["lat"]) * 111)
        d = d[d < 2.0]
        if lk["id"] not in a.cohort_ids or d.empty:
            notes.append(f"{lk['id']}: not selected ({'inference_only, no GLOID mask' if lk['id']=='south_lhonak' else 'unconfirmed match / no valid candidate - excluded'})")
            continue
        gid = d.idxmin(); cohort[gid] = lk["id"]
        notes.append(f"{lk['id']}: {gid} at {d.min():.2f} km, area {cand.loc[gid,'area']:.2f} km2" + ("  [VERIFY match]" if d.min() > 0.5 else ""))

    rng = random.Random(a.seed)
    chosen = list(cohort)
    pool = cand.drop(index=chosen)
    cells = {k: g.index.tolist() for k, g in pool.groupby("stratum")}
    for v in cells.values():
        rng.shuffle(v)
    keys = sorted(cells); rng.shuffle(keys)
    while len(chosen) < a.n and any(cells.values()):   # round-robin over strata
        for k in keys:
            if cells[k] and len(chosen) < a.n:
                chosen.append(cells[k].pop())
    sel = cand.loc[chosen].copy()
    sel["cohort_id"] = [cohort.get(i, "") for i in sel.index]
    sel["aoi_w_km"] = ((sel.maxx - sel.minx) * 98 + 2 * BUF_KM).round(2)
    sel["aoi_h_km"] = ((sel.maxy - sel.miny) * 111 + 2 * BUF_KM).round(2)
    sel["years"] = sel.years.map(lambda v: " ".join(map(str, v))); sel["s1_years"] = sel.s1_years.map(lambda v: " ".join(map(str, v)))
    sel["outlet"] = "unknown (not in GLOID; CONNECTIVITY used as proxy)"
    out = sel.reset_index().rename(columns={"index": "GLO_ID", "elev": "elevation_m", "area": "area_km2"})
    cols = ["GLO_ID", "cohort_id", "lat", "lon", "area_km2", "elevation_m", "conn", "country", "basin", "size", "elev_band", "n_years", "years", "s1_years", "aoi_w_km", "aoi_h_km", "outlet"]
    (R / "configs").mkdir(exist_ok=True); out[cols].to_csv(R / "configs" / "training_lakes.csv", index=False)

    print("\nCohort matching:"); [print("  ", n) for n in notes]
    print(f"\nSelected {len(out)} lakes ({sum(bool(c) for c in out.cohort_id)} cohort). Strata coverage:")
    for c in ["size", "elev_band", "conn", "country"]:
        print(f"  {c}: {out[c].value_counts().to_dict()}")
    ly = int(out.n_years.sum()); ly1 = int(out.s1_years.map(lambda s: len(s.split())).sum())
    px = (out.aoi_w_km * 100 * out.aoi_h_km * 100)             # 10 m pixels per AOI
    mb_stack = float((px * 10 * 4).sum() / 1e6)
    mean_years = out.n_years.mean()
    print(f"\nPLAN: {len(out)} lakes, {ly} lake-years (S2 valid), {ly1} with valid S1 polygons")
    print(f"  GEE requests per lake-year: S2 May-Nov composite + S2 Aug-Oct composite + S1 composite = 3; + 1 DEM per lake")
    print(f"  total ~{3*ly + len(out)} requests (small AOIs, getDownloadURL; mean AOI {out.aoi_w_km.mean():.1f} x {out.aoi_h_km.mean():.1f} km, max {out.aoi_w_km.max():.1f} x {out.aoi_h_km.max():.1f})")
    print(f"  interim 10-ch float32 stacks ~{mb_stack*mean_years/1e3:.1f} GB (uncompressed est.); raw downloads ~{mb_stack*mean_years*0.7/1e3:.1f} GB; tiles ~{mb_stack*mean_years*1.3/1e3:.1f} GB before npz/float16 (~/3 after)")
    print("  wrote configs/training_lakes.csv")


if __name__ == "__main__":
    main()
