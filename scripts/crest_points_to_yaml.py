"""Turn the QGIS crest-point GeoPackage into the YAML block for configs/lakes.yaml, with quality checks.

QGIS side (see docs/MILESTONES_7_TO_END.md, section "QGIS: moraine crest points"):
  GeoPackage with two layers
    crest_points : Point layer; fields  lake_id (text), elev_m (real), is_lowest (integer 0/1), note (text, optional)
    dam          : any layer (a table is fine); fields  lake_id (text), height_m (real), width_m (real),
                   has_outlet_channel (integer 0/1), toe_elev_m (real, optional), dem_source (text, optional), confidence (integer 1-3, optional)

    python scripts/crest_points_to_yaml.py data/qgis/crest_points.gpkg
    python scripts/crest_points_to_yaml.py data/qgis/crest_points.gpkg --out configs/crest_points_generated.yaml

lat/lon are taken from the point geometry (reprojected to WGS84), never typed by hand. The script never edits
lakes.yaml: you review the printed block and paste it in. Exit code 1 if any lake fails a check.
"""
import argparse
import math
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import yaml

R = Path(__file__).resolve().parents[1]
MIN_PTS, MAX_PTS = 5, 10


def hav_km(lat1, lon1, lat2, lon2):
    p = math.pi / 180
    a = math.sin((lat2 - lat1) * p / 2) ** 2 + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lon2 - lon1) * p / 2) ** 2
    return 12742 * math.asin(math.sqrt(a))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gpkg")
    ap.add_argument("--out")
    ap.add_argument("--max-lake-km", type=float, default=None, help="max distance of a crest point from the lake centre (default: lake buffer_km + 1)")
    a = ap.parse_args()

    lakes = {l["id"]: l for l in yaml.safe_load(open(R / "configs" / "lakes.yaml", encoding="utf-8"))["lakes"]}
    pts = gpd.read_file(a.gpkg, layer="crest_points").to_crs(4326)
    dam = pd.DataFrame(gpd.read_file(a.gpkg, layer="dam").drop(columns="geometry", errors="ignore"))
    errors, warns, blocks = [], [], []

    for lid, g in pts.groupby("lake_id"):
        e, w = [], []
        if lid not in lakes:
            errors.append(f"{lid}: not in configs/lakes.yaml")
            continue
        lake = lakes[lid]
        if not (MIN_PTS <= len(g) <= MAX_PTS):
            w.append(f"{len(g)} crest points (expected {MIN_PTS}-{MAX_PTS})")
        low = g[g["is_lowest"].astype(int) == 1]
        if len(low) != 1:
            e.append(f"{len(low)} points flagged is_lowest (need exactly 1)")
        elif low["elev_m"].iloc[0] > g["elev_m"].min() + 0.5:
            e.append(f"the flagged lowest point is {low['elev_m'].iloc[0]:.1f} m but another point is {g['elev_m'].min():.1f} m")
        if g["elev_m"].isna().any():
            e.append("elev_m missing for some points")
        lim = a.max_lake_km or (lake.get("buffer_km", 2.0) + 1.0)
        far = [round(hav_km(lake["lat"], lake["lon"], p.y, p.x), 1) for p in g.geometry]
        if max(far) > lim:
            e.append(f"a crest point is {max(far)} km from the lake centre (limit {lim} km): wrong lake or wrong CRS?")
        if abs(g["elev_m"].mean() - lake["alt_m"]) > 300:
            w.append(f"mean crest elevation {g['elev_m'].mean():.0f} m differs from lakes.yaml alt_m {lake['alt_m']} m by more than 300 m")
        d = dam[dam["lake_id"] == lid]
        if len(d) != 1:
            e.append(f"{len(d)} rows in the dam layer (need exactly 1)")
            h = wd = None
            outlet = False
        else:
            h, wd = d["height_m"].iloc[0], d["width_m"].iloc[0]
            outlet = bool(int(d["has_outlet_channel"].iloc[0]))
            if not (h and h > 0) or not (wd and wd > 0):
                e.append("dam height_m and width_m must both be > 0")
            elif wd / h < 1:
                w.append(f"width/height = {wd / h:.2f} (very narrow dam; check the profile)")
        if e:
            errors += [f"{lid}: {m}" for m in e]
        warns += [f"{lid}: {m}" for m in w]
        if not e:
            lines = [f"# --- {lid} ---", "crest_points:"]
            for _, r in g.sort_values("elev_m").iterrows():
                lines.append(f"  - {{lat: {r.geometry.y:.5f}, lon: {r.geometry.x:.5f}, elev_m: {r['elev_m']:.1f}, is_lowest: {str(int(r['is_lowest']) == 1).lower()}}}")
            lines += ["dam:", f"  height_m: {h:.1f}", f"  width_m: {wd:.1f}", f"  has_outlet_channel: {str(outlet).lower()}"]
            blocks.append("\n".join(lines))

    text = "\n\n".join(blocks)
    print(text if text else "(no lake passed all checks)")
    for m in warns:
        print("WARNING:", m, file=sys.stderr)
    for m in errors:
        print("ERROR:", m, file=sys.stderr)
    if a.out and blocks:
        Path(a.out).write_text(text + "\n", encoding="utf-8")
        print(f"\nwritten to {a.out} (review, then paste each block under the matching lake in configs/lakes.yaml)", file=sys.stderr)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
