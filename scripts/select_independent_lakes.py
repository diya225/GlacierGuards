"""Select the independent test lakes (lakes the model never trained on) from the Third Pole inventory.

Reference labels for these lakes are the inventory's manually delineated Sentinel-2 polygons (2018/2020/2022),
NOT GLOID. Deterministic: same inputs + --seed give the same CSV.

Rules
  * regions: Central + Eastern Himalaya ("core", 40 lakes) and Western Himalaya ("geo_shift", 10 lakes, a region the
    training lakes do not cover)
  * lake area 2022 >= 0.02 km2 (about 200 px at 10 m); area stable across 2018/2020/2022 (<= 25 % change per step),
    so one fixed 2022 polygon is a fair reference for a Jul-Nov 2022 composite
  * >= 3 km from every training lake (configs/training_lakes.csv) and from South Lhonak
  * >= 3 km between selected lakes, so image windows do not overlap
  * size classes quota per group (see QUOTA); within a class, dam types are rotated round-robin; supraglacial lakes capped
  * 'backup' lakes (2 per stratum) replace primary lakes whose download fails

    python scripts/select_independent_lakes.py            ->  configs/independent_test_lakes.csv
"""
import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

R = Path(__file__).resolve().parents[1]
TP = R / "data" / "external" / "third_pole_inventory" / "Third Pole" / "Third_Pole_Glacial Lake_2022.shp"
GLOID = R / "data" / "external" / "gloid" / "S2_20172024_NTB_GLOID_v1.02.gpkg"
SIZES = [("XS", 0.02, 0.05), ("S", 0.05, 0.2), ("M", 0.2, 0.5), ("L", 0.5, 1.0), ("XL", 1.0, 1e9)]
GROUPS = {"core": ["Central Himalaya", "Eastern Himalaya"], "geo_shift": ["Western Himalaya"]}
QUOTA = {"core": {"XS": 6, "S": 10, "M": 8, "L": 8, "XL": 8}, "geo_shift": {"XS": 2, "S": 3, "M": 2, "L": 2, "XL": 1}}
BACKUPS_PER_STRATUM = 2
SUPRA_CAP = 3
MIN_SPACING_M = 3000
MIN_TRAIN_DIST_M = 3000

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)

t = gpd.read_file(TP)
t["size_class"] = pd.cut(t.Area_2022, [s[1] for s in SIZES] + [1e9], labels=[s[0] for s in SIZES], right=False)
rp = gpd.GeoSeries(t.geometry.representative_point(), crs=4326)
t["utm"] = [int(32600 + (x + 180) // 6 + 1) for x in rp.x]
t["x"], t["y"] = rp.x, rp.y

# metric coordinates in a single projection suitable for the whole region (Asia North Albers-like: use EPSG:102025 equiv)
ALB = "+proj=aea +lat_1=15 +lat_2=65 +lat_0=30 +lon_0=95 +x_0=0 +y_0=0 +datum=WGS84 +units=m"
tm = gpd.GeoSeries(t.geometry, crs=4326).to_crs(ALB)

train = pd.read_csv(R / "configs" / "training_lakes.csv")
tp_pts = [Point(lo, la) for lo, la in zip(train.lon, train.lat)] + [Point(88.1976, 27.9124)]
tr_m = gpd.GeoSeries(tp_pts, crs=4326).to_crs(ALB)
near_train = np.zeros(len(t), bool)
for p in tr_m:
    near_train |= (tm.distance(p) < MIN_TRAIN_DIST_M).values

stable = ((t.Area_2022 - t.Area_2020).abs() / t.Area_2022 <= 0.25) & ((t.Area_2020 - t.Area_2018).abs() / t.Area_2020 <= 0.25)
ok = (t.Area_2022 >= 0.02) & stable & ~near_train

rows = []
chosen_pts = []
for grp, regions in GROUPS.items():
    for cls, _, _ in SIZES:
        cand = t[ok & t.Region.isin(regions) & (t.size_class == cls)].copy()
        cand["r"] = rng.random(len(cand))
        # round-robin over dam types (rarest first), random within a type
        cand = cand.sort_values("r")
        queues = {d: list(g.index) for d, g in cand.groupby("Dam_T")}
        order = []
        while any(queues.values()):
            for d in sorted(queues, key=lambda k: len(queues[k])):
                if queues[d]:
                    order.append(queues[d].pop(0))
        want = QUOTA[grp][cls] + BACKUPS_PER_STRATUM
        picked = []
        for i in order:
            if len(picked) >= want:
                break
            if t.loc[i, "Dam_T"] == "supra" and sum(r["dam_type"] == "supra" for r in rows) >= SUPRA_CAP:
                continue
            pm = tm.loc[i].representative_point()
            if any(pm.distance(q) < MIN_SPACING_M for q in chosen_pts):
                continue
            picked.append(i)
            chosen_pts.append(pm)
            r = t.loc[i]
            rows.append({"lake_id": f"TP_{int(r.ID_L)}", "tp_id": int(r.ID_L), "region_group": grp, "region": r.Region,
                         "lat": round(r.Lat, 5), "lon": round(r.Lon, 5), "utm_epsg": int(r.utm), "dam_type": r.Dam_T,
                         "elev_m": int(r.Elev), "size_class": cls, "area_2018_km2": round(r.Area_2018, 4),
                         "area_2020_km2": round(r.Area_2020, 4), "area_2022_km2": round(r.Area_2022, 4),
                         "area_error_2022_km2": round(r.Error22, 4), "stratum": f"{grp}/{cls}",
                         "role": "primary" if len(picked) <= QUOTA[grp][cls] else "backup", "years": "2022"})
df = pd.DataFrame(rows)

# does a GLOID S2 2022 polygon exist for the lake? (lets us also score GLOID itself on the same set)
g = gpd.read_file(GLOID)
g = g[g.AREA_YEAR == 2022].to_crs(4326)
pts = gpd.GeoDataFrame(df[["lake_id"]], geometry=[Point(x, y) for x, y in zip(t.set_index("ID_L").loc[df.tp_id, "x"], t.set_index("ID_L").loc[df.tp_id, "y"])], crs=4326)
j = gpd.sjoin(pts, g[["GLO_ID", "geometry"]], how="left", predicate="within").drop_duplicates("lake_id")
df["gloid_id_2022"] = j.set_index("lake_id").loc[df.lake_id, "GLO_ID"].values

# rank primaries first, then backups, inside each stratum order is already randomised
df["_r"] = (df.role == "backup").astype(int)
df = df.sort_values(["_r", "region_group", "stratum", "tp_id"]).drop(columns="_r").reset_index(drop=True)
out = R / "configs" / "independent_test_lakes.csv"
df.to_csv(out, index=False)

prim = df[df.role == "primary"]
print(f"{len(prim)} primary + {(df.role == 'backup').sum()} backup lakes -> {out}")
print("\nprimary by group/size:\n", prim.groupby(["region_group", "size_class"]).size().unstack(fill_value=0).to_string())
print("\nprimary by dam type:", prim.dam_type.value_counts().to_dict())
print("primary with a GLOID 2022 polygon:", int(prim.gloid_id_2022.notna().sum()), "of", len(prim))
short = {f"{g}/{c}": QUOTA[g][c] - int(((prim.region_group == g) & (prim.size_class == c)).sum()) for g in QUOTA for c in QUOTA[g]}
print("shortfalls vs quota:", {k: v for k, v in short.items() if v > 0} or "none")
