"""Build a fixed lake ROI so area series measure ONE lake, not all water in the 3 km window.

The ROI is an independent manual outline (Third Pole inventory, 2022) buffered by --buffer metres, chosen
before looking at any burst. It is rasterised onto the lake's stack grid and saved as
data/predictions/{lake}/roi.npy; lake_area_m2 / lake_expected_area_m2 are added to area_series.csv.

    python scripts/make_lake_roi.py --lake south_lhonak --tp-id 4272 --buffer 400
"""
import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features

R = Path(__file__).resolve().parents[1]
TP = R / "data" / "external" / "third_pole_inventory" / "Third Pole" / "Third_Pole_Glacial Lake_2022.shp"

ap = argparse.ArgumentParser()
ap.add_argument("--lake", required=True)
ap.add_argument("--tp-id", type=int, required=True, help="ID_L in the Third Pole inventory")
ap.add_argument("--buffer", type=float, default=400.0)
a = ap.parse_args()

pred = R / "data" / "predictions" / a.lake
tifs = sorted((R / "data" / "interim" / a.lake / "stacked").glob("*.tif"))
with rasterio.open(tifs[0]) as r:
    tr, crs, shape = r.transform, r.crs, (r.height, r.width)
    px = abs(r.transform.a * r.transform.e)
t = gpd.read_file(TP)
poly = t.loc[t.ID_L == a.tp_id].to_crs(crs).geometry.iloc[0]
roi = features.rasterize([poly.buffer(a.buffer)], out_shape=shape, transform=tr, fill=0, default_value=1).astype(bool)
np.save(pred / "roi.npy", roi)

csv = pred / "area_series.csv"
s = pd.read_csv(csv)
la, ea = [], []
for d in s["date"]:
    p = np.load(pred / f"{d}.npy").astype(np.float32)
    la.append(float(((p > 0.5) & roi).sum() * px))
    ea.append(float((p * roi).sum() * px))
s["lake_area_m2"], s["lake_expected_area_m2"] = la, ea
s.to_csv(csv, index=False)
print(f"ROI {int(roi.sum())} px ({roi.sum() * px / 1e6:.2f} km2); manual outline {a.tp_id} area {poly.area / 1e6:.3f} km2")
print(s[["area_m2", "lake_area_m2"]].describe().round(0).to_string())
