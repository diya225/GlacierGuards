"""Build a fixed lake ROI so area series measure ONE lake, not all water in the 3 km window.

The ROI is an independent manual outline (Third Pole inventory, 2022) buffered by --buffer metres, chosen
before looking at any burst. It is rasterised onto the lake's stack grid and saved as
data/predictions/{lake}/roi.npy; lake_area_m2 / lake_expected_area_m2 are added to area_series.csv.

    python scripts/make_lake_roi.py --lake south_lhonak --tp-id 4272 --buffer 400
    python scripts/make_lake_roi.py --lake thame_lower_ngole --tp-id 4556 --buffer 100 --exclude-others   (small lake: 100 m margin)
    python scripts/make_lake_roi.py --lake gongbatongsha_co --circle 86.0700,28.0781,150   (no mapped outline: fixed disc)
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
ap.add_argument("--tp-id", type=int, help="ID_L in the Third Pole inventory (manual 2022 outline)")
ap.add_argument("--circle", help="lon,lat,radius_m: fixed disc ROI for a lake with no mapped outline (for example one that drained before 2017)")
ap.add_argument("--exclude-others", action="store_true", help="remove the manual outlines of OTHER inventory lakes from the ROI, so neighbouring lakes are not counted")
ap.add_argument("--buffer", type=float, default=400.0)
ap.add_argument("--tp-year", type=int, default=2022, help="inventory snapshot to take the outline from (2018/2020/2022); for an outburst lake use the last snapshot BEFORE the event so the ROI is the pre-event lake")
ap.add_argument("--stack-subdir", default="stacked")
a = ap.parse_args()

pred = R / "data" / "predictions" / a.lake
tifs = sorted((R / "data" / "interim" / a.lake / a.stack_subdir).glob("*.tif"))
with rasterio.open(tifs[0]) as r:
    tr, crs, shape = r.transform, r.crs, (r.height, r.width)
    px = abs(r.transform.a * r.transform.e)
from shapely.geometry import Point

t = gpd.read_file(TP)
if a.circle:
    lo, la, rad = (float(x) for x in a.circle.split(","))
    c = gpd.GeoSeries([Point(lo, la)], crs=4326).to_crs(crs).iloc[0]
    poly = c.buffer(rad)
    roi_geom = poly
else:
    assert a.tp_id is not None, "give --tp-id or --circle"
    poly = t.loc[t.ID_L == a.tp_id].to_crs(crs).geometry.iloc[0]
    if a.tp_year != 2022:
        ty = gpd.read_file(TP.parent / f"Third_Pole_Glacial Lake_{a.tp_year}.shp").to_crs(crs)
        pt = poly.representative_point()
        hit = ty[ty.geometry.contains(pt)]
        if hit.empty:
            hit = ty[ty.geometry.distance(pt) < 100]
        assert len(hit), f"no {a.tp_year} inventory polygon at the 2022 outline of lake {a.tp_id}"
        poly = hit.loc[hit.geometry.area.idxmax()].geometry
    roi_geom = poly.buffer(a.buffer)
roi = features.rasterize([roi_geom], out_shape=shape, transform=tr, fill=0, default_value=1).astype(bool)
if a.exclude_others and not a.circle:
    oth = t.loc[t.ID_L != a.tp_id].to_crs(crs)
    oth = oth[oth.intersects(roi_geom)]
    if len(oth):
        roi &= ~features.rasterize(list(oth.geometry), out_shape=shape, transform=tr, fill=0, default_value=1).astype(bool)
        print(f"excluded {len(oth)} neighbouring inventory lake(s) from the ROI")
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
print(f"ROI {int(roi.sum())} px ({roi.sum() * px / 1e6:.3f} km2); {'disc' if a.circle else 'manual outline ' + str(a.tp_id) + ' (' + str(a.tp_year) + ' snapshot)'} area {poly.area / 1e6:.3f} km2")
print(s[["area_m2", "lake_area_m2"]].describe().round(0).to_string())
