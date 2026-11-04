"""Automatically refine NDWI drafts into lake-only masks (no human QGIS review).

Keeps the connected water component nearest the catalogued lake centroid, removes
speckle and fills holes. Output goes to ``mask.npy`` with a QA json whose
``label_source`` is ``auto_refined_unreviewed`` so downstream code and reports can
tell these apart from QGIS-reviewed masks (written by qgis_export_helper.py).
A reviewed mask is never overwritten.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import rasterio
from pyproj import Transformer
from scipy import ndimage as ndi

from src.utils.config_loader import get_lake_by_id


def refine(draft: np.ndarray, cy: float, cx: float, max_dist_px: float = 150) -> np.ndarray:
    m = ndi.binary_opening(draft.astype(bool), structure=np.ones((7, 7)))
    m = ndi.binary_closing(m, structure=np.ones((3, 3)))
    lab, n = ndi.label(m)
    if n == 0:
        return np.zeros_like(draft, dtype=np.uint8)
    # distance from centroid to each component's nearest pixel
    yy, xx = np.indices(m.shape)
    d = np.hypot(yy - cy, xx - cx)
    best, best_d = 0, np.inf
    for i in range(1, n + 1):
        di = d[lab == i].min()
        if di < best_d:
            best, best_d = i, di
    if best_d > max_dist_px:
        return np.zeros_like(draft, dtype=np.uint8)
    out = ndi.binary_fill_holes(lab == best)
    return out.astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lake", default="south_lhonak")
    ap.add_argument("--overwrite", action="store_true", help="also replace existing auto masks")
    args = ap.parse_args()

    lake = get_lake_by_id(args.lake)
    root = ROOT / "data" / "processed" / args.lake
    done = []
    for ddir in sorted(p for p in root.iterdir() if p.is_dir()):
        draft_p = ddir / "mask_draft_ndwi_green.npy"
        if not draft_p.exists():
            continue
        qa_p = ddir / "mask_qa.json"
        if (ddir / "mask.npy").exists() and qa_p.exists():
            src = json.loads(qa_p.read_text()).get("label_source", "reviewed")
            if src != "auto_refined_unreviewed" or not args.overwrite:
                continue
        with rasterio.open(ROOT / "data" / "interim" / args.lake / "stacked" / f"{ddir.name}.tif") as r:
            t = Transformer.from_crs("EPSG:4326", r.crs, always_xy=True)
            x, y = t.transform(lake["lon"], lake["lat"])
            row, col = r.index(x, y)
        mask = refine(np.load(draft_p), row, col)
        np.save(ddir / "mask.npy", mask)
        qa = {
            "label_source": "auto_refined_unreviewed",
            "lake_pixels": int(mask.sum()),
            "draft_pixels": int(np.load(draft_p).sum()),
            "flag_small": bool(mask.sum() < 10),
            "flag_large": bool(mask.mean() > 0.95),
        }
        qa_p.write_text(json.dumps(qa, indent=2))
        done.append((ddir.name, qa["draft_pixels"], qa["lake_pixels"]))
    for d in done:
        print(d)


if __name__ == "__main__":
    main()
