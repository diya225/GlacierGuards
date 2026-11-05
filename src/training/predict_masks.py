"""Run the trained segmenter on every stacked date and write probability maps + area series.

Dates with <80% valid optical pixels are run SAR+terrain only (optical channels zeroed),
matching the optical-dropout augmentation used in training.

    python src/training/predict_masks.py --lake south_lhonak
Output: data/predictions/{lake}/{date}.npy (float16 prob) and area_series.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import rasterio
import torch

from src.models.segmentation_model import build_model
from src.training.seg_dataset import NODATA, OPTICAL_IDX, normalize


def load_model(ckpt_path: Path, dev="cpu"):
    ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
    m = build_model("efficientnet-b3", len(ck["channels"]), pretrained=False)
    m.load_state_dict(ck["model"])
    return m.to(dev).eval(), ck["channels"]


@torch.no_grad()
def predict_scene(model, chans, stack, min_optical=0.8, dev="cpu"):
    valid = np.isfinite(stack) & (stack != NODATA)
    cov = float(valid[2:7].all(0).mean())
    x = normalize(stack, valid)
    use_optical = cov >= min_optical
    if not use_optical:
        x[OPTICAL_IDX] = 0
    h, w = x.shape[1:]
    ph, pw = (-h) % 32, (-w) % 32
    xp = np.pad(x[chans], ((0, 0), (0, ph), (0, pw)))
    p = torch.sigmoid(model(torch.from_numpy(xp)[None].to(dev)))[0, 0].cpu().numpy()[:h, :w]
    return p, cov, use_optical


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lake", default="south_lhonak")
    ap.add_argument("--ckpt", default=str(ROOT / "models" / "segmentation_checkpoint.pt"))
    ap.add_argument("--stack-subdir", default="stacked", help="folder under data/interim/{lake}/ holding the stacks (stacked_s1date for the Sentinel-1-date stacks)")
    a = ap.parse_args()
    model, chans = load_model(Path(a.ckpt))
    out = ROOT / "data" / "predictions" / a.lake
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for tif in sorted((ROOT / "data" / "interim" / a.lake / a.stack_subdir).glob("*.tif")):
        with rasterio.open(tif) as r:
            stack = r.read().astype(np.float32)
            px = abs(r.transform.a * r.transform.e)
        p, cov, opt = predict_scene(model, chans, stack)
        np.save(out / f"{tif.stem}.npy", p.astype(np.float16))
        lab = (ROOT / "data" / "processed" / a.lake / tif.stem / "mask.npy").exists()
        rows.append({"date": tif.stem, "area_m2": float((p > 0.5).sum() * px), "expected_area_m2": float(p.sum() * px),
                     "optical_valid_fraction": round(cov, 4), "used_optical": opt, "in_training_labels": lab})
        print(rows[-1])
    pd.DataFrame(rows).to_csv(out / "area_series.csv", index=False)


if __name__ == "__main__":
    main()
