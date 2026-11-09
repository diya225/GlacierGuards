"""Random-forest pixel baseline (approach of Pathak et al. 2025) trained ONLY on the 22 pool lakes.

Reads kaggle_dataset/splits.json (train_pool) and kaggle_dataset/tiles/{lake}_{year}.npz.
Features = the same 10 channels as the U-Net, normalised with src/training/seg_dataset.normalize
(invalid pixels -> 0, as in the U-Net input). Samples ~200k pixels per class (balanced); half of the
negatives are hard negatives (within HARD_PX pixels of a lake edge). Never uses test or independent lakes.

    python scripts/train_rf_baseline.py [--per-class 200000]
Output: models/rf_baseline.joblib (git-ignored)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
from scipy.ndimage import distance_transform_edt
from sklearn.ensemble import RandomForestClassifier

from src.training.seg_dataset import normalize

HARD_PX = 30          # 300 m at 10 m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "kaggle_dataset"))
    ap.add_argument("--per-class", type=int, default=200_000)
    ap.add_argument("--out", default=str(ROOT / "models" / "rf_baseline.joblib"))
    a = ap.parse_args()
    data = Path(a.data)
    splits = json.loads((data / "splits.json").read_text())
    pool = set(splits["train_pool"])
    assert not pool & set(splits["test_lakes"]), "pool and test lakes overlap"
    files = [f for f in sorted((data / "tiles").glob("*.npz")) if f.name.rsplit("_", 1)[0] in pool]
    assert files, "no pool tiles found"
    rng = np.random.default_rng(0)
    # pass 1: count candidate pixels per tile, then draw a proportional random sample
    pos_n = neg_hard_n = neg_rand_n = 0
    cache = []
    for f in files:
        z = np.load(f)
        y, v = z["y"] == 1, z["valid"].any(1)
        hard = np.stack([(distance_transform_edt(~t) <= HARD_PX) & ~t if t.any() else np.zeros_like(t) for t in y])
        pos, hd, rd = y & v, hard & v, ~y & ~hard & v
        cache.append((f, pos.sum(), hd.sum(), rd.sum()))
        pos_n += pos.sum(); neg_hard_n += hd.sum(); neg_rand_n += rd.sum()
    p_pos = min(1.0, a.per_class / pos_n)
    p_hard = min(1.0, a.per_class / 2 / max(neg_hard_n, 1))
    p_rand = min(1.0, a.per_class / 2 / max(neg_rand_n, 1))
    print(f"pixels: pos {pos_n}, hard-neg {neg_hard_n}, other-neg {neg_rand_n}; sampling p = {p_pos:.3f}/{p_hard:.3f}/{p_rand:.3f}")
    X, Y = [], []
    for f, *_ in cache:
        z = np.load(f)
        y, v = z["y"] == 1, z["valid"].any(1)
        hard = np.stack([(distance_transform_edt(~t) <= HARD_PX) & ~t if t.any() else np.zeros_like(t) for t in y])
        sel = (y & v & (rng.random(y.shape) < p_pos)) | (hard & v & (rng.random(y.shape) < p_hard)) | (~y & ~hard & v & (rng.random(y.shape) < p_rand))
        for i in np.where(sel.any((1, 2)))[0]:
            raw = np.concatenate([z["x"][i, :7].astype(np.float32), z["elev"][i][None], z["x"][i, 7:].astype(np.float32)])
            xn = normalize(raw, z["valid"][i])
            X.append(xn[:, sel[i]].T)
            Y.append(y[i][sel[i]])
    X, Y = np.concatenate(X), np.concatenate(Y).astype(np.uint8)
    print(f"training set {X.shape}, positives {int(Y.sum())}")
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0).fit(X, Y)
    Path(a.out).parent.mkdir(exist_ok=True)
    joblib.dump({"model": rf, "train_lakes": sorted(pool), "n_samples": int(len(Y))}, a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
