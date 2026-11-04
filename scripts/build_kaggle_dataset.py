"""Phase 4: build kaggle_dataset/ + glacierguards_kaggle.zip.

  kaggle_dataset/tiles/{GLO_ID}_{year}.npz   x (n,9,256,256 f16), elev (n,256,256 f32), valid (n,10,256,256 bool), y (n,256,256 u8), offsets
  kaggle_dataset/splits.json   test_lakes (never used for train/val), train_pool, cv_folds (5, lake-grouped), lolo
  kaggle_dataset/manifest.csv, DATA_PROVENANCE.md, dataset-metadata.json, code/ (src + configs)
Tiles: all tiles with water + a random sample of water-free tiles so negatives are ~25% of tiles.
    python scripts/build_kaggle_dataset.py [--max-gb 5]
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1]
PROC = R / "data" / "processed"
OUT = R / "kaggle_dataset"
NEG_TARGET = 0.25
TEST_COHORT = ["imja_tsho", "thulagi"]   # held-out cohort lakes: later stages run on lakes the model never trained on
N_TEST_OTHER = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-gb", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    meta = pd.read_csv(R / "configs" / "training_lakes.csv").set_index("GLO_ID")
    meta["cohort_id"] = meta.cohort_id.fillna("")
    assert not any("lhonak" in str(c).lower() for c in meta.cohort_id), "South Lhonak must be excluded"
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "tiles").mkdir(parents=True)

    # drop lake-years with no usable input (SAR and optical both < 50% valid); drop lakes left with < 3 years
    rep = pd.read_csv(R / "reports" / "dataset_report.csv")
    rep["usable"] = (rep[["valid_VV", "valid_NDWI_green"]].max(axis=1) >= 0.5)
    ok_years = {(r.lake, str(r.year)) for r in rep.itertuples() if r.usable}
    cnt = rep[rep.usable].groupby("lake").size()
    good_lakes = set(cnt[cnt >= 3].index)
    dropped = sorted({(r.lake, r.year) for r in rep.itertuples() if not r.usable or r.lake not in good_lakes})
    (R / "reports").mkdir(exist_ok=True)
    pd.DataFrame(dropped, columns=["lake", "year"]).to_csv(R / "reports" / "package_dropped_lakeyears.csv", index=False)
    print(f"dropped {len(dropped)} lake-years for unusable inputs / lakes with <3 usable years; lakes lost: {sorted(set(rep.lake) - good_lakes)}")

    entries, pos_all, neg_pool = [], 0, []
    for d in sorted(PROC.glob("GLO_*/20*/mask.npy")):
        ydir = d.parent
        gid, y = ydir.parent.name, ydir.name
        if (gid, y) not in ok_years or gid not in good_lakes:
            continue
        idx = json.loads((ydir / "tile_index.json").read_text())
        m = np.load(d)
        n = len(idx["tiles"])
        labels = np.zeros((n, 256, 256), np.uint8)
        for i, t in enumerate(idx["tiles"]):
            sub = m[t["row_offset"]:t["row_offset"] + 256, t["col_offset"]:t["col_offset"] + 256]
            labels[i, :sub.shape[0], :sub.shape[1]] = sub
        has = labels.reshape(n, -1).any(1)
        entries.append((gid, y, ydir, idx, labels, has))
        pos_all += int(has.sum())
        neg_pool += [(gid, y, int(i)) for i in np.flatnonzero(~has)]
    n_neg = int(round(pos_all * NEG_TARGET / (1 - NEG_TARGET)))
    rng.shuffle(neg_pool)
    keep_neg = set(neg_pool[:n_neg])

    rows = []
    for gid, y, ydir, idx, labels, has in entries:
        sel = [i for i in range(len(has)) if has[i] or (gid, y, i) in keep_neg]
        if not sel:
            continue
        z = np.load(ydir / "tiles.npz")
        qa = json.loads((ydir / "mask_qa.json").read_text())
        np.savez_compressed(OUT / "tiles" / f"{gid}_{y}.npz", x=z["x"][sel], elev=z["elev"][sel], valid=z["valid"][sel], y=labels[sel],
                            offsets=np.array([[idx["tiles"][i]["row_offset"], idx["tiles"][i]["col_offset"]] for i in sel]))
        rows.append({"lake": gid, "year": int(y), "tiles": len(sel), "pos_tiles": int(has[sel].sum()), "neg_tiles": int((~has[sel]).sum()),
                     "water_frac": float(labels[sel].mean()), "label_source": qa["label_source"], "uncertainty_rel": qa["uncertainty_rel"],
                     "cohort_id": meta.loc[gid, "cohort_id"], "size": meta.loc[gid, "size"], "elev_band": meta.loc[gid, "elev_band"],
                     "connectivity": meta.loc[gid, "conn"]})
    man = pd.DataFrame(rows)
    man.to_csv(OUT / "manifest.csv", index=False)

    # ---- lake-level splits ----
    lakes = man.groupby("lake").first()
    cohort_test = [g for g in lakes.index if lakes.loc[g, "cohort_id"] in TEST_COHORT]
    others = [g for g in lakes.index if lakes.loc[g, "cohort_id"] == ""]
    order = sorted(others, key=lambda g: (lakes.loc[g, "size"], lakes.loc[g, "elev_band"], rng.random()))
    test_other = order[:: max(1, len(order) // N_TEST_OTHER)][:N_TEST_OTHER]
    test = cohort_test + test_other
    pool = [g for g in lakes.index if g not in test]
    pool_sorted = sorted(pool, key=lambda g: (lakes.loc[g, "size"], lakes.loc[g, "elev_band"], rng.random()))
    folds = [[] for _ in range(5)]
    for i, g in enumerate(pool_sorted):
        folds[i % 5].append(g)
    assert not set(test) & set(pool) and all(not set(f) & set(test) for f in folds)
    assert len({g for f in folds for g in f}) == len(pool) == sum(len(f) for f in folds)
    (OUT / "splits.json").write_text(json.dumps({
        "test_lakes": test, "train_pool": pool, "cv_folds": folds, "lolo": pool, "seed": a.seed,
        "note": "lake-level; no lake appears in more than one of test / any CV fold; tiles of a lake are never split."}, indent=1))

    shutil.copy(R / "DATA_PROVENANCE.md", OUT / "DATA_PROVENANCE.md")
    for src in (R / "src").rglob("*.py"):
        dst = OUT / "code" / src.relative_to(R)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
    (OUT / "code" / "configs").mkdir(parents=True, exist_ok=True)
    for c in ("pipeline.yaml", "lakes.yaml"):
        shutil.copy(R / "configs" / c, OUT / "code" / "configs" / c)
    (OUT / "dataset-metadata.json").write_text(json.dumps({
        "title": "GlacierGuards GLOID lake segmentation tiles", "id": "YOUR_KAGGLE_USERNAME/glacierguards-gloid",
        "licenses": [{"name": "CC-BY-4.0"}],
        "description": "10-channel (VV,VH,NDWI_blue,NDWI_green,NDVI,NDSI,NDGI,elevation,slope,aspect) 256x256 tiles with weak lake masks "
                       "rasterized from GLO/GLOID polygons (doi:10.5281/zenodo.19370146, CC BY 4.0). See DATA_PROVENANCE.md."}, indent=2))

    zp = R / "glacierguards_kaggle.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:      # npz already compressed
        for f in OUT.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(OUT))
    gb = zp.stat().st_size / 1e9
    print(f"lakes {len(lakes)} (test {len(test)}, pool {len(pool)}), lake-years {len(man)}, tiles {int(man.tiles.sum())} "
          f"(pos {int(man.pos_tiles.sum())}, neg {int(man.neg_tiles.sum())} = {man.neg_tiles.sum() / man.tiles.sum():.0%}); zip {gb:.2f} GB")
    print("test lakes:", test)
    if gb > a.max_gb:
        raise SystemExit(f"zip {gb:.2f} GB exceeds {a.max_gb} GB")


if __name__ == "__main__":
    main()
