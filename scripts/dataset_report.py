"""Phase 3: dataset QA report. Fails loudly (exit 1) on shape/CRS/alignment/NaN problems.
    python scripts/dataset_report.py   -> reports/dataset_report.md, reports/dataset_report.csv
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

R = Path(__file__).resolve().parents[1]
PROC, INTERIM = R / "data" / "processed", R / "data" / "interim"
CH = ["VV", "VH", "NDWI_blue", "NDWI_green", "NDVI", "NDSI", "NDGI", "elevation", "slope", "aspect"]


def main():
    lakes = pd.read_csv(R / "configs" / "training_lakes.csv").GLO_ID.tolist()
    rows, errors = [], []
    for gid in lakes:
        for ydir in sorted((PROC / gid).glob("20*")) if (PROC / gid).exists() else []:
            y = ydir.name
            mp, qp = ydir / "mask.npy", ydir / "mask_qa.json"
            if not mp.exists():
                continue
            idx = json.loads((ydir / "tile_index.json").read_text())
            H, W = idx["shape"][1:]
            m = np.load(mp)
            qa = json.loads(qp.read_text())
            if m.shape != (H, W):
                errors.append(f"{gid}/{y}: mask {m.shape} != stack {(H, W)}")
            st = json.loads((INTERIM / gid / y / "stack_meta.json").read_text())["grid"]
            if qa["crs"] != st["crs"]:
                errors.append(f"{gid}/{y}: CRS mismatch {qa['crs']} vs {st['crs']}")
            z = np.load(ydir / "tiles.npz")
            x, v = z["x"], z["valid"]
            if len(x) != len(idx["tiles"]):
                errors.append(f"{gid}/{y}: tile count mismatch")
            if not np.isfinite(x).all() or not np.isfinite(z["elev"]).all():
                errors.append(f"{gid}/{y}: NaN/inf in tiles")
            if set(np.unique(m)) - {0, 1}:
                errors.append(f"{gid}/{y}: mask not binary")
            cv = v.mean((0, 2, 3))
            rows.append({"lake": gid, "year": int(y), "tiles": len(x), "water_frac": float(m.mean()), "water_px": int(m.sum()),
                         "label_source": qa["label_source"], "uncertainty": round(qa["uncertainty_rel"], 3),
                         **{f"valid_{c}": round(float(cv[i]), 3) for i, c in enumerate(CH)}})
    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit("no labeled lake-years found")
    df.to_csv(R / "reports" / "dataset_report.csv", index=False)
    per_lake = df.groupby("lake").agg(years=("year", "count"), tiles=("tiles", "sum"), water_frac=("water_frac", "mean"))
    few = per_lake[per_lake.years < 3].index.tolist()
    miss_s1 = df[df.valid_VV < 0.5][["lake", "year"]].values.tolist()
    lines = [f"# Dataset report", f"- labeled lake-years: {len(df)} across {df.lake.nunique()} lakes; tiles: {int(df.tiles.sum())}",
             f"- water pixel fraction: mean {df.water_frac.mean():.4f}, median {df.water_frac.median():.4f} (class balance water:background ~ 1:{(1-df.water_frac.mean())/df.water_frac.mean():.0f})",
             f"- lakes with < 3 valid years: {few or 'none'}", f"- lake-years with SAR missing (VV valid < 50%): {miss_s1 or 'none'}",
             f"- lake-years with optical valid < 50%: {df[df.valid_NDWI_green < 0.5][['lake','year']].values.tolist() or 'none'}",
             f"- label sources: {df.label_source.value_counts().to_dict()}", "", "## per lake", per_lake.round(4).to_markdown(), "",
             "## errors" if errors else "## errors: none"] + [f"- {e}" for e in errors]
    (R / "reports" / "dataset_report.md").write_text("\n".join(lines))
    print("\n".join(lines[:8]))
    if errors:
        print(f"\nFAILED: {len(errors)} errors (see reports/dataset_report.md)")
        sys.exit(1)


if __name__ == "__main__":
    main()
