"""Milestone 6 Kaggle package: code + South Lhonak predictions (about 70 MB) -> glacierguards_temporal.zip.

Layout inside the zip mirrors the repo so the scripts run unchanged:
  src/**/*.py, configs/*.yaml, data/predictions/south_lhonak/{*.npy, area_series.csv, roi.npy}
Upload the zip as a Kaggle dataset named  glacierguards-temporal  and run kaggle/kaggle_temporal.ipynb.
No imagery, labels or model weights are included.

    python scripts/package_temporal_for_kaggle.py
"""
import zipfile
from pathlib import Path

R = Path(__file__).resolve().parents[1]
LAKE = "south_lhonak"
pred = R / "data" / "predictions" / LAKE
assert (pred / "roi.npy").exists() and "lake_area_m2" in (pred / "area_series.csv").read_text().splitlines()[0], \
    "run scripts/make_lake_roi.py first (needs roi.npy and the lake_area_m2 column)"
out = R / "glacierguards_temporal.zip"
n = 0
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in (R / "src").rglob("*.py"):
        z.write(f, f.relative_to(R)); n += 1
    for f in (R / "configs").glob("*.yaml"):
        z.write(f, f.relative_to(R)); n += 1
    for f in pred.glob("*"):
        if f.suffix in {".npy", ".csv"} and "smoke" not in f.name and f.name != "anomaly_series.csv":   # drop the stale anomaly file
            z.write(f, f.relative_to(R)); n += 1
print(f"{n} files, {out.stat().st_size / 1e6:.0f} MB -> {out}")
