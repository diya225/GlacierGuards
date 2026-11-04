"""Zip everything Kaggle needs (code, configs, labeled tiles, stacks) into kaggle/glof_ews_kaggle.zip.

    python scripts/package_for_kaggle.py            # all lakes found under data/processed
Upload the zip as a Kaggle Dataset (name it glof-ews).
"""
import json, zipfile
from pathlib import Path

R = Path(__file__).resolve().parents[1]
out = R / "kaggle" / "glof_ews_kaggle.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for p in list((R / "src").rglob("*.py")) + list((R / "configs").glob("*.yaml")):
        z.write(p, p.relative_to(R))
    for lake in (R / "data" / "processed").iterdir():
        if not lake.is_dir():
            continue
        for d in lake.iterdir():
            if not d.is_dir() or not (d / "tile_index.json").exists():
                continue
            # tiles only for dates with labels; full stacks for every date (used for prediction)
            if (d / "mask.npy").exists():
                for f in list(d.glob("tile_*.npy")) + [d / "tile_index.json", d / "mask.npy", d / "mask_qa.json"]:
                    if f.exists():
                        z.write(f, f.relative_to(R))
        st = R / "data" / "interim" / lake.name / "stacked"
        for f in st.glob("*.tif"):
            z.write(f, f.relative_to(R))
print(out, round(out.stat().st_size / 1e6), "MB")
