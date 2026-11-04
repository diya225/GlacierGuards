"""Phase 3: visual QA without QGIS. 24 random labeled tiles + the 6 lowest S1-vs-S2 IoU lake-years -> reports/qa/."""
from __future__ import annotations
import json, random
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

R = Path(__file__).resolve().parents[1]
PROC = R / "data" / "processed"
OUT = R / "reports" / "qa"


def rgb(x):  # x: (9,H,W) f16 (no elevation): channels 2,3,6 = NDWI_blue, NDWI_green, NDGI
    return np.clip(np.stack([x[2], x[3], x[6]], -1).astype(np.float32), -1, 1) * 0.5 + 0.5


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    random.seed(0)
    items = [(p.parent.parent.name, p.parent.name) for p in PROC.glob("GLO_*/*/mask.npy")]
    pos = []
    for gid, y in items:
        idx = json.loads((PROC / gid / y / "tile_index.json").read_text())
        for i, t in enumerate(idx["tiles"]):
            pos.append((gid, y, i, t["row_offset"], t["col_offset"]))
    pick = random.sample(pos, min(24, len(pos)))
    fig, ax = plt.subplots(4, 6, figsize=(24, 16))
    for a, (gid, y, i, r, c) in zip(ax.flat, pick):
        z = np.load(PROC / gid / y / "tiles.npz")
        m = np.zeros((256, 256), np.uint8)
        sub = np.load(PROC / gid / y / "mask.npy")[r:r + 256, c:c + 256]
        m[:sub.shape[0], :sub.shape[1]] = sub
        a.imshow(rgb(z["x"][i])); a.contour(m, [.5], colors="r", linewidths=.8); a.set_title(f"{gid[4:16]} {y} t{i}", fontsize=8); a.axis("off")
    fig.tight_layout(); fig.savefig(OUT / "random_tiles.png", dpi=60); plt.close(fig)

    mf = pd.read_csv(R / "reports" / "mask_manifest.csv")
    low = mf[mf.iou_s1_s2.notna()].sort_values("iou_s1_s2").head(6)
    fig, ax = plt.subplots(1, max(len(low), 1), figsize=(5 * max(len(low), 1), 5), squeeze=False)
    for a, (_, r) in zip(ax[0], low.iterrows()):
        d = PROC / r.lake / str(r.year)
        z = np.load(d / "masks_s1s2.npz")
        import rasterio
        with rasterio.open(R / "data" / "interim" / r.lake / str(r.year) / "stack.tif") as s:
            st = s.read([3, 4, 7])
        a.imshow(np.clip(np.moveaxis(st, 0, -1), -1, 1) * .5 + .5)
        a.contour(z["s2"], [.5], colors="r", linewidths=.8); a.contour(z["s1"], [.5], colors="c", linewidths=.8)
        a.set_title(f"{r.lake[4:16]} {r.year}\nIoU {r.iou_s1_s2} ({r.status})", fontsize=8); a.axis("off")
    fig.tight_layout(); fig.savefig(OUT / "lowest_iou_s1_vs_s2.png", dpi=60)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
