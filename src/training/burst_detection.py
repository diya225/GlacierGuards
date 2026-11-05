"""Simple, assumption-free burst detector on the lake-specific area series (the baseline any forecaster must beat).

Rule: a scene is flagged when the area drops, relative to the previous scene (gap <= --max-gap days), by more than
k x sigma, where sigma is the robust (MAD) spread of scene-to-scene changes measured on PRE-burst data only.

    python src/training/burst_detection.py --lake south_lhonak --burst-date 2023-10-04
Outputs: reports/{lake}_burst_detection.csv, reports/{lake}_burst_detection.png
Reports false alarms before the burst, the first detection and its lag after the burst, separately for the
whole series and for optical vs SAR-only scenes (SAR-only is noisier).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lake", default="south_lhonak")
    ap.add_argument("--burst-date", default="2023-10-04")
    ap.add_argument("--max-gap", type=int, default=15, help="only compare scenes at most this many days apart (excludes winter gaps)")
    ap.add_argument("--k", type=float, default=4.0, help="threshold in robust sigmas")
    a = ap.parse_args()
    d = pd.read_csv(ROOT / "data" / "predictions" / a.lake / "area_series.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    col = "lake_area_m2" if "lake_area_m2" in d else "area_m2"
    if col == "area_m2":
        print("WARNING: no lake_area_m2 column; run scripts/make_lake_roi.py first. Using whole-window area (includes other water).")
    d["area_km2"] = d[col] / 1e6
    d["diff_km2"] = d["area_km2"].diff()
    d["gap_days"] = d["date"].diff().dt.days
    burst = pd.Timestamp(a.burst_date)
    ok = d["gap_days"] <= a.max_gap
    pre = d[ok & (d["date"] < burst)]
    sigma = 1.4826 * np.median(np.abs(pre["diff_km2"] - pre["diff_km2"].median()))
    d["z"] = (d["diff_km2"] - pre["diff_km2"].median()) / sigma
    d["flag"] = ok & (d["z"] < -a.k)
    d["period"] = np.where(d["date"] < burst, "pre", "post")
    fa = d[d.flag & (d.period == "pre")]
    det = d[d.flag & (d.period == "post")]
    print(f"{a.lake}: {len(d)} scenes, {len(pre)} comparable pre-burst pairs (gap <= {a.max_gap} d)")
    print(f"robust sigma of pre-burst changes: {sigma:.3f} km2 (std {pre['diff_km2'].std():.3f}); threshold = {a.k} sigma = {a.k * sigma:.3f} km2 drop")
    for opt in (True, False):
        s = pre[pre.used_optical == opt]["diff_km2"]
        print(f"  pre-burst changes, {'optical' if opt else 'SAR-only'} scenes: n={len(s)}, std={s.std():.3f} km2")
    print(f"false alarms before the burst: {len(fa)} of {len(pre)} pairs")
    if len(det):
        f = det.iloc[0]
        print(f"first detection: {f['date'].date()} (drop {f['diff_km2']:.2f} km2, z={f['z']:.1f}), {(f['date'] - burst).days} days after the burst date")
    else:
        print("NO detection after the burst at this threshold")
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    d[["date", "area_km2", "diff_km2", "gap_days", "z", "flag", "period", "used_optical"]].to_csv(out / f"{a.lake}_burst_detection.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(d["date"], d["area_km2"], color="#9aa", lw=0.8)
    ax.scatter(d["date"], d["area_km2"], c=np.where(d["used_optical"], "#1f77b4", "#7f7f7f"), s=14, zorder=3)
    ax.scatter(d[d.flag]["date"], d[d.flag]["area_km2"], facecolors="none", edgecolors="r", s=90, zorder=4, label=f"drop > {a.k:g} sigma")
    ax.axvline(burst, color="k", ls=":", label="burst")
    ax.set_ylabel("lake area (km$^2$)")
    ax.set_title(f"{a.lake}: lake-specific area (blue = optical used, grey = SAR-only)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / f"{a.lake}_burst_detection.png", dpi=130)


if __name__ == "__main__":
    main()
