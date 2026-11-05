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
    ap.add_argument("--per-track", action="store_true", help="exploratory: compare each scene with the previous scene of the SAME Sentinel-1 relative-orbit track (12-day repeat, same viewing geometry), removing ascending/descending/track offsets. Needs data/raw/{lake}/s1/{date}_meta.json and stacks on Sentinel-1 dates. Motivated by the quiet-lake noise, not tuned on bursts; no persistence confirmation in this mode.")
    ap.add_argument("--min-sigma", type=float, default=0.001, help="km2; below this the normal-noise estimate is degenerate (10 pixels at 10 m) and the lake is reported as not evaluable")
    a = ap.parse_args()
    d = pd.read_csv(ROOT / "data" / "predictions" / a.lake / "area_series.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    col = "lake_area_m2" if "lake_area_m2" in d else "area_m2"
    if col == "area_m2":
        print("WARNING: no lake_area_m2 column; run scripts/make_lake_roi.py first. Using whole-window area (includes other water).")
    d["area_km2"] = d[col] / 1e6
    d["diff_km2"] = d["area_km2"].diff()
    d["gap_days"] = d["date"].diff().dt.days
    suffix = ""
    if a.per_track:
        import json
        suffix = "_pertrack"
        trk = {}
        for f in (ROOT / "data" / "raw" / a.lake / "s1").glob("*_meta.json"):
            m = json.loads(f.read_text())
            trk[m["date"]] = m["properties"].get("relativeOrbitNumber_start")
        d["track"] = d["date"].dt.strftime("%Y-%m-%d").map(trk)
        assert d["track"].notna().all(), "some scenes have no Sentinel-1 track metadata (per-track mode needs stacks on Sentinel-1 dates)"
        d["diff_km2"] = d.groupby("track")["area_km2"].diff()
        d["gap_days"] = d.groupby("track")["date"].diff().dt.days
    burst = pd.Timestamp(a.burst_date)
    ok = d["gap_days"] <= a.max_gap
    pre = d[ok & (d["date"] < burst)]
    sigma = 1.4826 * np.median(np.abs(pre["diff_km2"] - pre["diff_km2"].median()))
    evaluable = bool(np.isfinite(sigma) and sigma >= a.min_sigma)
    if not evaluable:
        print(f"NOT EVALUABLE: robust sigma of pre-burst changes is {sigma:.4f} km2 (< {a.min_sigma} km2, or no comparable pairs). "
              "The series is degenerate (for example the lake is below the radar detection limit, or there is no data before the event); no detection result is reported.")
        sigma = np.nan
    d["z"] = (d["diff_km2"] - pre["diff_km2"].median()) / sigma
    d["flag"] = ok & (d["z"] < -a.k)
    d["period"] = np.where(d["date"] < burst, "pre", "post")
    # Persistence confirmation (rule fixed on 2026-11-05 BEFORE Thulagi and Gongbatongsha were looked at; first seen on Imja Tsho,
    # Lower Ngole and South Lhonak): a flag counts as confirmed when the next scene (gap <= max-gap) is still at least half of the
    # drop below the pre-drop level. Single-scene outliers recover and fail this; a drained lake stays low.
    d["flag_confirmed"] = False
    for i in d.index[d.flag if not a.per_track else d.flag & False]:
        j = i + 1
        if j in d.index and d.loc[j, "gap_days"] <= a.max_gap:
            drop = d.loc[i - 1, "area_km2"] - d.loc[i, "area_km2"]
            d.loc[i, "flag_confirmed"] = bool(d.loc[j, "area_km2"] <= d.loc[i - 1, "area_km2"] - 0.5 * drop)
    fa = d[d.flag & (d.period == "pre")]
    det = d[d.flag & (d.period == "post")]
    fa_c = d[d.flag_confirmed & (d.period == "pre")]
    det_c = d[d.flag_confirmed & (d.period == "post")]
    print(f"{a.lake}: {len(d)} scenes, {len(pre)} comparable pre-burst pairs (gap <= {a.max_gap} d)")
    print(f"robust sigma of pre-burst changes: {sigma:.3f} km2 (std {pre['diff_km2'].std():.3f}); threshold = {a.k} sigma = {a.k * sigma:.3f} km2 drop")
    for opt in (True, False):
        s = pre[pre.used_optical == opt]["diff_km2"]
        print(f"  pre-burst changes, {'optical' if opt else 'SAR-only'} scenes: n={len(s)}, std={s.std():.3f} km2")
    print(f"false alarms before the burst: {len(fa)} of {len(pre)} pairs")
    print(f"with persistence confirmation: {len(fa_c)} false alarms before the burst; " +
          (f"first confirmed detection {det_c.iloc[0]['date'].date()} ({(det_c.iloc[0]['date'] - burst).days} days after the burst date)" if len(det_c) else "no confirmed detection after the burst"))
    if len(det):
        f = det.iloc[0]
        print(f"first detection: {f['date'].date()} (drop {f['diff_km2']:.2f} km2, z={f['z']:.1f}), {(f['date'] - burst).days} days after the burst date")
    else:
        print("NO detection after the burst at this threshold")
    out = ROOT / "reports"
    if a.per_track:                       # exploratory output: table only, in its own subfolder (no figure)
        out = out / "per_track"
    out.mkdir(parents=True, exist_ok=True)
    d[["date", "area_km2", "diff_km2", "gap_days", "z", "flag", "flag_confirmed", "period", "used_optical"]].to_csv(out / f"{a.lake}_burst_detection{suffix}.csv", index=False)

    if a.per_track:
        return
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(d["date"], d["area_km2"], color="#9aa", lw=0.8)
    ax.scatter(d["date"], d["area_km2"], c=np.where(d["used_optical"], "#1f77b4", "#7f7f7f"), s=14, zorder=3)
    ax.scatter(d[d.flag]["date"], d[d.flag]["area_km2"], facecolors="none", edgecolors="r", s=90, zorder=4, label=f"drop > {a.k:g} sigma")
    has_burst = d["date"].min() <= burst <= d["date"].max() + pd.Timedelta(days=366)   # a placeholder burst date (e.g. 2100) must not stretch the axis
    if has_burst:
        ax.axvline(burst, color="k", ls=":", label="burst")
    pad = pd.Timedelta(days=45)
    ax.set_xlim(d["date"].min() - pad, d["date"].max() + pad)
    ax.grid(alpha=0.25)
    ax.set_ylabel("lake area (km$^2$)")
    ax.set_title(f"{a.lake}: lake-specific area (blue = optical used, grey = SAR-only)" + (" [flags: same-track comparison]" if a.per_track else "") + ("" if has_burst else "; no burst (quiet lake)"))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / f"{a.lake}_burst_detection{suffix}.png", dpi=130)


if __name__ == "__main__":
    main()
