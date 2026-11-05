"""One table across the lakes run through burst_detection.py: noise, minimum detectable drop, false alarms, detection.
    python scripts/summarize_burst_detection.py     ->  reports/m6_multi_lake_summary.csv
Burst dates and manual outline areas are listed below; lakes that were not evaluable were deleted and are not in the table."""
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1]
LAKES = [  # id, display name, burst date (None = quiet), manual 2022 outline km2 (Third Pole inventory), series basis
    ("south_lhonak", "South Lhonak", "2023-10-04", 1.667, "Sentinel-1 dates"),
    ("imja_tsho", "Imja Tsho", None, 1.719, "Sentinel-1 dates"),
    ("thulagi", "Thulagi", None, 0.970, "Sentinel-1 dates"),
    ("thame_lower_ngole", "Lower Ngole (Thame)", "2024-08-16", 0.045, "Sentinel-1 dates"),
    ("jinwuco", "Jinwuco (SE Tibet)", "2020-06-26", 0.556, "Sentinel-1 dates"),   # manual outline = 2018 (pre-event) snapshot
    ("chamai", "Chamai (W Tibet)", "2021-10-05", 0.219, "Sentinel-1 dates"),      # manual outline = 2020 (pre-event) snapshot
]
rows = []
for lid, name, burst, manual, basis in LAKES:
    f = R / "reports" / f"{lid}_burst_detection.csv"
    if not f.exists():
        continue
    d = pd.read_csv(f, parse_dates=["date"])
    b = pd.Timestamp(burst) if burst else pd.Timestamp("2100-01-01")
    pre = d[(d.gap_days <= 15) & (d.date < b)]
    sig = 1.4826 * np.median(np.abs(pre.diff_km2 - pre.diff_km2.median()))
    med = d[d.date < b].area_km2.median()
    fa, fac = d[d.flag & (d.date < b)], d[d.flag_confirmed & (d.date < b)]
    det, detc = d[d.flag & (d.date >= b)], d[d.flag_confirmed & (d.date >= b)]
    # exploratory per-track variant (reports/per_track): each scene compared with the previous scene of the same Sentinel-1 track
    pt = {}
    g = R / "reports" / "per_track" / f"{lid}_burst_detection_pertrack.csv"
    if g.exists():
        e = pd.read_csv(g, parse_dates=["date"])
        pp = e[(e.gap_days <= 15) & (e.date < b)]
        ps = 1.4826 * np.median(np.abs(pp.diff_km2 - pp.diff_km2.median()))
        pf = e[e.flag & (e.date < b)]
        pd_ = e[e.flag & (e.date >= b)]
        pt = {"pertrack_sigma_km2": round(ps, 3), "pertrack_min_detectable_pct_of_area": round(100 * 4 * ps / med),
              "pertrack_false_alarms": f"{len(pf)} of {len(pp)}",
              "pertrack_first_detection": (str(pd_.iloc[0].date.date()) if len(pd_) else "none") if burst else ""}
    rows.append({"lake": name, "role": "burst 2023-10-04" if lid == "south_lhonak" else (f"burst {burst}" if burst else "quiet"), "series_basis": basis, "scenes": len(d),
                 "median_area_km2": round(med, 3), "manual_outline_km2": manual, "normal_change_sigma_km2": round(sig, 3),
                 "min_detectable_drop_km2_4sigma": round(4 * sig, 3), "min_detectable_pct_of_area": round(100 * 4 * sig / med),
                 "false_alarms": f"{len(fa)} of {len(pre)}", "false_alarms_confirmed": f"{len(fac)} of {len(pre)}",
                 "first_detection": str(det.iloc[0].date.date()) if burst and len(det) else ("none" if burst else ""),
                 "detection_lag_days": int((det.iloc[0].date - b).days) if burst and len(det) else "",
                 "first_confirmed_detection": str(detc.iloc[0].date.date()) if burst and len(detc) else ("none" if burst else ""), **pt})
out = pd.DataFrame(rows)
out.to_csv(R / "reports" / "m6_multi_lake_summary.csv", index=False)
print(out.drop(columns=["series_basis", "role"]).to_string(index=False))
