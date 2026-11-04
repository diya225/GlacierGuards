"""Plot observed vs predicted lake area and anomaly residual (M6-A visual check)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.utils.config_loader import get_lake_by_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lake", default="south_lhonak")
    a = ap.parse_args()
    d = ROOT / "data" / "predictions" / a.lake
    area = pd.read_csv(d / "area_series.csv", parse_dates=["date"])
    an = pd.read_csv(d / "anomaly_series.csv", parse_dates=["date"])
    burst = pd.to_datetime(get_lake_by_id(a.lake).get("burst_date"))
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(area["date"], area["area_m2"] / 1e6, "o-", label="segmented area (all dates)")
    ax[0].plot(an["date"], an["predicted_area_m2"] / 1e6, "s--", label="temporal-model forecast")
    ax[0].set_ylabel("area (km²)"); ax[0].legend()
    ax[1].bar(an["date"], an["residual_m2"] / 1e6, width=1.5, color="C3")
    ax[1].axhline(0, color="k", lw=.5); ax[1].set_ylabel("residual obs − forecast (km²)")
    for x in ax:
        if pd.notna(burst):
            x.axvline(burst, color="k", ls=":", label="burst")
    ax[0].set_title(f"{a.lake}: area and temporal anomaly (out_of_sample={bool(an['out_of_sample'].iloc[0])})")
    out = ROOT / "reports"; out.mkdir(exist_ok=True)
    fig.tight_layout(); fig.savefig(out / f"{a.lake}_anomaly.png", dpi=110)
    print(out / f"{a.lake}_anomaly.png")


if __name__ == "__main__":
    main()
