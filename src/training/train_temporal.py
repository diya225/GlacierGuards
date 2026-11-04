"""Train the shared temporal forecaster on predicted lake-mask sequences and emit anomaly residuals.

One model is shared across all lakes that have data/predictions/{lake}/ (currently South Lhonak only).
Sliding window of N=pipeline.yaml model.temporal.sequence_length past masks; shorter histories are
left-padded with the earliest frame. Irregular revisit is handled by feeding day-offsets as inputs.

    python src/training/train_temporal.py --loo     # out-of-sample residuals + final checkpoint
Outputs: models/temporal_checkpoint.pt, data/predictions/{lake}/anomaly_series.csv
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.models.temporal_model import build_temporal
from src.utils.config_loader import load_pipeline_config

RES = 64          # working resolution (lake AOI downsampled by area averaging)
MIN_HISTORY = 2


def load_series(lake: str):
    d = ROOT / "data" / "predictions" / lake
    csv = pd.read_csv(d / "area_series.csv")
    frames, px = [], None
    for dt in csv["date"]:
        p = torch.from_numpy(np.load(d / f"{dt}.npy").astype(np.float32))[None, None]
        frames.append(F.adaptive_avg_pool2d(p, RES)[0])
    # pixel area (m^2) of one working-res cell, from the 10 m grid
    h, w = np.load(d / f"{csv['date'][0]}.npy").shape
    cell_m2 = (h * 100.0 / RES) * (w / RES)
    days = [date.fromisoformat(x).toordinal() for x in csv["date"]]
    return csv, torch.stack(frames), np.array(days), cell_m2


def build_samples(frames, days, N, lake_id):
    S = []
    for i in range(MIN_HISTORY, len(frames)):
        idx = list(range(max(0, i - N), i))
        idx = [idx[0]] * (N - len(idx)) + idx
        S.append({"lake": lake_id, "i": i, "x": frames[idx], "dt": torch.tensor(days[i] - days[idx], dtype=torch.float32), "y": frames[i]})
    return S


def batch(samples):
    return (torch.stack([s["x"] for s in samples]), torch.stack([s["dt"] for s in samples]), torch.stack([s["y"] for s in samples]))


def fit(samples, kind, hidden, layers, epochs, lr, seed=0):
    torch.manual_seed(seed)
    m = build_temporal(kind, hidden, layers)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    x, dt, y = batch(samples)
    for _ in range(epochs):
        perm = torch.randperm(len(x))
        for b in perm.split(8):
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(m(x[b], dt[b]), y[b])
            loss.backward()
            opt.step()
    return m


@torch.no_grad()
def predict(m, s):
    m.eval()
    return torch.sigmoid(m(s["x"][None], s["dt"][None]))[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lakes", nargs="+", default=["south_lhonak"])
    ap.add_argument("--kind", choices=["conv_lstm", "attention"])
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--train-before", help="YYYY-MM-DD: fit only on targets dated before this (e.g. burst date); later targets are scored out-of-sample")
    ap.add_argument("--cv-year", action="store_true", help="blocked CV: score each season with a model that never saw that season (cheap, out-of-sample)")
    ap.add_argument("--loo", action="store_true", help="leave-one-target-out residuals (out-of-sample)")
    a = ap.parse_args()
    cfg = load_pipeline_config()["model"]["temporal"]
    kind = a.kind or cfg["type"]
    N, layers = cfg["sequence_length"], min(cfg["num_layers"], 2)

    data = {}
    samples = []
    for lake in a.lakes:
        csv, frames, days, cell = load_series(lake)
        data[lake] = (csv, cell)
        samples += build_samples(frames, days, N, lake)
    print(f"{len(samples)} sequences, N={N}, model={kind}, hidden={a.hidden}")

    def sdate(t):
        return data[t["lake"]][0]["date"][t["i"]]
    train = [t for t in samples if not a.train_before or sdate(t) < a.train_before]
    print(f"fit on {len(train)}/{len(samples)} sequences" + (f" (targets before {a.train_before})" if a.train_before else ""))
    final = fit(train, kind, a.hidden, layers, a.epochs, a.lr)
    (ROOT / "models").mkdir(exist_ok=True)
    torch.save({"model": final.state_dict(), "kind": kind, "hidden": a.hidden, "layers": layers, "N": N, "res": RES},
               ROOT / "models" / "temporal_checkpoint.pt")

    cv_models = {}
    for lake, (csv, cell) in data.items():
        rows = []
        for s in [s for s in samples if s["lake"] == lake]:
            if a.cv_year:
                yr = csv["date"][s["i"]][:4]
                model = cv_models.setdefault(yr, fit([t for t in train if sdate(t)[:4] != yr], kind, a.hidden, layers, a.epochs, a.lr))
            elif a.loo and any(t is s for t in train):
                model = fit([t for t in train if t is not s], kind, a.hidden, layers, a.epochs, a.lr)
            else:
                model = final
            pred = predict(model, s)
            obs_a = float(s["y"].sum()) * cell
            pred_a = float(pred.sum()) * cell
            persist = float(s["x"][-1].sum()) * cell
            rows.append({"date": csv["date"][s["i"]], "observed_area_m2": obs_a, "predicted_area_m2": pred_a,
                         "persistence_area_m2": persist, "residual_m2": obs_a - pred_a,
                         "pixel_mae": float((pred - s["y"]).abs().mean()), "out_of_sample": bool(a.loo or a.cv_year or (a.train_before and csv["date"][s["i"]] >= a.train_before))})
        df = pd.DataFrame(rows)
        mad = np.median(np.abs(df["residual_m2"] - df["residual_m2"].median())) * 1.4826
        df["residual_z"] = (df["residual_m2"] - df["residual_m2"].median()) / max(mad, 1.0)
        out = ROOT / "data" / "predictions" / lake / "anomaly_series.csv"
        df.to_csv(out, index=False)
        print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
