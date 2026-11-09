"""Statistics, figures and SUMMARY.md for the independent test (reads reports/independent_eval/per_lake.csv).

  * mean target IoU per model with 95 % bootstrap CI over lakes (10,000 resamples), overall and by size_class / region_group / dam_type
  * paired Wilcoxon signed-rank: U-Net vs each other model on per-lake IoU
  * figures: IoU vs reference area (log x), boxplot by size class; overlays of the 5 worst U-Net lakes -> reports/qa/independent/
  * acceptance checks 4 and 6 (all primary lakes present or replaced; reference areas within 5 %)

    python scripts/independent_stats.py
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
OUT, QA = ROOT / "reports" / "independent_eval", ROOT / "reports" / "qa" / "independent"
N_BOOT, RNG = 10_000, np.random.default_rng(0)


def boot_ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, np.nan
    m = RNG.choice(x, (N_BOOT, len(x))).mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def table(df, by=None):
    rows = []
    for mdl, g in df.groupby("model"):
        groups = [("all", g)] if by is None else [(k, s) for k, s in g.groupby(by)]
        for k, s in groups:
            lo, hi = boot_ci(s.iou_target)
            rows.append({"model": mdl, "group": k, "n": int(s.iou_target.notna().sum()), "mean_iou": s.iou_target.mean(),
                         "ci_lo": lo, "ci_hi": hi, "median_iou": s.iou_target.median(), "mean_f1": s.f1_target.mean(),
                         "median_abs_area_err_pct": s.area_err_pct.abs().median()})
    return pd.DataFrame(rows)


def md(df):
    return df.round(3).to_markdown(index=False) if hasattr(df, "to_markdown") else df.round(3).to_string(index=False)


def main():
    pl = pd.read_csv(OUT / "per_lake.csv")
    lakes = pd.read_csv(ROOT / "configs" / "independent_test_lakes.csv")
    prim = set(lakes[lakes.role == "primary"].lake_id)
    repl_p = ROOT / "reports" / "independent_replacements.csv"
    repl = pd.read_csv(repl_p) if repl_p.exists() else pd.DataFrame(columns=["replaced", "reason"])
    absent = prim - set(pl.lake_id) - set(repl.get("replaced", []))
    print("check 4 (every primary lake scored or replaced):", "OK" if not absent else f"MISSING {sorted(absent)}")
    ref = ROOT / "reports" / "independent_eval" / "reference_check.csv"
    if ref.exists():
        rc = pd.read_csv(ref)
        bad = rc[rc.flag_over_5pct].lake_id.tolist()
        print("check 6 (reference area within 5 %):", "OK" if not bad else f"FLAGGED {bad}")
    qa_ok = pl[(pl.model == "unet")]
    fails = qa_ok[qa_ok.quality_gate_value < 0.5].lake_id.tolist()
    if fails:
        print("quality gate failures (replace per rule, not for score):", fails)

    overall = table(pl)
    overall.to_csv(OUT / "summary_overall.csv", index=False)
    parts = {k: table(pl, k) for k in ("size_class", "region_group", "dam_type")}
    for k, t in parts.items():
        t.to_csv(OUT / f"summary_by_{k}.csv", index=False)

    u = pl[pl.model == "unet"].set_index("lake_id").iou_target
    wil = []
    for mdl in sorted(set(pl.model) - {"unet"}):
        b = pl[pl.model == mdl].set_index("lake_id").iou_target
        j = pd.concat([u, b], axis=1, keys=["u", "b"]).dropna()
        d = j.u - j.b
        p = wilcoxon(d).pvalue if len(j) >= 6 and (d != 0).any() else np.nan
        wil.append({"vs": mdl, "n_lakes": len(j), "mean_diff_iou": d.mean(), "unet_wins": int((d > 0).sum()), "wilcoxon_p": p})
    wil = pd.DataFrame(wil)
    wil.to_csv(OUT / "wilcoxon_vs_unet.csv", index=False)

    # figures
    QA.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mdl, g in pl.groupby("model"):
        if mdl in ("unet", "ndwi_0.3", "rf_0.5", "sar_-15dB"):
            ax.scatter(g.area_ref_km2, g.iou_target, s=14, label=mdl, alpha=0.8)
    ax.set_xscale("log"); ax.set_xlabel("reference area (km2)"); ax.set_ylabel("IoU (target, 400 m buffer)"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "iou_vs_area.png", dpi=140); plt.close(fig)
    order = [c for c in ("XS", "S", "M", "L", "XL") if c in set(pl.size_class)]
    mods = [m for m in ("unet", "rf_0.5", "ndwi_0.3", "sar_-15dB") if m in set(pl.model)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, m in enumerate(mods):
        data = [pl[(pl.model == m) & (pl.size_class == c)].iou_target.dropna() for c in order]
        ax.boxplot(data, positions=np.arange(len(order)) * (len(mods) + 1) + i, widths=0.8, patch_artist=True,
                   boxprops=dict(facecolor=f"C{i}", alpha=.6), medianprops=dict(color="k"), showfliers=True)
    ax.set_xticks(np.arange(len(order)) * (len(mods) + 1) + (len(mods) - 1) / 2); ax.set_xticklabels(order)
    ax.legend([plt.Rectangle((0, 0), 1, 1, fc=f"C{i}", alpha=.6) for i in range(len(mods))], mods)
    ax.set_ylabel("IoU"); ax.set_xlabel("size class (2022 area)"); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "iou_by_size_class.png", dpi=140); plt.close(fig)

    worst = pl[pl.model == "unet"].nsmallest(5, "iou_target")
    for r in worst.itertuples():
        ed = ROOT / "data" / "eval" / r.lake_id
        if not (ed / "pred_unet.npy").exists():
            continue
        pred = np.load(ed / "pred_unet.npy").astype(np.float32) > 0.5
        rt, ra = np.load(ed / "ref_target.npy") > 0, np.load(ed / "ref_all.npy") > 0
        ys, xs = np.where(rt)
        pad = 60
        sl = (slice(max(ys.min() - pad, 0), ys.max() + pad), slice(max(xs.min() - pad, 0), xs.max() + pad))
        with rasterio.open(ROOT / "data" / "interim" / r.lake_id / "2022" / "stack.tif") as s:
            ndwi = s.read(4)[sl]
        rgb = np.where(ndwi > -9000, np.clip((ndwi + 1) / 2, 0, 1), 0.3)
        img = np.dstack([rgb] * 3)
        img[pred[sl] & ~ra[sl]] = [1, .2, .2]                       # FP: red
        img[~pred[sl] & rt[sl]] = [1, 1, 0]                         # FN: yellow
        img[pred[sl] & ra[sl]] = [0, .6, 1]                         # hit: blue
        fig, ax = plt.subplots(figsize=(5, 5)); ax.imshow(img); ax.axis("off")
        ax.set_title(f"{r.lake_id} IoU {r.iou_target:.2f} ({r.size_class}, {r.dam_type})\nblue hit, red false pos, yellow missed", fontsize=9)
        fig.tight_layout(); fig.savefig(QA / f"{r.lake_id}.png", dpi=110); plt.close(fig)

    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        commit = "unknown"
    lines = [f"# Independent test: summary (git commit {commit}, uncommitted changes possible)", "",
             f"Lakes scored: {pl.lake_id.nunique()}; models: {', '.join(sorted(set(pl.model)))}. Reference: Third Pole 2022 manual polygons. "
             "Checkpoint models/seg_fused_test.pt, threshold 0.5, 400 m buffer. Mean IoU with 95 % bootstrap CI over lakes.", "",
             "## Overall", md(overall), "", "## U-Net vs each baseline (paired, per-lake IoU)", md(wil), ""]
    for k, t in parts.items():
        lines += [f"## By {k}", md(t[t.model.isin(["unet"] + [m for m in mods if m != "unet"])]), ""]
    lines += ["## 5 worst U-Net lakes", md(worst[["lake_id", "size_class", "dam_type", "region_group", "area_ref_km2", "iou_target", "area_err_pct"]]), "",
              "Overlays: reports/qa/independent/. Figures: iou_vs_area.png, iou_by_size_class.png.", "",
              "## Commands", "```", "python src/ingestion/tp_fetch.py --workers 4", "python src/preprocessing/gloid_stack_tile.py --lakes-csv configs/independent_test_lakes.csv --id-col lake_id --skip-tiles",
              "python scripts/tp_to_masks.py", "python scripts/train_rf_baseline.py", "python scripts/eval_independent.py", "python scripts/independent_stats.py", "```"]
    (OUT / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(overall.round(3).to_string(index=False))
    print(wil.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
