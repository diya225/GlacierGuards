"""Score the frozen U-Net and the baselines on the independent test lakes (Third Pole 2022 manual polygons).

FIXED BEFORE SCORING (do not change after looking at results): checkpoint models/seg_fused_test.pt, threshold 0.5,
NDWI_green > 0.3, per-lake Otsu, VV < -15 dB, RF thresholds 0.5 / 0.75, 400 m buffer, metrics below.

Primary score = target lake: prediction and reference restricted to the target polygon buffered by 400 m.
Pixels of OTHER inventory lakes inside that buffer are excluded from the buffer region (they are neither hit nor miss).
Secondary score = whole window against ref_all (all inventory lakes).
Invalid pixels are not masked: a baseline that cannot see a pixel predicts "not water" there.

    python scripts/eval_independent.py                       # primary lakes that have a stack + reference
    python scripts/eval_independent.py --lakes TP_3630 --no-rf
Output: reports/independent_eval/per_lake.csv ; predictions of the U-Net in data/eval/{lake}/pred_unet.npy
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import distance_transform_edt
from skimage.filters import threshold_otsu

from src.training.predict_masks import load_model, predict_scene
from src.training.seg_dataset import NODATA, OPTICAL_IDX, normalize

EVAL, INTERIM = ROOT / "data" / "eval", ROOT / "data" / "interim"
OUT = ROOT / "reports" / "independent_eval"
CKPT = ROOT / "models" / "seg_fused_test.pt"
BUFFER_M, THR = 400, 0.5
GLOID_S2 = ROOT / "data" / "external" / "gloid" / "S2_20172024_NTB_GLOID_v1.02.gpkg"


def metrics(pred, ref, region):
    p, r = pred[region].astype(bool), ref[region].astype(bool)
    tp, fp, fn = int((p & r).sum()), int((p & ~r).sum()), int((~p & r).sum())
    iou = tp / max(tp + fp + fn, 1)
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"iou": iou, "f1": 2 * tp / max(2 * tp + fp + fn, 1), "precision": prec, "recall": rec, "n_pred": int(p.sum()), "n_ref": int(r.sum())}


def otsu_pred(band, valid):
    if valid.sum() < 1000 or np.ptp(band[valid]) < 1e-6:
        return None
    return valid & (band > threshold_otsu(band[valid])), float(threshold_otsu(band[valid]))


def check_checkpoint():
    m = json.loads(CKPT.with_name(CKPT.stem + "_metrics.json").read_text())
    assert CKPT.name == "seg_fused_test.pt" and m["split"] == "test" and m["channels"] == "fused", "wrong checkpoint"
    print("checkpoint OK:", CKPT.name, "split", m["split"], "(pool lakes only)")


def check_distance(lakes):
    tr = pd.read_csv(ROOT / "configs" / "training_lakes.csv")
    lat0 = np.radians(lakes.lat.values)[:, None]
    dx = (lakes.lon.values[:, None] - tr.lon.values[None]) * 111.32 * np.cos(lat0)
    dy = (lakes.lat.values[:, None] - tr.lat.values[None]) * 110.57
    d = np.sqrt(dx ** 2 + dy ** 2).min(1)
    bad = lakes.lake_id[d < 3.0].tolist()
    assert not bad, f"independent lakes within 3 km (centroid distance) of a training lake: {bad}"
    print(f"distance check OK: nearest training lake to any independent lake is {d.min():.1f} km")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roles", nargs="+", default=["primary"])
    ap.add_argument("--lakes", nargs="+")
    ap.add_argument("--no-rf", action="store_true")
    ap.add_argument("--no-gloid", action="store_true")
    ap.add_argument("--out", default=str(OUT / "per_lake.csv"))
    a = ap.parse_args()
    check_checkpoint()
    lakes = pd.read_csv(ROOT / "configs" / "independent_test_lakes.csv")
    check_distance(lakes)
    lakes = lakes[lakes.lake_id.isin(a.lakes)] if a.lakes else lakes[lakes.role.isin(a.roles)]
    model, chans = load_model(CKPT)
    rf = None
    rf_path = ROOT / "models" / "rf_baseline.joblib"
    if not a.no_rf and rf_path.exists():
        import joblib
        rf = joblib.load(rf_path)["model"]
    elif not a.no_rf:
        print("NOTE: models/rf_baseline.joblib missing -> RF rows skipped (run scripts/train_rf_baseline.py)")
    gloid = None
    if not a.no_gloid and GLOID_S2.exists():
        import geopandas as gpd
        gloid = gpd.read_file(GLOID_S2)
        gloid = gloid[gloid.AREA_YEAR == 2022]
    rows, missing = [], []
    for r in lakes.itertuples():
        sp, ed = INTERIM / r.lake_id / "2022" / "stack.tif", EVAL / r.lake_id
        if not sp.exists() or not (ed / "ref_target.npy").exists():
            missing.append(r.lake_id)
            continue
        meta = json.loads((INTERIM / r.lake_id / "2022" / "stack_meta.json").read_text())
        gate = max(meta.get("vv_valid_fraction", 0), meta.get("ndwi_green_valid_fraction", 0))
        with rasterio.open(sp) as src:
            stack = src.read().astype(np.float32)
            px = abs(src.transform.a)
        ref_all, ref_t = np.load(ed / "ref_all.npy").astype(bool), np.load(ed / "ref_target.npy").astype(bool)
        valid = np.isfinite(stack) & (stack != NODATA)
        region = (distance_transform_edt(~ref_t) * px <= BUFFER_M) & ~(ref_all & ~ref_t)
        whole = np.ones_like(ref_t)
        area_ref = float(ref_t.sum() * px * px / 1e6)
        p_unet, cov, used_opt = predict_scene(model, chans, stack)
        np.save(ed / "pred_unet.npy", p_unet.astype(np.float16))
        preds = {"unet": p_unet > THR}
        ndwi_ok = valid[3]
        preds["ndwi_0.3"] = ndwi_ok & (stack[3] > 0.3)
        o = otsu_pred(stack[3], ndwi_ok)
        preds["ndwi_otsu"] = o[0] if o else None
        vv_ok = valid[0]
        preds["sar_-15dB"] = vv_ok & (stack[0] < -15.0)
        o = otsu_pred(-stack[0], vv_ok)             # low backscatter = water -> Otsu on -VV, pixels above threshold
        preds["sar_otsu"] = o[0] if o else None
        if rf is not None:
            x = normalize(stack, valid)
            if cov < 0.8:
                x[OPTICAL_IDX] = 0
            pr = rf.predict_proba(x.reshape(10, -1).T)[:, 1].reshape(ref_t.shape)
            preds["rf_0.5"], preds["rf_0.75"] = pr > 0.5, pr > 0.75
        if gloid is not None and isinstance(r.gloid_id_2022, str) and r.gloid_id_2022:
            from rasterio.features import rasterize
            from rasterio.transform import from_origin
            from shapely.geometry import box
            gr = meta["grid"]
            H, W = ref_t.shape
            win = box(gr["x0"], gr["y_top"] - H * px, gr["x0"] + W * px, gr["y_top"])
            g = gloid.to_crs(gr["crs"])
            g = g[g.intersects(win)]
            preds["gloid_2022"] = rasterize([(x, 1) for x in g.geometry], out_shape=(H, W), transform=from_origin(gr["x0"], gr["y_top"], px, px),
                                            fill=0, dtype="uint8").astype(bool) if len(g) else np.zeros_like(ref_t)
        for name, pred in preds.items():
            row = {"lake_id": r.lake_id, "tp_id": r.tp_id, "region_group": r.region_group, "size_class": r.size_class,
                   "dam_type": r.dam_type, "model": name, "optical_valid_fraction": round(cov, 4), "used_optical": bool(used_opt),
                   "quality_gate_value": gate, "area_ref_km2": round(area_ref, 4)}
            if pred is None:
                row.update(iou_target=np.nan)
            else:
                m, w = metrics(pred, ref_t, region), metrics(pred, ref_all, whole)
                a_pred = m["n_pred"] * px * px / 1e6
                row.update(iou_target=m["iou"], f1_target=m["f1"], precision_target=m["precision"], recall_target=m["recall"],
                           area_pred_km2=round(a_pred, 4), area_err_pct=100 * (a_pred - area_ref) / area_ref,
                           iou_window=w["iou"], f1_window=w["f1"])
            rows.append(row)
        print(r.lake_id, "unet IoU %.3f" % rows[-len(preds)]["iou_target"], "| gate %.2f" % gate, flush=True)
    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {df.lake_id.nunique()} lakes x {df.model.nunique()} models")
    if missing:
        print("NOT SCORED (no stack/reference yet):", missing)
    low = df[(df.model == "unet") & (df.quality_gate_value < 0.5)].lake_id.tolist()
    if low:
        print("FAILS quality gate (>=0.5 valid VV or NDWI) -> apply replacement rule, do not drop for score:", low)


if __name__ == "__main__":
    main()
