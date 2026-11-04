"""Train the U-Net lake segmenter.

Split modes: random (80/20 tiles), date (hold out whole dates; stricter, default),
lake (leave-one-lake-out via --holdout-lake). Channel subsets for ablations:
--channels fused|sar|optical.

    python src/training/train_segmentation.py --epochs 20 --split date
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.segmentation_model import CompositeLoss, binary_metrics, build_model, summarize
from src.training.seg_dataset import NpzTileDataset, TileDataset, channel_subset, list_dates
from src.utils.config_loader import load_pipeline_config


def make_split(lakes, split, holdout_lake, val_frac, seed):
    items = [(l, d) for l in lakes for d in list_dates(l)]
    rng = random.Random(seed)
    if split == "lake":
        tr = [i for i in items if i[0] != holdout_lake]
        va = [i for i in items if i[0] == holdout_lake]
    elif split == "date":
        sh = items[:]
        rng.shuffle(sh)
        k = max(1, int(round(len(sh) * val_frac)))
        va, tr = sh[:k], sh[k:]
    else:  # random tiles: same dates may appear in both
        return items, items
    return tr, va


def npz_split(root: Path, split: str, holdout_lake, fold):
    """Lake-grouped splits from splits.json. Test lakes are never in train/val unless split == 'test'."""
    sp = json.loads((root / "splits.json").read_text())
    pool = sp["train_pool"]
    if split == "lake":                        # leave-one-lake-out (within the train pool)
        val = [holdout_lake]
    elif split == "cv":                        # fixed 5-fold lake-grouped CV
        val = sp["cv_folds"][fold]
    elif split == "test":                      # final: train on whole pool, report on held-out test lakes
        return pool, sp["test_lakes"]
    else:
        raise SystemExit("with --data-root use --split lake|cv|test")
    assert not set(val) & set(sp["test_lakes"]), "test lake leaked into validation"
    return [l for l in pool if l not in val], val


def files_for(root: Path, lakes):
    return sorted(f for l in lakes for f in (root / "tiles").glob(f"{l}_*.npz"))


def evaluate(model, loader, loss_fn, dev):
    model.eval()
    c = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}
    tot = 0.0
    n = 0
    with torch.no_grad():
        for x, m, v in loader:
            x, m, v = x.to(dev), m.to(dev), v.to(dev)
            lg = model(x)
            tot += loss_fn(lg, m, v)[0].item() * len(x)
            n += len(x)
            for k, val in binary_metrics(lg, m, v).items():
                c[k] += val
    return {"loss": tot / max(n, 1), **summarize(c)}


def run(args) -> dict:
    cfg = load_pipeline_config()
    tcfg, mcfg = cfg["training"], cfg["model"]["segmentation"]
    random.seed(tcfg["seed"]); np.random.seed(tcfg["seed"]); torch.manual_seed(tcfg["seed"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    chans = channel_subset(args.channels)

    tr_items = va_items = None
    if args.data_root:
        root = Path(args.data_root)
        tr_lakes, va_lakes = npz_split(root, args.split, args.holdout_lake, args.fold)
        assert not set(tr_lakes) & set(va_lakes), "lake overlap between train and val"
        assert not any("lhonak" in l.lower() for l in tr_lakes + va_lakes), "South Lhonak must not be in training data"
        tr_ds = NpzTileDataset(files_for(root, tr_lakes), chans, augment=True)
        va_ds = NpzTileDataset(files_for(root, va_lakes), chans)
        va_items = [(l, "") for l in va_lakes]
    else:
        tr_items, va_items = make_split(args.lakes, args.split, args.holdout_lake, tcfg["val_split"], tcfg["seed"])
    if args.data_root:
        pass
    elif args.split == "random":
        full = TileDataset(tr_items, chans, augment=False)
        idx = np.random.RandomState(tcfg["seed"]).permutation(len(full))
        k = int(len(idx) * (1 - tcfg["val_split"]))
        tr_ds = torch.utils.data.Subset(TileDataset(tr_items, chans, augment=True), idx[:k].tolist())
        va_ds = torch.utils.data.Subset(full, idx[k:].tolist())
    else:
        tr_ds, va_ds = TileDataset(tr_items, chans, augment=True), TileDataset(va_items, chans)
    print(f"train {len(tr_ds)} tiles / val {len(va_ds)} tiles; val: {[d or l for l, d in va_items] if va_items else 'random tiles'}")

    bs = args.batch_size or tcfg["batch_size"]
    tl = DataLoader(tr_ds, batch_size=bs, shuffle=True, num_workers=0)
    vl = DataLoader(va_ds, batch_size=bs, num_workers=0)

    model = build_model(mcfg["encoder"], len(chans), mcfg["pretrained"] and not args.no_pretrained).to(dev)
    lc = mcfg["loss"]
    loss_fn = CompositeLoss(lc["bce_weight"], lc["dice_weight"], lc["focal_weight"], lc["focal_gamma"], lc["focal_alpha"])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr or tcfg["learning_rate"], weight_decay=tcfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    use_amp = dev == "cuda" and tcfg["mixed_precision"]
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out = ROOT / "models"
    out.mkdir(exist_ok=True)
    tag = f"_{args.holdout_lake}" if args.holdout_lake else (f"_fold{args.fold}" if args.split == "cv" else "")
    name = args.name or ("segmentation_checkpoint" if args.channels == "fused" and args.split not in ("lake", "cv", "test") else f"seg_{args.channels}_{args.split}{tag}")
    best, bad, hist = -1.0, 0, []
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        comp = {"bce": 0.0, "dice": 0.0, "focal": 0.0}
        n = 0
        for step, (x, m, v) in enumerate(tl):
            if args.max_steps and step >= args.max_steps:
                break
            x, m, v = x.to(dev), m.to(dev), v.to(dev)
            opt.zero_grad()
            with torch.autocast(dev, enabled=use_amp):
                lg = model(x)
            loss, parts = loss_fn(lg.float(), m, v)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            for k in comp:
                comp[k] += parts[k] * len(x)
            n += len(x)
        sched.step()
        ev = evaluate(model, vl, loss_fn, dev)
        rec = {"epoch": ep, **{f"train_{k}": v / n for k, v in comp.items()}, **{f"val_{k}": v for k, v in ev.items()}}
        hist.append(rec)
        print(f"ep{ep:02d} {time.time()-t0:.0f}s bce {rec['train_bce']:.3f} dice {rec['train_dice']:.3f} focal {rec['train_focal']:.3f} | val IoU {ev['iou']:.3f} F1 {ev['f1']:.3f} P {ev['precision']:.3f} R {ev['recall']:.3f}")
        if ev["iou"] > best:
            best, bad = ev["iou"], 0
            torch.save({"model": model.state_dict(), "channels": chans, "args": vars(args), "val": ev}, out / f"{name}.pt")
        else:
            bad += 1
            if bad >= tcfg["patience"]:
                break
    per_lake = {}
    if args.data_root and (out / f"{name}.pt").exists():
        model.load_state_dict(torch.load(out / f"{name}.pt", map_location=dev, weights_only=False)["model"])
        for lake in sorted(set(va_ds.lake)):
            idx = [i for i, l in enumerate(va_ds.lake) if l == lake]
            per_lake[lake] = evaluate(model, DataLoader(torch.utils.data.Subset(va_ds, idx), batch_size=bs), loss_fn, dev)
            print(f"  {lake}: IoU {per_lake[lake]['iou']:.3f} F1 {per_lake[lake]['f1']:.3f} P {per_lake[lake]['precision']:.3f} R {per_lake[lake]['recall']:.3f}")
    final = max(hist, key=lambda r: r["val_iou"])
    res = {"name": name, "channels": args.channels, "split": args.split, "holdout_lake": args.holdout_lake,
           "best_epoch": final["epoch"], **{k: final[f"val_{k}"] for k in ("iou", "f1", "precision", "recall", "accuracy")},
           "label_source": "gloid_annual polygons" if args.data_root else "auto_refined_unreviewed NDWI masks", "per_lake": per_lake, "history": hist}
    (out / f"{name}_metrics.json").write_text(json.dumps(res, indent=2))
    return res


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lakes", nargs="+", default=["south_lhonak"])
    p.add_argument("--split", choices=["random", "date", "lake", "cv", "test"], default="date")
    p.add_argument("--holdout-lake")
    p.add_argument("--channels", choices=["fused", "sar", "optical"], default="fused")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--name")
    p.add_argument("--lr", type=float)
    p.add_argument("--data-root", help="Kaggle dataset dir (tiles/*.npz + splits.json); omit for the old per-date .npy layout")
    p.add_argument("--fold", type=int, default=0, help="CV fold index for --split cv")
    p.add_argument("--no-pretrained", action="store_true", help="skip ImageNet weights (internet off)")
    p.add_argument("--max-steps", type=int, help="cap steps per epoch (smoke tests)")
    return p.parse_args()


if __name__ == "__main__":
    run(parse())
