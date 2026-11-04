"""Dataset + fixed normalization for segmentation tiles.

Tiles come from tile_and_pad.py (10,256,256 + validity sidecar); labels from
mask.npy (full-grid) cropped with tile_index.json offsets. Normalization is fixed
(not per-scene) so SAR level is preserved between dates.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
NODATA = -9999.0
OPTICAL_IDX = [2, 3, 4, 5, 6]
SAR_IDX = [0, 1]
CH_NAMES = ["VV", "VH", "NDWI_blue", "NDWI_green", "NDVI", "NDSI", "NDGI", "elevation", "slope", "aspect"]


def normalize(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """x: (10,H,W) raw; valid: (10,H,W) bool. Returns float32 with invalid -> 0."""
    x = np.where(valid & np.isfinite(x) & (x != NODATA), x, np.nan).astype(np.float32)
    out = np.empty_like(x)
    out[0:2] = np.clip((x[0:2] + 30.0) / 30.0, 0, 1)        # dB -30..0 -> 0..1
    out[2:7] = np.clip(x[2:7], -1, 1)
    out[7] = (x[7] - 5500.0) / 1000.0                        # elevation (m)
    out[8] = np.clip(x[8] / 60.0, 0, 1.5)                    # slope (deg)
    out[9] = x[9] / 360.0                                    # aspect (deg)
    return np.nan_to_num(out, nan=0.0)


def channel_subset(mode: str) -> list[int]:
    return {"fused": list(range(10)), "sar": [0, 1], "optical": OPTICAL_IDX}[mode]


def list_dates(lake: str, labeled_only=True) -> list[str]:
    out = []
    for d in sorted((PROCESSED / lake).iterdir()):
        if not (d.is_dir() and (d / "tile_index.json").exists()):
            continue
        if labeled_only:
            qa = d / "mask_qa.json"
            if not (d / "mask.npy").exists() or (qa.exists() and json.loads(qa.read_text()).get("exclude_from_training")):
                continue
        out.append(d.name)
    return out


class TileDataset(Dataset):
    def __init__(self, items: list[tuple[str, str]], channels: list[int] | None = None,
                 augment: bool = False, optical_dropout: float = 0.25, min_label_valid: float = 0.05):
        self.channels = channels or list(range(10))
        self.augment, self.optical_dropout = augment, optical_dropout
        self.samples = []
        for lake, date in items:
            ddir = PROCESSED / lake / date
            idx = json.loads((ddir / "tile_index.json").read_text())
            mask = np.load(ddir / "mask.npy")
            for t in idx["tiles"]:
                self.samples.append((ddir, t, mask))
        self.mask_cache = {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        ddir, t, mask = self.samples[i]
        raw = np.load(ddir / t["file"])
        valid = np.load(ddir / t["valid_mask_file"])
        x = normalize(raw, valid)
        r, c = t["row_offset"], t["col_offset"]
        m = np.zeros((256, 256), np.float32)
        sub = mask[r:r + 256, c:c + 256]
        m[:sub.shape[0], :sub.shape[1]] = sub
        # Loss only where the label is trustworthy: optical (NDWI_green) valid and inside the scene.
        lv = valid[3].astype(np.float32)
        if self.augment:
            if np.random.rand() < 0.5:
                x, m, lv = x[:, :, ::-1], m[:, ::-1], lv[:, ::-1]
            if np.random.rand() < 0.5:
                x, m, lv = x[:, ::-1], m[::-1], lv[::-1]
            if np.random.rand() < self.optical_dropout:  # teach the net to cope with cloudy / missing optical
                x = x.copy(); x[OPTICAL_IDX] = 0
        x = x[self.channels]
        return (torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(m))[None],
                torch.from_numpy(np.ascontiguousarray(lv))[None])


class NpzTileDataset(Dataset):
    """Tiles from the Kaggle dataset: tiles/{lake}_{year}.npz with x (n,9,256,256 f16, no elevation),
    elev (n,256,256), valid (n,10,256,256), y (n,256,256) labels. Loaded into RAM once."""

    def __init__(self, files, channels=None, augment=False, optical_dropout=0.25, sar_dropout=0.1):
        self.channels = channels or list(range(10))
        self.augment, self.optical_dropout, self.sar_dropout = augment, optical_dropout, sar_dropout
        xs, es, vs, ys, self.lake = [], [], [], [], []
        for f in files:
            z = np.load(f)
            xs.append(z["x"]); es.append(z["elev"]); vs.append(z["valid"]); ys.append(z["y"])
            self.lake += [Path(f).name.rsplit("_", 1)[0]] * len(z["y"])
        self.x, self.e = np.concatenate(xs), np.concatenate(es)
        self.v, self.y = np.concatenate(vs), np.concatenate(ys)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        raw = np.concatenate([self.x[i, :7].astype(np.float32), self.e[i][None], self.x[i, 7:].astype(np.float32)])
        valid = self.v[i]
        x = normalize(raw, valid)
        m = self.y[i].astype(np.float32)
        lv = valid.any(0).astype(np.float32)          # ignore only padding / fully-empty pixels
        if self.augment:
            if np.random.rand() < 0.5:
                x, m, lv = x[:, :, ::-1], m[:, ::-1], lv[:, ::-1]
            if np.random.rand() < 0.5:
                x, m, lv = x[:, ::-1], m[::-1], lv[::-1]
            x = x.copy()
            if np.random.rand() < self.optical_dropout:
                x[OPTICAL_IDX] = 0
            if np.random.rand() < self.sar_dropout:
                x[SAR_IDX] = 0
        x = x[self.channels]
        return (torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(m))[None],
                torch.from_numpy(np.ascontiguousarray(lv))[None])
