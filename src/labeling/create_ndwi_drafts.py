"""Create review-only lake mask drafts from stacked green-NIR NDWI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio


ROOT = Path(__file__).resolve().parents[2]
STACK_DIR = ROOT / "data" / "interim" / "south_lhonak" / "stacked"
OUT_DIR = ROOT / "data" / "processed" / "south_lhonak"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lake", default="south_lhonak")
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--min-optical-valid", type=float, default=0.8)
    args = parser.parse_args()

    stack_dir = ROOT / "data" / "interim" / args.lake / "stacked"
    out_root = ROOT / "data" / "processed" / args.lake
    out_root.mkdir(parents=True, exist_ok=True)
    summary = []

    for stack_path in sorted(stack_dir.glob("*.tif")):
        with rasterio.open(stack_path) as src:
            stack = src.read()
            nodata = src.nodata if src.nodata is not None else -9999.0
            # Channels 2–6 are the five optical indices; channel 3 is green-NIR NDWI.
            optical_valid = np.all(np.isfinite(stack[2:7]) & (stack[2:7] != nodata), axis=0)
            coverage = float(optical_valid.mean())
            if coverage < args.min_optical_valid:
                continue
            draft = (optical_valid & (stack[3] > args.threshold)).astype(np.uint8)
            draft_geotiff = np.where(optical_valid, draft, 255).astype(np.uint8)
            date_dir = out_root / stack_path.stem
            date_dir.mkdir(parents=True, exist_ok=True)
            mask_path = date_dir / "mask_draft_ndwi_green.npy"
            np.save(mask_path, draft)
            with rasterio.open(stack_path) as src:
                profile = src.profile.copy()
                profile.update(count=1, dtype="uint8", nodata=255, compress="lzw")
                with rasterio.open(date_dir / "mask_draft_ndwi_green.tif", "w", **profile) as dst:
                    dst.write(draft_geotiff, 1)
                    dst.set_band_description(1, "NDWI_green threshold draft: 1=water, 0=other, 255=invalid")
                    dst.update_tags(status="DRAFT_REQUIRES_QGIS_REVIEW", threshold=args.threshold)
            lake_pixels = int(draft.sum())
            metadata = {
                "lake_id": args.lake,
                "date": stack_path.stem,
                "status": "DRAFT_REQUIRES_QGIS_REVIEW",
                "source_stack": str(stack_path),
                "index": "NDWI_green = (B3 - B8) / (B3 + B8)",
                "threshold": args.threshold,
                "optical_valid_fraction": round(coverage, 4),
                "lake_pixels": lake_pixels,
                "lake_fraction_of_full_grid": round(lake_pixels / draft.size, 4),
                "mask_file": str(mask_path),
            }
            (date_dir / "mask_draft_ndwi_green.json").write_text(json.dumps(metadata, indent=2))
            summary.append(metadata)

    (out_root / "ndwi_draft_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"draft_count": len(summary), "dates": [x["date"] for x in summary]}, indent=2))


if __name__ == "__main__":
    main()
