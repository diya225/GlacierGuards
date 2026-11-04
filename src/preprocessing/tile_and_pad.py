"""
tile_and_pad.py — Tile stacked 10-channel tensors into 256×256 patches.

Reads the co-registered stacked GeoTIFFs from data/interim/{lake}/stacked/
and tiles them into 256×256 NumPy arrays saved to data/processed/{lake}/{date}/.
Boundary tiles are zero-padded to exactly 256×256.

Usage
-----
    python src/preprocessing/tile_and_pad.py --lake south_lhonak
    python src/preprocessing/tile_and_pad.py --lake all

Output
------
    data/processed/{lake_id}/{YYYY-MM-DD}/tile_{row:03d}_{col:03d}.npy
        shape: (10, 256, 256), float32

    data/processed/{lake_id}/{YYYY-MM-DD}/tile_index.json
        Maps tile IDs → pixel offsets in the original full image.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import rasterio
from tqdm import tqdm

from src.utils.config_loader import get_lake_by_id, get_active_lakes, load_pipeline_config

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

TILE_SIZE = 256       # Default tile dimension in pixels
CHANNEL_COUNT = 10    # Must match the stacked tensors
NODATA_VALUE = -9999.0


# ─────────────────────────────────────────────────────────────────────────────
# Tiling logic
# ─────────────────────────────────────────────────────────────────────────────

def tile_image(
    image: np.ndarray,
    tile_size: int = TILE_SIZE,
    overlap: int = 0,
) -> list[tuple[np.ndarray, int, int]]:
    """
    Tile a (C, H, W) image into non-overlapping (or overlapping) 256×256 patches.
    Boundary patches are zero-padded to the full tile_size.

    Args:
        image:     Float32 array of shape (C, H, W).
        tile_size: Output tile side length in pixels.
        overlap:   Number of overlapping pixels between adjacent tiles.

    Returns:
        List of (tile, row_offset, col_offset) tuples where:
            tile:        (C, tile_size, tile_size) float32 array
            row_offset:  top-left row in the original image (before padding)
            col_offset:  top-left column in the original image
    """
    _, H, W = image.shape
    stride = tile_size - overlap

    tiles = []
    for row_start in range(0, H, stride):
        for col_start in range(0, W, stride):
            row_end = row_start + tile_size
            col_end = col_start + tile_size

            # Crop (may be smaller than tile_size at boundaries)
            patch = image[:, row_start:row_end, col_start:col_end]
            _, ph, pw = patch.shape

            if ph == tile_size and pw == tile_size:
                tiles.append((patch, row_start, col_start))
            else:
                # Zero-pad to full tile size
                padded = np.zeros(
                    (CHANNEL_COUNT, tile_size, tile_size), dtype=np.float32
                )
                padded.fill(NODATA_VALUE)
                padded[:, :ph, :pw] = patch
                tiles.append((padded, row_start, col_start))

    return tiles


def tile_date(
    lake_id: str,
    date_str: str,
    tile_size: int = TILE_SIZE,
    overlap: int = 0,
    nodata_skip_threshold: float = 0.9,
) -> int:
    """
    Tile all stacked channels for one lake+date.

    Args:
        lake_id:               Lake identifier.
        date_str:              Date string YYYY-MM-DD.
        tile_size:             Tile dimension.
        overlap:               Overlap in pixels.
        nodata_skip_threshold: Skip tiles where more than this fraction of
                               pixels across all channels is nodata (-9999).
                               Default: 0.9 (skip tiles that are >90% nodata).

    Returns:
        Number of tiles saved.
    """
    stack_path = DATA_INTERIM / lake_id / "stacked" / f"{date_str}.tif"
    if not stack_path.exists():
        logger.warning(
            "[%s] Stacked file not found for %s — skipping. "
            "Run coregister_stack.py first.",
            lake_id, date_str,
        )
        return 0

    out_dir = DATA_PROCESSED / lake_id / date_str
    index_path = out_dir / "tile_index.json"

    if index_path.exists():
        existing = json.loads(index_path.read_text())
        logger.debug(
            "[%s] %s already tiled (%d tiles) — skipping.",
            lake_id, date_str, len(existing["tiles"]),
        )
        return len(existing["tiles"])

    out_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(stack_path) as src:
        image = src.read().astype(np.float32)  # (10, H, W)
        meta = {
            "crs": str(src.crs),
            "transform": list(src.transform),
            "shape": [src.count, src.height, src.width],
        }

    tiles = tile_image(image, tile_size=tile_size, overlap=overlap)
    saved = []

    for tile_arr, r0, c0 in tiles:
        # Skip tiles that are almost entirely nodata
        nodata_fraction = np.mean(tile_arr == -9999.0)
        if nodata_fraction > nodata_skip_threshold:
            continue

        row_id = r0 // max(tile_size - overlap, 1)
        col_id = c0 // max(tile_size - overlap, 1)
        tile_name = f"tile_{row_id:03d}_{col_id:03d}.npy"
        tile_path = out_dir / tile_name
        np.save(tile_path, tile_arr)
        valid_name = tile_name.replace(".npy", "_valid.npy")
        valid_path = out_dir / valid_name
        np.save(valid_path, np.isfinite(tile_arr) & (tile_arr != NODATA_VALUE))
        saved.append(
            {
                "file": tile_name,
                "valid_mask_file": valid_name,
                "row_offset": r0,
                "col_offset": c0,
                "nodata_fraction": float(nodata_fraction),
            }
        )

    # Write tile index
    index = {
        "lake_id": lake_id,
        "date": date_str,
        "tile_size": tile_size,
        "overlap": overlap,
        "source_stack": str(stack_path),
        "source_meta": meta,
        "tiles": saved,
    }
    with open(index_path, "w") as fh:
        json.dump(index, fh, indent=2)

    return len(saved)


# ─────────────────────────────────────────────────────────────────────────────
# Lake-level batch tiling
# ─────────────────────────────────────────────────────────────────────────────

def tile_lake(lake: dict, cfg: dict) -> None:
    """Tile all stacked dates for one lake."""
    tile_size = cfg["preprocessing"].get("tile_size", TILE_SIZE)
    overlap = cfg["preprocessing"].get("tile_overlap", 0)

    stacked_dir = DATA_INTERIM / lake["id"] / "stacked"
    if not stacked_dir.exists():
        logger.warning(
            "[%s] Stacked directory not found. Run coregister_stack.py first.",
            lake["id"],
        )
        return

    date_files = sorted(stacked_dir.glob("*.tif"))
    if not date_files:
        logger.warning("[%s] No stacked .tif files found.", lake["id"])
        return

    logger.info(
        "[%s] Tiling %d dates (tile=%dpx, overlap=%dpx) …",
        lake["id"], len(date_files), tile_size, overlap,
    )

    total_tiles = 0
    for stack_path in tqdm(date_files, desc=f"Tile {lake['id']}", unit="date"):
        date_str = stack_path.stem  # e.g. "2023-06-15"
        n = tile_date(lake["id"], date_str, tile_size=tile_size, overlap=overlap)
        total_tiles += n

    logger.info(
        "[%s] Tiling done — %d total tiles across %d dates.",
        lake["id"], total_tiles, len(date_files),
    )


def print_tile_stats(lake_id: str) -> None:
    """Print summary statistics for all tiled dates of a lake."""
    processed_dir = DATA_PROCESSED / lake_id
    if not processed_dir.exists():
        logger.warning("[%s] No processed data found.", lake_id)
        return

    date_dirs = sorted(processed_dir.iterdir())
    print(f"\n{'─'*60}")
    print(f"  Lake: {lake_id} — {len(date_dirs)} dates tiled")
    print(f"{'─'*60}")
    total = 0
    for d in date_dirs:
        index_path = d / "tile_index.json"
        if index_path.exists():
            idx = json.loads(index_path.read_text())
            n = len(idx["tiles"])
            total += n
            print(f"  {d.name}: {n} tiles")
    print(f"{'─'*60}")
    print(f"  TOTAL: {total} tiles")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tile 10-channel stacked GeoTIFFs into 256×256 NumPy patches."
    )
    parser.add_argument("--lake", required=True, help="Lake ID or 'all'.")
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print tile count stats for the lake without running tiling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_pipeline_config()

    if args.lake == "all":
        lakes = get_active_lakes(exclude_placeholder_coords=True)
    else:
        lakes = [get_lake_by_id(args.lake)]

    for lake in lakes:
        logger.info("━━━ Lake: %s (%s) ━━━", lake["name"], lake["id"])
        if args.stats:
            print_tile_stats(lake["id"])
        else:
            tile_lake(lake, cfg)

    logger.info("Tiling complete.")


if __name__ == "__main__":
    main()
