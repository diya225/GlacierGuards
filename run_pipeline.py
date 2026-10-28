"""
run_pipeline.py — Quick-start runner: chains M1→M4 steps for a single lake.

Usage
-----
    # Run full ingestion + preprocessing for one lake:
    python run_pipeline.py --lake south_lhonak --start 2022-01-01 --end 2023-12-31

    # Only preprocess (assumes data already downloaded):
    python run_pipeline.py --lake south_lhonak --skip-download

    # Only download, skip preprocessing:
    python run_pipeline.py --lake south_lhonak --skip-preprocess
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_step(script: str, args: list[str], step_name: str) -> bool:
    """Run a pipeline step as a subprocess. Returns True if successful."""
    cmd = [sys.executable, script] + args
    logger.info("─" * 60)
    logger.info("STEP: %s", step_name)
    logger.info("CMD:  %s", " ".join(cmd))
    logger.info("─" * 60)
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        logger.error("STEP FAILED: %s (exit code %d)", step_name, result.returncode)
        return False
    logger.info("STEP DONE: %s ✓", step_name)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GLOF EWS pipeline steps M1→M4 for a single lake."
    )
    parser.add_argument("--lake", required=True, help="Lake ID (e.g. 'south_lhonak').")
    parser.add_argument("--start", default="2020-01-01", help="Start date YYYY-MM-DD.")
    parser.add_argument("--end", default="2023-12-31", help="End date YYYY-MM-DD.")
    parser.add_argument("--skip-download", action="store_true", help="Skip ingestion step.")
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip preprocessing steps.")
    parser.add_argument("--sensor", default="all", choices=["s1", "s2", "dem", "all"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    failures = []

    print("\n" + "═" * 60)
    print(f"  GLOF EWS Pipeline — Lake: {args.lake}")
    print(f"  Date range: {args.start} → {args.end}")
    print("═" * 60 + "\n")

    # ── M2: Ingestion ─────────────────────────────────────────────────────────
    if not args.skip_download:
        ok = run_step(
            "src/ingestion/gee_fetch.py",
            ["--lake", args.lake, "--start", args.start, "--end", args.end, "--sensor", args.sensor],
            "M2 — GEE Download (S1 + S2 + DEM)",
        )
        if not ok:
            failures.append("GEE Download")

    # ── M3: Preprocessing ─────────────────────────────────────────────────────
    if not args.skip_preprocess:
        steps = [
            ("src/preprocessing/sar_preprocess.py",   ["--lake", args.lake], "M3a — SAR Enhanced Lee Filter"),
            ("src/preprocessing/optical_indices.py",  ["--lake", args.lake], "M3b — Spectral Indices"),
            ("src/preprocessing/coregister_stack.py", ["--lake", args.lake], "M3c — Co-register & Stack"),
            ("src/preprocessing/tile_and_pad.py",     ["--lake", args.lake], "M3d — Tile 256×256"),
        ]
        for script, step_args, name in steps:
            ok = run_step(script, step_args, name)
            if not ok:
                failures.append(name)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    if failures:
        print(f"  ❌ Completed with {len(failures)} failure(s):")
        for f in failures:
            print(f"     • {f}")
    else:
        print("  ✅ All pipeline steps completed successfully!")
        print()
        print("  Next steps:")
        print("  1. Check data/interim/{lake}/stacked/ for stacked tensors")
        print("  2. Open QGIS and start digitizing masks:")
        print("     python src/labeling/qgis_export_helper.py --help-qgis")
        print("  3. After digitizing, convert shapefiles to masks:")
        print("     python src/labeling/qgis_export_helper.py --lake", args.lake, "--shapefile-dir data/labels/raw/")
    print("═" * 60 + "\n")

    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
