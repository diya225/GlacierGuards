"""
asf_hyp3_fetch.py — Submit SAR RTC jobs to NASA HyP3 via ASF and download results.

For lakes in steep terrain (Tsho Rolpa, South Lhonak, Gongbatongsha Co), GEE's
built-in S1 data does not apply full radiometric terrain correction (RTC). This
script submits on-demand RTC processing jobs to NASA HyP3, which applies Enhanced
Lee filtering and range-Doppler terrain correction. The results are analysis-ready
backscatter rasters that reduce geometric distortion in steep terrain.

Usage
-----
    # Submit and download RTC jobs for South Lhonak, 2022:
    python src/ingestion/asf_hyp3_fetch.py --lake south_lhonak --start 2022-01-01 --end 2022-12-31

    # Check status of running jobs without submitting new ones:
    python src/ingestion/asf_hyp3_fetch.py --lake south_lhonak --status-only

Notes
-----
    - HyP3 free tier: 8,000 credits/month. One RTC_GAMMA job ≈ 25 credits.
      That's ~320 jobs/month — enough for ~27 jobs per lake across 10 lakes.
    - Use this script selectively for high-priority steep-terrain lakes only.
    - GEE-fetched S1 (gee_fetch.py) is sufficient for lower-relief sites.

Output
------
    data/raw/{lake_id}/s1_rtc/{YYYY-MM-DD}/   HyP3 RTC product folder per scene
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import asf_search as asf
import hyp3_sdk as hyp3
from shapely.geometry import box, mapping

from src.utils.auth import get_earthdata_credentials
from src.utils.config_loader import get_lake_by_id, get_active_lakes, load_pipeline_config

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

DATA_RAW = PROJECT_ROOT / "data" / "raw"

# Lakes that justify RTC processing due to steep surrounding terrain
STEEP_TERRAIN_LAKES = {"south_lhonak", "tsho_rolpa", "gongbatongsha_co"}

# HyP3 job parameters
RTC_RESOLUTION = 30          # metres — HyP3 standard RTC output
RTC_DEM_MATCHING = True
RTC_INCLUDE_DEM = True
RTC_INCLUDE_INC_MAP = True


def _lake_wkt_polygon(lake: dict, buffer_km: float | None = None) -> str:
    """Return a WKT polygon for the lake AOI (used in ASF search)."""
    km = buffer_km or lake.get("buffer_km", 2.0)
    deg = km / 111.0  # rough degree equivalent
    lat, lon = lake["lat"], lake["lon"]
    poly = box(lon - deg, lat - deg, lon + deg, lat + deg)
    return poly.wkt


def search_s1_granules(
    lake: dict,
    start_date: str,
    end_date: str,
) -> list:
    """Search ASF for Sentinel-1 GRD_HD granules over the lake AOI."""
    wkt = _lake_wkt_polygon(lake)
    logger.info(
        "[%s] Searching ASF for S1 granules %s → %s …",
        lake["id"], start_date, end_date,
    )
    results = asf.search(
        platform=asf.PLATFORM.SENTINEL1,
        processingLevel=[asf.PRODUCT_TYPE.GRD_HD],
        intersectsWith=wkt,
        start=start_date,
        end=end_date,
        beamMode=[asf.BEAMMODE.IW],
    )
    logger.info("[%s] Found %d ASF granules.", lake["id"], len(results))
    return results


def submit_rtc_jobs(
    lake: dict,
    granules: list,
    hyp3_client: hyp3.HyP3,
) -> list:
    """
    Submit RTC_GAMMA jobs for granules not already downloaded.
    Skips granules whose output folder already exists (idempotency).
    Returns list of submitted HyP3 Job objects.
    """
    out_base = DATA_RAW / lake["id"] / "s1_rtc"
    out_base.mkdir(parents=True, exist_ok=True)

    submitted = []
    skipped = 0

    for granule in granules:
        granule_name = granule.properties["sceneName"]
        acq_date = granule.properties["startTime"][:10]  # YYYY-MM-DD
        out_dir = out_base / acq_date

        if out_dir.exists() and any(out_dir.iterdir()):
            logger.debug("RTC %s already downloaded — skipping.", granule_name)
            skipped += 1
            continue

        job = hyp3_client.submit_rtc_job(
            granule=granule_name,
            name=f"glof_ews_{lake['id']}_{acq_date}",
            resolution=RTC_RESOLUTION,
            dem_matching=RTC_DEM_MATCHING,
            include_dem=RTC_INCLUDE_DEM,
            include_inc_map=RTC_INCLUDE_INC_MAP,
        )
        submitted.append((job, lake["id"], acq_date))
        logger.info("Submitted RTC job: %s (%s)", granule_name, acq_date)

    if skipped:
        logger.info("[%s] Skipped %d already-processed granules.", lake["id"], skipped)
    logger.info("[%s] Submitted %d new RTC jobs.", lake["id"], len(submitted))
    return submitted


def wait_and_download(
    jobs: list,
    hyp3_client: hyp3.HyP3,
    poll_interval_s: int = 60,
    max_wait_h: float = 4.0,
) -> None:
    """Poll HyP3 for job completion and download output products."""
    if not jobs:
        logger.info("No pending jobs to wait for.")
        return

    deadline = time.time() + max_wait_h * 3600
    pending = list(jobs)  # (job_obj, lake_id, date_str)
    logger.info("Waiting for %d HyP3 RTC jobs (timeout %.1fh) …", len(pending), max_wait_h)

    while pending and time.time() < deadline:
        time.sleep(poll_interval_s)
        still_pending = []

        for job_obj, lake_id, date_str in pending:
            refreshed = hyp3_client.get_job_by_id(job_obj.job_id)

            if refreshed.status_code == "SUCCEEDED":
                out_dir = DATA_RAW / lake_id / "s1_rtc" / date_str
                out_dir.mkdir(parents=True, exist_ok=True)
                logger.info(
                    "Job SUCCEEDED: %s %s — downloading …", lake_id, date_str
                )
                refreshed.download_files(location=out_dir)
                logger.info("Downloaded RTC product → %s", out_dir)

            elif refreshed.status_code == "FAILED":
                logger.error(
                    "Job FAILED: %s %s — check HyP3 dashboard for details.",
                    lake_id, date_str,
                )
            else:
                # RUNNING or PENDING
                still_pending.append((job_obj, lake_id, date_str))

        pending = still_pending
        if pending:
            logger.info(
                "%d job(s) still pending … checking again in %ds",
                len(pending), poll_interval_s,
            )

    if pending:
        logger.warning(
            "%d job(s) did not complete within %.1fh. "
            "Run with --status-only later to download them.",
            len(pending), max_wait_h,
        )


def check_running_jobs(hyp3_client: hyp3.HyP3) -> None:
    """Print the status of all running HyP3 jobs for this project."""
    logger.info("Fetching all HyP3 jobs for account …")
    jobs = hyp3_client.find_jobs(name="glof_ews")
    counts: dict[str, int] = {}
    for job in jobs:
        counts[job.status_code] = counts.get(job.status_code, 0) + 1
    for status, count in counts.items():
        logger.info("  %s: %d job(s)", status, count)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit and download HyP3 RTC SAR jobs via ASF for GLOF lake monitoring."
    )
    parser.add_argument(
        "--lake",
        default=None,
        help=(
            "Lake ID (e.g. 'south_lhonak') or 'all' for all steep-terrain lakes. "
            "If omitted, processes the STEEP_TERRAIN_LAKES set."
        ),
    )
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD.")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD.")
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Only check and print job statuses without submitting new ones.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    username, password = get_earthdata_credentials()
    hyp3_client = hyp3.HyP3(username=username, password=password)

    if args.status_only:
        check_running_jobs(hyp3_client)
        return

    cfg = load_pipeline_config()
    start_date = args.start or cfg["data"]["start_date"]
    end_date = args.end or cfg["data"]["end_date"]

    # ── Resolve lake list ────────────────────────────────────────────────────
    if args.lake == "all":
        lakes = [
            get_lake_by_id(lid)
            for lid in STEEP_TERRAIN_LAKES
        ]
    elif args.lake:
        lakes = [get_lake_by_id(args.lake)]
    else:
        # Default: all steep-terrain lakes that are verified
        all_active = {l["id"]: l for l in get_active_lakes()}
        lakes = [all_active[lid] for lid in STEEP_TERRAIN_LAKES if lid in all_active]

    # ── Search + submit + download ───────────────────────────────────────────
    all_jobs = []
    for lake in lakes:
        logger.info("━━━ Lake: %s (%s) ━━━", lake["name"], lake["id"])
        granules = search_s1_granules(lake, start_date, end_date)
        if not granules:
            logger.warning("[%s] No granules found — check AOI and date range.", lake["id"])
            continue
        jobs = submit_rtc_jobs(lake, granules, hyp3_client)
        all_jobs.extend(jobs)

    wait_and_download(all_jobs, hyp3_client)
    logger.info("ASF/HyP3 fetch complete.")


if __name__ == "__main__":
    main()
