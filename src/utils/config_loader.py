"""
config_loader.py — Loads and validates lakes.yaml and pipeline.yaml.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"


def load_pipeline_config() -> dict[str, Any]:
    """Load configs/pipeline.yaml and return as a dict."""
    path = CONFIGS_DIR / "pipeline.yaml"
    if not path.exists():
        raise FileNotFoundError(f"pipeline.yaml not found at {path}")
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    logger.debug("Pipeline config loaded from %s", path)
    return cfg


def load_lakes_config() -> list[dict[str, Any]]:
    """
    Load configs/lakes.yaml and return the list of lake dicts.
    Warns about any lake whose coordinates are still at placeholder 0.0.
    """
    path = CONFIGS_DIR / "lakes.yaml"
    if not path.exists():
        raise FileNotFoundError(f"lakes.yaml not found at {path}")
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    lakes = raw.get("lakes", [])
    for lake in lakes:
        if lake.get("lat") == 0.0 or lake.get("lon") == 0.0:
            logger.warning(
                "Lake '%s' has placeholder coordinates (0.0, 0.0). "
                "Verify and update configs/lakes.yaml before running ingestion.",
                lake.get("id", "unknown"),
            )
    logger.info("Loaded %d lake entries from lakes.yaml", len(lakes))
    return lakes


def get_lake_by_id(lake_id: str) -> dict[str, Any]:
    """Return the lake config dict for a given lake_id. Raises KeyError if not found."""
    lakes = load_lakes_config()
    for lake in lakes:
        if lake["id"] == lake_id:
            return lake
    raise KeyError(
        f"Lake '{lake_id}' not found in configs/lakes.yaml. "
        f"Available IDs: {[l['id'] for l in lakes]}"
    )


def get_active_lakes(
    exclude_placeholder_coords: bool = True,
    roles: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Return lakes ready for processing.

    Args:
        exclude_placeholder_coords: Skip lakes still at (0.0, 0.0).
        roles: If provided, only return lakes whose 'role' is in this list.
               e.g. ['hindcast_positive', 'monitored', 'control']
    """
    lakes = load_lakes_config()
    result = []
    for lake in lakes:
        if exclude_placeholder_coords and (
            lake.get("lat") == 0.0 or lake.get("lon") == 0.0
        ):
            continue
        if roles and lake.get("role") not in roles:
            continue
        result.append(lake)
    return result
