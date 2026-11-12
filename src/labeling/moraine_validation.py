"""
moraine_validation.py — Validate and export manual moraine-dam QGIS annotations.

Validates a 3-layer GeoPackage (`dam_footprint`, `crest_line`, `crest_points`),
computes geometric and elevation summary metrics in a metric UTM CRS, writes
JSON metadata + validation reports, and produces a backward-compatible YAML
block for `configs/lakes.yaml`.

Does NOT silently modify invalid geometries.
"""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
from pyproj import CRS
from shapely.validation import explain_validity

from src.labeling.dem_sampling import DEM_DISCLAIMER, sample_dem_at_points
from src.utils.config_loader import PROJECT_ROOT, get_lake_by_id

logger = logging.getLogger(__name__)

LAYER_SPECS: dict[str, dict[str, Any]] = {
    "dam_footprint": {
        "geom_type": "Polygon",
        "required_cols": ["annotation_id", "lake_id", "confidence", "notes"],
        "non_null_cols": ["annotation_id", "lake_id"],
    },
    "crest_line": {
        "geom_type": "LineString",
        "required_cols": ["annotation_id", "lake_id", "confidence", "notes"],
        "non_null_cols": ["annotation_id", "lake_id"],
    },
    "crest_points": {
        "geom_type": "Point",
        "required_cols": [
            "point_id",
            "lake_id",
            "point_type",
            "confidence",
            "elevation_m",
            "notes",
        ],
        "non_null_cols": ["point_id", "lake_id", "point_type", "elevation_m"],
    },
}

VALID_POINT_TYPES = {
    "crest_endpoint",
    "lowest_saddle",
    "local_depression",
    "representative_crest",
}


def _utm_crs_for_point(lat: float, lon: float) -> CRS:
    """Compute the metric UTM CRS for a WGS84 lat/lon coordinate."""
    zone = int((lon + 180.0) / 6.0) + 1
    epsg = (32600 if lat >= 0 else 32700) + zone
    return CRS.from_epsg(epsg)


def _is_missing(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return True
    if isinstance(val, str) and val.strip() == "":
        return True
    return False


def _validate_layer_structure(
    gdf: gpd.GeoDataFrame,
    layer_name: str,
    expected_lake_id: str | None,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate non-emptiness, CRS, geometry types/validity, and attributes."""
    spec = LAYER_SPECS[layer_name]

    if gdf.empty:
        errors.append(f"[{layer_name}] Layer is empty (must contain at least 1 feature).")
        return

    if gdf.crs is None:
        errors.append(f"[{layer_name}] Coordinate reference system (CRS) is missing.")

    missing_cols = [c for c in spec["required_cols"] if c not in gdf.columns]
    if missing_cols:
        errors.append(f"[{layer_name}] Missing required attribute columns: {missing_cols}")

    expected_geom = spec["geom_type"]
    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            errors.append(f"[{layer_name}] Feature {idx} has null or empty geometry.")
            continue
        if geom.geom_type != expected_geom:
            errors.append(
                f"[{layer_name}] Feature {idx} has geometry type '{geom.geom_type}' "
                f"(expected '{expected_geom}')."
            )
        if not geom.is_valid:
            reason = explain_validity(geom)
            errors.append(
                f"[{layer_name}] Feature {idx} has invalid geometry: {reason}"
            )

    for col in spec["non_null_cols"]:
        if col not in gdf.columns:
            continue
        null_indices = [idx for idx, val in gdf[col].items() if _is_missing(val)]
        if null_indices:
            errors.append(
                f"[{layer_name}] Required attribute '{col}' is missing/empty for features: {null_indices}"
            )

    if "lake_id" in gdf.columns and expected_lake_id is not None:
        mismatched = [
            str(val)
            for val in gdf["lake_id"]
            if not _is_missing(val) and str(val).strip() != expected_lake_id
        ]
        if mismatched:
            errors.append(
                f"[{layer_name}] Found lake_id values {sorted(set(mismatched))} "
                f"that do not match expected '{expected_lake_id}'."
            )

    if "confidence" in gdf.columns:
        bad_conf = []
        for idx, val in gdf["confidence"].items():
            if _is_missing(val):
                warnings.append(f"[{layer_name}] Feature {idx} has missing confidence.")
                continue
            try:
                c_int = int(val)
                if c_int < 1 or c_int > 3:
                    bad_conf.append((idx, val))
            except (TypeError, ValueError):
                bad_conf.append((idx, val))
        if bad_conf:
            warnings.append(
                f"[{layer_name}] Confidence values outside 1..3 range: {bad_conf}"
            )


def validate_moraine_geopackage(
    gpkg_path: Path,
    lake_id: str,
    dem_path: Path | None = None,
    extract_dem: bool = False,
    save_extracted_gpkg: bool = False,
) -> dict[str, Any]:
    """
    Load, optionally sample DEM elevations, validate, and summarize a 3-layer
    moraine annotation GeoPackage.

    Returns a structured report dict containing:
      - `valid` (bool)
      - `errors` (list[str])
      - `warnings` (list[str])
      - `metrics` (dict)
      - `crs_info` (dict)
      - `dem_sampling` (dict | None)
      - `crest_points_wgs84` (list[dict])
    """
    gpkg_path = Path(gpkg_path)
    if not gpkg_path.exists():
        raise FileNotFoundError(f"GeoPackage not found: {gpkg_path}")

    lake_cfg = get_lake_by_id(lake_id)
    errors: list[str] = []
    warnings: list[str] = []
    layers: dict[str, gpd.GeoDataFrame] = {}

    for layer_name in LAYER_SPECS:
        try:
            layers[layer_name] = gpd.read_file(gpkg_path, layer=layer_name)
        except Exception as exc:
            errors.append(f"[{layer_name}] Could not read layer from {gpkg_path}: {exc}")
            layers[layer_name] = gpd.GeoDataFrame()

    dem_meta: dict[str, Any] | None = None
    if extract_dem and dem_path is not None and not layers["crest_points"].empty:
        pts_sampled, dem_meta = sample_dem_at_points(
            layers["crest_points"],
            dem_path=dem_path,
            elevation_col="elevation_m",
            raise_on_invalid=False,
        )
        layers["crest_points"] = pts_sampled
        if dem_meta["invalid_samples"]:
            errors.append(
                f"[crest_points] DEM sampling failed for {len(dem_meta['invalid_samples'])} "
                f"point(s): {dem_meta['invalid_samples']}"
            )
        if save_extracted_gpkg and not errors:
            # Write updated crest_points back to GeoPackage only if requested and valid
            pts_sampled.to_file(gpkg_path, layer="crest_points", driver="GPKG")

    # Validate each layer's structure, CRS, geometry, and attributes
    for layer_name, gdf in layers.items():
        if gdf.empty and any(f"[{layer_name}] Could not read layer" in e for e in errors):
            continue
        _validate_layer_structure(gdf, layer_name, lake_id, errors, warnings)

    # Point-specific validations
    pts = layers["crest_points"]
    if not pts.empty:
        if "point_id" in pts.columns:
            non_null_ids = [str(v).strip() for v in pts["point_id"] if not _is_missing(v)]
            seen: set[str] = set()
            dups: set[str] = set()
            for pid in non_null_ids:
                if pid in seen:
                    dups.add(pid)
                seen.add(pid)
            if dups:
                errors.append(f"[crest_points] Duplicate point_id values: {sorted(dups)}")

        if "point_type" in pts.columns:
            pt_types = [
                str(v).strip() if not _is_missing(v) else ""
                for v in pts["point_type"]
            ]
            invalid_types = sorted(
                {t for t in pt_types if t and t not in VALID_POINT_TYPES}
            )
            if invalid_types:
                errors.append(
                    f"[crest_points] Unsupported point_type value(s): {invalid_types}. "
                    f"Allowed: {sorted(VALID_POINT_TYPES)}"
                )

            lowest_mask = [t == "lowest_saddle" for t in pt_types]
            n_lowest = sum(lowest_mask)
            if n_lowest == 0:
                errors.append(
                    "[crest_points] Missing 'lowest_saddle' annotation "
                    "(exactly 1 point must have point_type='lowest_saddle')."
                )
            elif n_lowest > 1:
                errors.append(
                    f"[crest_points] Found {n_lowest} 'lowest_saddle' points "
                    "(expected exactly 1)."
                )

        if "elevation_m" in pts.columns:
            elevs: list[float] = []
            bad_elevs: list[tuple[Any, Any]] = []
            for idx, val in pts["elevation_m"].items():
                if _is_missing(val):
                    bad_elevs.append((idx, val))
                    continue
                try:
                    fval = float(val)
                    if not math.isfinite(fval) or fval <= 0.0 or fval > 9000.0:
                        bad_elevs.append((idx, val))
                    else:
                        elevs.append(fval)
                except (TypeError, ValueError):
                    bad_elevs.append((idx, val))

            if bad_elevs:
                errors.append(
                    f"[crest_points] Missing or invalid elevation_m values: {bad_elevs}"
                )

            if (
                "point_type" in pts.columns
                and not bad_elevs
                and elevs
                and sum(str(v).strip() == "lowest_saddle" for v in pts["point_type"] if not _is_missing(v)) == 1
            ):
                saddle_row = pts[pts["point_type"].astype(str).str.strip() == "lowest_saddle"].iloc[0]
                saddle_elev = float(saddle_row["elevation_m"])
                min_elev = min(elevs)
                if saddle_elev > min_elev + 0.5:
                    warnings.append(
                        f"[crest_points] lowest_saddle elevation ({saddle_elev:.2f} m) is "
                        f"higher than minimum crest point elevation ({min_elev:.2f} m)."
                    )

        if len(pts) < 3:
            warnings.append(
                f"[crest_points] Only {len(pts)} crest point(s) annotated; "
                "recommend at least 1 lowest_saddle + 3-5 representative_crest points."
            )

    # Compute metrics in metric UTM CRS
    metric_crs = _utm_crs_for_point(lake_cfg["lat"], lake_cfg["lon"])
    metrics: dict[str, Any] = {
        "dam_footprint_area_m2": None,
        "crest_line_length_m": None,
        "crest_elevation_min_m": None,
        "crest_elevation_max_m": None,
        "crest_elevation_mean_m": None,
        "lowest_saddle_elevation_m": None,
        "lowest_saddle_point_id": None,
        "num_crest_points": int(len(pts)),
    }

    dam_gdf = layers["dam_footprint"]
    if not dam_gdf.empty and dam_gdf.crs is not None and all(g is not None and g.is_valid for g in dam_gdf.geometry):
        dam_utm = dam_gdf.to_crs(metric_crs)
        metrics["dam_footprint_area_m2"] = round(float(dam_utm.geometry.area.sum()), 2)

    line_gdf = layers["crest_line"]
    if not line_gdf.empty and line_gdf.crs is not None and all(g is not None and g.is_valid for g in line_gdf.geometry):
        line_utm = line_gdf.to_crs(metric_crs)
        metrics["crest_line_length_m"] = round(float(line_utm.geometry.length.sum()), 2)

    crest_points_wgs84: list[dict[str, Any]] = []
    if not pts.empty and "elevation_m" in pts.columns:
        valid_elev_vals = [
            float(v) for v in pts["elevation_m"]
            if not _is_missing(v) and math.isfinite(float(v))
        ]
        if valid_elev_vals:
            metrics["crest_elevation_min_m"] = round(float(np.min(valid_elev_vals)), 3)
            metrics["crest_elevation_max_m"] = round(float(np.max(valid_elev_vals)), 3)
            metrics["crest_elevation_mean_m"] = round(float(np.mean(valid_elev_vals)), 3)

        if "point_type" in pts.columns:
            saddles = pts[pts["point_type"].astype(str).str.strip() == "lowest_saddle"]
            if len(saddles) == 1 and not _is_missing(saddles.iloc[0]["elevation_m"]):
                s_row = saddles.iloc[0]
                metrics["lowest_saddle_elevation_m"] = round(float(s_row["elevation_m"]), 3)
                metrics["lowest_saddle_point_id"] = str(s_row.get("point_id", ""))

        if pts.crs is not None:
            pts_4326 = pts.to_crs(epsg=4326)
            for _, row in pts_4326.iterrows():
                geom = row.geometry
                if geom is None or geom.is_empty or geom.geom_type != "Point":
                    continue
                elev_val = row.get("elevation_m")
                p_type = str(row.get("point_type", "")).strip()
                conf_val = row.get("confidence")
                entry: dict[str, Any] = {
                    "lat": round(float(geom.y), 6),
                    "lon": round(float(geom.x), 6),
                    "elev_m": None if _is_missing(elev_val) else round(float(elev_val), 2),
                    "is_lowest": p_type == "lowest_saddle",
                    "point_id": str(row.get("point_id", "")),
                    "point_type": p_type,
                }
                if not _is_missing(conf_val):
                    entry["confidence"] = int(conf_val)
                crest_points_wgs84.append(entry)

            # Sort lowest_saddle first, then by elevation ascending
            crest_points_wgs84.sort(
                key=lambda d: (
                    not d["is_lowest"],
                    float("inf") if d["elev_m"] is None else d["elev_m"],
                )
            )

    crs_info = {
        layer_name: (str(gdf.crs) if not gdf.empty and gdf.crs is not None else None)
        for layer_name, gdf in layers.items()
    }
    crs_info["metric_calculation_crs"] = str(metric_crs)

    return {
        "lake_id": lake_id,
        "gpkg_path": str(gpkg_path),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "metrics": metrics,
        "crs_info": crs_info,
        "dem_sampling": dem_meta,
        "crest_points_wgs84": crest_points_wgs84,
        "disclaimer": DEM_DISCLAIMER,
    }


def format_lakes_yaml_block(report: dict[str, Any]) -> str:
    """
    Format a backward-compatible YAML snippet for `configs/lakes.yaml`.

    Preserves the existing `crest_points:` list item keys (`lat`, `lon`, `elev_m`,
    `is_lowest`) while adding optional `point_id`, `point_type`, and `confidence`,
    plus a minimal `moraine_annotation:` reference block.
    """
    lake_id = report["lake_id"]
    pts = report.get("crest_points_wgs84", [])
    metrics = report.get("metrics", {})
    gpkg_rel = Path(report["gpkg_path"])
    try:
        gpkg_rel_str = gpkg_rel.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        gpkg_rel_str = gpkg_rel.as_posix()

    lines = [f"# --- {lake_id} (validated moraine annotation) ---", "    crest_points:"]
    if not pts:
        lines.append("      []")
    else:
        for p in pts:
            elev_str = "null" if p["elev_m"] is None else f"{p['elev_m']:.2f}"
            parts = [
                f"lat: {p['lat']:.6f}",
                f"lon: {p['lon']:.6f}",
                f"elev_m: {elev_str}",
                f"is_lowest: {str(bool(p['is_lowest'])).lower()}",
                f"point_id: {p['point_id']}",
                f"point_type: {p['point_type']}",
            ]
            if "confidence" in p:
                parts.append(f"confidence: {p['confidence']}")
            lines.append("      - {" + ", ".join(parts) + "}")

    lines.append("    moraine_annotation:")
    lines.append(f"      gpkg_path: \"{gpkg_rel_str}\"")
    if metrics.get("lowest_saddle_elevation_m") is not None:
        lines.append(f"      lowest_saddle_elev_m: {metrics['lowest_saddle_elevation_m']:.2f}")
    if metrics.get("crest_line_length_m") is not None:
        lines.append(f"      crest_length_m: {metrics['crest_line_length_m']:.2f}")
    if metrics.get("dam_footprint_area_m2") is not None:
        lines.append(f"      footprint_area_m2: {metrics['dam_footprint_area_m2']:.2f}")
    lines.append("      surveyed: false")

    return "\n".join(lines)


def write_annotation_outputs(
    report: dict[str, Any],
    output_dir: Path,
    yaml_out: Path | None = None,
) -> tuple[Path, Path]:
    """
    Write `{lake_id}_annotation_report.json` and `{lake_id}_annotation_metadata.json`
    (plus optional YAML snippet) into `output_dir`.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lake_id = report["lake_id"]

    report_path = output_dir / f"{lake_id}_annotation_report.json"
    report_payload = {
        "lake_id": lake_id,
        "generated_at_utc": report["generated_at_utc"],
        "valid": report["valid"],
        "errors": report["errors"],
        "warnings": report["warnings"],
        "metrics": report["metrics"],
    }
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    metadata_path = output_dir / f"{lake_id}_annotation_metadata.json"
    metadata_payload = {
        "lake_id": lake_id,
        "gpkg_path": report["gpkg_path"],
        "generated_at_utc": report["generated_at_utc"],
        "crs_info": report["crs_info"],
        "dem_sampling": report["dem_sampling"],
        "crest_points_wgs84": report["crest_points_wgs84"],
        "disclaimer": report["disclaimer"],
    }
    metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")

    if yaml_out is not None:
        yaml_out = Path(yaml_out)
        yaml_out.parent.mkdir(parents=True, exist_ok=True)
        yaml_out.write_text(format_lakes_yaml_block(report) + "\n", encoding="utf-8")

    return report_path, metadata_path


def apply_to_lakes_yaml(
    report: dict[str, Any],
    lakes_yaml_path: Path | None = None,
) -> Path:
    """
    Surgically update the target lake's `crest_points:` (and `moraine_annotation:`)
    in `configs/lakes.yaml` while preserving all comments, formatting, and other lakes.

    Raises ValueError if the report is invalid or the lake block cannot be located.
    """
    if not report.get("valid"):
        raise ValueError("Refusing to update configs/lakes.yaml: annotation report has validation errors.")

    path = Path(lakes_yaml_path) if lakes_yaml_path else (PROJECT_ROOT / "configs" / "lakes.yaml")
    text = path.read_text(encoding="utf-8")
    lake_id = report["lake_id"]

    # Locate the `- id: <lake_id>` block up to the next `  - id:` or EOF
    pattern = re.compile(
        rf"(^  - id:\s*{re.escape(lake_id)}\b.*?)(?=^  - id:|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        raise ValueError(f"Could not find lake block '- id: {lake_id}' in {path}")

    block = match.group(1)

    # Build replacement crest_points + moraine_annotation lines
    yaml_lines = format_lakes_yaml_block(report).splitlines()
    # Strip the leading comment line `# --- ... ---`
    yaml_body = "\n".join(line for line in yaml_lines if not line.startswith("#"))

    # Replace existing `crest_points:` (and optional `moraine_annotation:`)
    # First remove existing `moraine_annotation:` block if present
    block = re.sub(
        r"\n    moraine_annotation:\n(?:      .*\n)*",
        "\n",
        block,
    )

    cp_pattern = re.compile(
        r"    crest_points:.*?(?=\n    [a-zA-Z_]+:|\Z)",
        re.DOTALL,
    )
    if cp_pattern.search(block):
        # Extract just the crest_points section and moraine_annotation section
        updated_block = cp_pattern.sub(yaml_body, block, count=1)
    else:
        updated_block = block.rstrip() + "\n" + yaml_body + "\n"

    new_text = text[: match.start(1)] + updated_block + text[match.end(1) :]
    path.write_text(new_text, encoding="utf-8")
    return path
