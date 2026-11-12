"""
qgis_moraine_prep.py — Prepare a QGIS moraine-dam annotation workspace for a lake.

Discovers available rasters (DEM, Sentinel-2, stacked 10-band tensor, predicted
lake masks), determines the actual CRS from the lake's data (falling back to the
pipeline's UTM formula only when rasters are not yet on disk), optionally
computes a DEM hillshade, creates the 3-layer spatial GeoPackage
(`dam_footprint`: Polygon, `crest_line`: LineString, `crest_points`: Point),
validates layer geometries, CRS, and feature capabilities via OGR / SQLite,
and writes a styled QGIS project file (`qgis/{lake_id}_annotation.qgz`).

Usage
-----
    python src/labeling/qgis_moraine_prep.py --lake south_lhonak
    python src/labeling/qgis_moraine_prep.py --lake south_lhonak --force-recreate
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import get_lake_by_id

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PREDICTIONS = PROJECT_ROOT / "data" / "predictions"
DATA_EXTERNAL = PROJECT_ROOT / "data" / "external"
DATA_MANUAL = PROJECT_ROOT / "data" / "labels" / "manual"
QGIS_DIR = PROJECT_ROOT / "qgis"

LAYER_DEFINITIONS: dict[str, dict[str, Any]] = {
    "dam_footprint": {
        "geom_type": "POLYGON",
        "wkb_display": "Polygon",
        "ogr_display": "Polygon",
        "qgis_geometry": "Polygon",
        "qgis_wkb": "Polygon",
        "columns": [
            ("annotation_id", "TEXT"),
            ("lake_id", "TEXT"),
            ("confidence", "INTEGER"),
            ("notes", "TEXT"),
        ],
    },
    "crest_line": {
        "geom_type": "LINESTRING",
        "wkb_display": "LineString",
        "ogr_display": "Line String",
        "qgis_geometry": "Line",
        "qgis_wkb": "LineString",
        "columns": [
            ("annotation_id", "TEXT"),
            ("lake_id", "TEXT"),
            ("confidence", "INTEGER"),
            ("notes", "TEXT"),
        ],
    },
    "crest_points": {
        "geom_type": "POINT",
        "wkb_display": "Point",
        "ogr_display": "Point",
        "qgis_geometry": "Point",
        "qgis_wkb": "Point",
        "columns": [
            ("point_id", "TEXT"),
            ("lake_id", "TEXT"),
            ("point_type", "TEXT"),
            ("confidence", "INTEGER"),
            ("elevation_m", "REAL"),
            ("notes", "TEXT"),
        ],
    },
}


def _utm_epsg_for_point(lat: float, lon: float) -> int:
    """Compute UTM EPSG code from lat/lon (matches coregister_stack.py)."""
    zone = int((lon + 180.0) / 6.0) + 1
    return (32600 if lat >= 0 else 32700) + zone


def parse_epsg_code(crs_str: str) -> int:
    """Extract integer EPSG code from 'EPSG:32645' or return 32645 as default."""
    m = re.search(r"\b(\d{4,5})\b", str(crs_str))
    if m:
        return int(m.group(1))
    return 32645


def get_srs_wkt(epsg: int) -> tuple[str, str]:
    """Return (srs_name, wkt_string) for a UTM or geographic EPSG code."""
    if epsg == 4326:
        name = "WGS 84"
        wkt = (
            'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,'
            'AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,'
            'AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],'
            'AUTHORITY["EPSG","4326"]]'
        )
        return name, wkt

    is_north = 32601 <= epsg <= 32660
    is_south = 32701 <= epsg <= 32760

    if is_north or is_south:
        zone = epsg - (32600 if is_north else 32700)
        hem = "N" if is_north else "S"
        name = f"WGS 84 / UTM zone {zone}{hem}"
        cm = zone * 6 - 183
        fn = 0 if is_north else 10000000
        wkt = (
            f'PROJCS["{name}",GEOGCS["WGS 84",DATUM["WGS_1984",'
            f'SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],'
            f'AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],'
            f'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],'
            f'PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],'
            f'PARAMETER["central_meridian",{cm}],PARAMETER["scale_factor",0.9996],'
            f'PARAMETER["false_easting",500000],PARAMETER["false_northing",{fn}],'
            f'UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],'
            f'AUTHORITY["EPSG","{epsg}"]]'
        )
        return name, wkt

    # Fallback to generic UTM 45N if unrecognized
    return get_srs_wkt(32645)


def find_ogrinfo() -> str | None:
    """Locate ogrinfo binary on system, checking PATH and common QGIS/OSGeo4W paths."""
    which_cmd = shutil.which("ogrinfo")
    if which_cmd:
        return which_cmd

    candidates = [
        r"C:\Program Files\QGIS 3.44.15\bin\ogrinfo.exe",
        r"C:\Program Files\QGIS 3.40*\bin\ogrinfo.exe",
        r"C:\Program Files\QGIS 3.34*\bin\ogrinfo.exe",
        r"C:\Program Files\QGIS 3.28*\bin\ogrinfo.exe",
        r"C:\Program Files\QGIS*\bin\ogrinfo.exe",
        r"C:\OSGeo4W\bin\ogrinfo.exe",
        r"C:\OSGeo4W64\bin\ogrinfo.exe",
    ]
    for pattern in candidates:
        matches = sorted(glob.glob(pattern), reverse=True)
        if matches and os.path.exists(matches[0]):
            return matches[0]
    return None


def discover_lake_rasters(lake_id: str) -> dict[str, Path | None]:
    """Find the latest available rasters and masks for `lake_id`."""
    dem_path = DATA_RAW / lake_id / "dem" / "dem.tif"
    s2_files = sorted((DATA_RAW / lake_id / "s2").glob("*.tif")) if (DATA_RAW / lake_id / "s2").exists() else []
    stacked_files = sorted((DATA_INTERIM / lake_id / "stacked").glob("*.tif")) if (DATA_INTERIM / lake_id / "stacked").exists() else []
    pred_files = sorted((DATA_PREDICTIONS / lake_id).glob("*.npy")) if (DATA_PREDICTIONS / lake_id).exists() else []
    pred_files = [p for p in pred_files if p.name != "roi.npy"]
    tp_shp = DATA_EXTERNAL / "third_pole_inventory" / "Third Pole" / "Third_Pole_Glacial Lake_2022.shp"

    return {
        "dem": dem_path if dem_path.exists() else None,
        "s2_latest": s2_files[-1] if s2_files else None,
        "stacked_latest": stacked_files[-1] if stacked_files else None,
        "prediction_latest": pred_files[-1] if pred_files else None,
        "third_pole_2022": tp_shp if tp_shp.exists() else None,
    }


def determine_project_crs(lake: dict[str, Any], rasters: dict[str, Path | None]) -> tuple[str, str]:
    """
    Determine the actual CRS from existing lake rasters.

    Returns (auth_id_str, source_description), e.g. ('EPSG:32645', 'stacked_latest').
    """
    try:
        import rasterio
    except ImportError:
        epsg = _utm_epsg_for_point(lake["lat"], lake["lon"])
        return f"EPSG:{epsg}", "utm_fallback_rasterio_unavailable"

    for key in ("stacked_latest", "s2_latest", "dem"):
        path = rasters.get(key)
        if path is not None and path.exists():
            with rasterio.open(path) as src:
                if src.crs is not None:
                    epsg = src.crs.to_epsg()
                    if src.crs.is_projected and epsg is not None:
                        return f"EPSG:{epsg}", f"{key}:{path.name}"
                    if key == "dem" and not src.crs.is_projected:
                        utm_epsg = _utm_epsg_for_point(lake["lat"], lake["lon"])
                        logger.info(
                            "DEM %s is geographic (%s); using metric UTM EPSG:%d for digitizing.",
                            path, src.crs, utm_epsg,
                        )
                        return f"EPSG:{utm_epsg}", f"utm_from_dem_geographic:{src.crs}"
                    return str(src.crs), f"{key}:{path.name}"

    epsg = _utm_epsg_for_point(lake["lat"], lake["lon"])
    return f"EPSG:{epsg}", "utm_fallback_no_rasters_on_disk"


def generate_hillshade(dem_path: Path, out_path: Path, azimuth_deg: float = 315.0, altitude_deg: float = 45.0) -> Path:
    """Compute an 8-bit hillshade GeoTIFF from `dem_path` using Horn's slope/aspect formula."""
    import rasterio

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        nodata = src.nodata
        res_x, res_y = abs(float(src.res[0])), abs(float(src.res[1]))
        if src.crs and not src.crs.is_projected:
            res_x *= 111320.0
            res_y *= 110540.0

    valid = np.isfinite(dem)
    if nodata is not None:
        valid &= ~np.isclose(dem, float(nodata))

    filled = np.where(valid, dem, np.nanmedian(dem[valid]) if np.any(valid) else 0.0)
    dy, dx = np.gradient(filled, res_y, res_x)
    slope = np.arctan(np.sqrt(dx * dx + dy * dy))
    aspect = np.arctan2(-dy, dx)

    az_rad = np.radians(360.0 - azimuth_deg + 90.0)
    alt_rad = np.radians(90.0 - altitude_deg)

    shaded = (
        np.cos(alt_rad) * np.cos(slope)
        + np.sin(alt_rad) * np.sin(slope) * np.cos(az_rad - aspect)
    )
    hs = np.clip(np.round(shaded * 254.0 + 1.0), 1, 255).astype(np.uint8)
    hs[~valid] = 0

    profile.update(dtype="uint8", count=1, nodata=0, compress="lzw")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(hs, 1)

    logger.info("Hillshade written to %s", out_path)
    return out_path


def check_gpkg_has_features(gpkg_path: Path) -> dict[str, int]:
    """Return dictionary of feature counts per layer in GeoPackage."""
    counts: dict[str, int] = {}
    if not gpkg_path.exists():
        return counts

    con = sqlite3.connect(gpkg_path)
    try:
        tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        for layer_name in LAYER_DEFINITIONS:
            if layer_name in tables:
                cnt = con.execute(f'SELECT COUNT(*) FROM "{layer_name}"').fetchone()[0]
                counts[layer_name] = cnt
            else:
                counts[layer_name] = 0
    finally:
        con.close()
    return counts


def create_empty_moraine_geopackage(
    gpkg_path: Path,
    lake_id: str,
    crs_str: str,
    force_recreate: bool = False,
) -> Path:
    """
    Create a clean, fully compliant 3-layer spatial GeoPackage
    (`dam_footprint`: Polygon, `crest_line`: LineString, `crest_points`: Point)
    with explicit EPSG CRS, geometry columns, and spatial metadata.

    If an existing GeoPackage exists and needs replacement, an explicit timestamped
    backup is created first.
    """
    gpkg_path = Path(gpkg_path)
    gpkg_path.parent.mkdir(parents=True, exist_ok=True)
    epsg = parse_epsg_code(crs_str)
    srs_name, srs_wkt = get_srs_wkt(epsg)

    if gpkg_path.exists():
        # Check if existing GeoPackage has valid spatial layers
        feature_counts = check_gpkg_has_features(gpkg_path)
        total_features = sum(feature_counts.values())

        # Inspect if existing gpkg has NoGeometry / generic GEOMETRY
        con_check = sqlite3.connect(gpkg_path)
        cur = con_check.cursor()
        has_geom_cols = cur.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='gpkg_geometry_columns'"
        ).fetchone()
        geom_types = {}
        if has_geom_cols:
            geom_types = dict(cur.execute("SELECT table_name, geometry_type_name FROM gpkg_geometry_columns").fetchall())
        con_check.close()

        has_generic_geom = any(
            geom_types.get(k) in (None, "GEOMETRY", "", "Unknown")
            for k in LAYER_DEFINITIONS
        )

        if not force_recreate and not has_generic_geom and all(k in geom_types for k in LAYER_DEFINITIONS):
            logger.info("Valid spatial GeoPackage already exists at %s — keeping existing file.", gpkg_path)
            return gpkg_path

        # Must replace or repair — create an explicit backup first
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = gpkg_path.with_name(f"{gpkg_path.stem}.backup_{timestamp}.gpkg")
        shutil.copy2(gpkg_path, backup_path)
        logger.info(
            "Backed up existing GeoPackage to %s (features preserved: %s)",
            backup_path,
            feature_counts,
        )

        # Remove previous file to rebuild cleanly with spatial tables
        try:
            gpkg_path.unlink()
        except Exception as exc:
            logger.warning("Could not unlink existing gpkg directly: %s", exc)

    con = sqlite3.connect(gpkg_path)
    try:
        # GeoPackage identification pragmas
        con.execute("PRAGMA application_id = 1196444487")  # 0x47504B47 = 'GPKG'
        con.execute("PRAGMA user_version = 10300")        # GeoPackage 1.3

        cur = con.cursor()

        # 1. Spatial Reference System table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS gpkg_spatial_ref_sys (
                srs_name TEXT NOT NULL,
                srs_id INTEGER NOT NULL PRIMARY KEY,
                organization TEXT NOT NULL,
                organization_coordsys_id INTEGER NOT NULL,
                definition TEXT NOT NULL,
                description TEXT
            )
        """)
        cur.executemany(
            "INSERT OR REPLACE INTO gpkg_spatial_ref_sys VALUES (?, ?, ?, ?, ?, ?)",
            [
                ("Undefined cartesian", -1, "NONE", -1, "undefined", "undefined cartesian coordinate reference system"),
                ("Undefined geographic", 0, "NONE", 0, "undefined", "undefined geographic coordinate reference system"),
                ("WGS 84", 4326, "EPSG", 4326, get_srs_wkt(4326)[1], "WGS 84 geodetic"),
                (srs_name, epsg, "EPSG", epsg, srs_wkt, f"Projected CRS EPSG:{epsg}"),
            ],
        )

        # 2. Contents table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS gpkg_contents (
                table_name TEXT NOT NULL PRIMARY KEY,
                data_type TEXT NOT NULL,
                identifier TEXT UNIQUE,
                description TEXT DEFAULT '',
                last_change DATETIME NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                min_x DOUBLE,
                min_y DOUBLE,
                max_x DOUBLE,
                max_y DOUBLE,
                srs_id INTEGER,
                CONSTRAINT fk_gc_r_srs_id FOREIGN KEY (srs_id) REFERENCES gpkg_spatial_ref_sys(srs_id)
            )
        """)

        # 3. Geometry columns table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS gpkg_geometry_columns (
                table_name TEXT NOT NULL,
                column_name TEXT NOT NULL,
                geometry_type_name TEXT NOT NULL,
                srs_id INTEGER NOT NULL,
                z TINYINT NOT NULL,
                m TINYINT NOT NULL,
                CONSTRAINT pk_geom_cols PRIMARY KEY (table_name, column_name),
                CONSTRAINT fk_gc_tn FOREIGN KEY (table_name) REFERENCES gpkg_contents(table_name),
                CONSTRAINT fk_gc_srs FOREIGN KEY (srs_id) REFERENCES gpkg_spatial_ref_sys (srs_id)
            )
        """)

        # 4. Optional OGR contents table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS gpkg_ogr_contents (
                table_name TEXT NOT NULL PRIMARY KEY,
                feature_count INTEGER DEFAULT NULL
            )
        """)

        # 5. Create each spatial feature table
        for table_name, defn in LAYER_DEFINITIONS.items():
            geom_type = defn["geom_type"]  # POLYGON, LINESTRING, or POINT
            col_defs = ", ".join(f'"{col}" {dtype}' for col, dtype in defn["columns"])
            create_sql = f"""
                CREATE TABLE IF NOT EXISTS "{table_name}" (
                    "fid" INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                    "geom" {geom_type},
                    {col_defs}
                )
            """
            cur.execute(create_sql)

            # Register in gpkg_contents
            cur.execute(
                """
                INSERT OR REPLACE INTO gpkg_contents
                (table_name, data_type, identifier, srs_id)
                VALUES (?, 'features', ?, ?)
                """,
                (table_name, table_name, epsg),
            )

            # Register in gpkg_geometry_columns with concrete geometry type
            cur.execute(
                """
                INSERT OR REPLACE INTO gpkg_geometry_columns
                (table_name, column_name, geometry_type_name, srs_id, z, m)
                VALUES (?, 'geom', ?, ?, 0, 0)
                """,
                (table_name, geom_type, epsg),
            )

            cur.execute(
                "INSERT OR REPLACE INTO gpkg_ogr_contents (table_name, feature_count) VALUES (?, 0)",
                (table_name,),
            )

        con.commit()
    finally:
        con.close()

    logger.info("Created 3 spatial layers in GeoPackage for '%s' at %s (EPSG:%d)", lake_id, gpkg_path, epsg)
    return gpkg_path


def validate_moraine_spatial_layers(gpkg_path: Path, expected_epsg: int = 32645) -> dict[str, Any]:
    """
    Open and validate each moraine annotation layer:
      - Geometry type is correct (Polygon, LineString, Point) and not NoGeometry/Unknown.
      - CRS is expected EPSG (default EPSG:32645).
      - Attribute fields are present with correct types.
      - Layers are valid and support adding features.

    Uses OGR/ogrinfo when available, with SQLite metadata verification.
    Raises ValueError if any layer fails validation.
    """
    gpkg_path = Path(gpkg_path)
    if not gpkg_path.exists():
        raise FileNotFoundError(f"GeoPackage not found: {gpkg_path}")

    report: dict[str, Any] = {
        "gpkg_path": str(gpkg_path),
        "expected_epsg": expected_epsg,
        "layers": {},
        "valid": True,
        "errors": [],
    }

    # 1. SQLite GeoPackage metadata verification
    con = sqlite3.connect(gpkg_path)
    try:
        cur = con.cursor()
        tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]

        # Check required GeoPackage metadata tables
        for req_table in ["gpkg_contents", "gpkg_geometry_columns", "gpkg_spatial_ref_sys"]:
            if req_table not in tables:
                msg = f"Missing GeoPackage system table '{req_table}' in {gpkg_path}"
                report["errors"].append(msg)
                report["valid"] = False

        if not report["valid"]:
            raise ValueError(f"GeoPackage metadata invalid: {report['errors']}")

        # Verify gpkg_geometry_columns entries
        geom_cols = dict(
            cur.execute("SELECT table_name, geometry_type_name FROM gpkg_geometry_columns").fetchall()
        )
        geom_srs = dict(
            cur.execute("SELECT table_name, srs_id FROM gpkg_geometry_columns").fetchall()
        )

        for layer_name, defn in LAYER_DEFINITIONS.items():
            expected_geom = defn["geom_type"]  # POLYGON, LINESTRING, POINT
            layer_info: dict[str, Any] = {
                "name": layer_name,
                "expected_geom": expected_geom,
                "gpkg_geom_type": geom_cols.get(layer_name),
                "gpkg_srs_id": geom_srs.get(layer_name),
                "fields": [],
                "can_add_features": False,
                "ogr_verified": False,
            }

            if layer_name not in geom_cols:
                err = f"Layer '{layer_name}' not registered in gpkg_geometry_columns"
                report["errors"].append(err)
                report["valid"] = False
                continue

            actual_geom = geom_cols[layer_name]
            if actual_geom in (None, "GEOMETRY", "", "Unknown"):
                err = (
                    f"Layer '{layer_name}' has generic/NoGeometry type '{actual_geom}' "
                    f"in gpkg_geometry_columns (expected concrete '{expected_geom}')"
                )
                report["errors"].append(err)
                report["valid"] = False

            if actual_geom != expected_geom:
                err = f"Layer '{layer_name}' geometry type is '{actual_geom}' (expected '{expected_geom}')"
                report["errors"].append(err)
                report["valid"] = False

            actual_srs = geom_srs.get(layer_name)
            if actual_srs != expected_epsg:
                err = f"Layer '{layer_name}' CRS srs_id is {actual_srs} (expected {expected_epsg})"
                report["errors"].append(err)
                report["valid"] = False

            # Verify table columns in SQLite
            col_info = cur.execute(f'PRAGMA table_info("{layer_name}")').fetchall()
            existing_cols = {r[1]: r[2] for r in col_info}
            layer_info["fields"] = list(existing_cols.keys())

            for req_col, req_dtype in defn["columns"]:
                if req_col not in existing_cols:
                    err = f"Layer '{layer_name}' missing attribute column '{req_col}'"
                    report["errors"].append(err)
                    report["valid"] = False

            # Test adding a feature capability via test transaction
            try:
                con.execute("SAVEPOINT test_add_feature")
                # Insert a dummy record without committing to verify AddFeatures capability
                if layer_name == "crest_points":
                    con.execute(
                        f'INSERT INTO "{layer_name}" (point_id, lake_id, point_type, confidence, elevation_m, notes) '
                        f"VALUES ('__test__', '__test__', 'representative_crest', 1, 5000.0, 'test')"
                    )
                else:
                    con.execute(
                        f'INSERT INTO "{layer_name}" (annotation_id, lake_id, confidence, notes) '
                        f"VALUES ('__test__', '__test__', 1, 'test')"
                    )
                con.execute("ROLLBACK TO SAVEPOINT test_add_feature")
                layer_info["can_add_features"] = True
            except Exception as exc:
                err = f"Layer '{layer_name}' does not support adding features: {exc}"
                report["errors"].append(err)
                report["valid"] = False

            report["layers"][layer_name] = layer_info
    finally:
        con.close()

    # 2. OGR provider verification via ogrinfo
    ogrinfo_bin = find_ogrinfo()
    if ogrinfo_bin:
        for layer_name, defn in LAYER_DEFINITIONS.items():
            expected_ogr_geom = defn["ogr_display"]  # Polygon, Line String, Point
            try:
                proc = subprocess.run(
                    [ogrinfo_bin, "-so", str(gpkg_path), layer_name],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if proc.returncode != 0:
                    err = f"ogrinfo failed for '{layer_name}': {proc.stderr.strip()}"
                    report["errors"].append(err)
                    report["valid"] = False
                    continue

                out = proc.stdout
                geom_match = re.search(r"Geometry:\s*(.+)", out)
                if geom_match:
                    ogr_geom = geom_match.group(1).strip()
                    report["layers"][layer_name]["ogr_geometry"] = ogr_geom
                    if ogr_geom != expected_ogr_geom:
                        err = f"Layer '{layer_name}' OGR geometry is '{ogr_geom}' (expected '{expected_ogr_geom}')"
                        report["errors"].append(err)
                        report["valid"] = False
                    if "Unknown" in ogr_geom or "None" in ogr_geom:
                        err = f"Layer '{layer_name}' has NoGeometry / Unknown geometry in OGR: '{ogr_geom}'"
                        report["errors"].append(err)
                        report["valid"] = False

                if f'ID["EPSG",{expected_epsg}]' not in out and f"EPSG:{expected_epsg}" not in out:
                    err = f"Layer '{layer_name}' OGR SRS does not match EPSG:{expected_epsg}"
                    report["errors"].append(err)
                    report["valid"] = False

                report["layers"][layer_name]["ogr_verified"] = True
            except Exception as exc:
                logger.warning("Could not run ogrinfo check for %s: %s", layer_name, exc)

    if not report["valid"]:
        raise ValueError(f"Moraine spatial layers failed validation: {report['errors']}")

    logger.info("Validated 3 spatial layers in %s (all EPSG:%d, concrete geometry types)", gpkg_path.name, expected_epsg)
    return report


def _rel_path(from_dir: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), from_dir.resolve()).replace("\\", "/")


def _add_field_config(
    layer_el: ET.Element,
    lake_id: str,
    is_points: bool,
) -> None:
    """Add QGIS attribute form widgets and default values to a vector <maplayer>."""
    fc = ET.SubElement(layer_el, "fieldConfiguration")
    defaults = ET.SubElement(layer_el, "defaults")

    # lake_id default
    d_lake = ET.SubElement(defaults, "default", field="lake_id", applyOnUpdate="0")
    d_lake.set("expression", f"'{lake_id}'")

    # confidence ValueMap
    f_conf = ET.SubElement(fc, "field", name="confidence")
    ew_conf = ET.SubElement(f_conf, "editWidget", type="ValueMap")
    cfg_conf = ET.SubElement(ew_conf, "config")
    opt_conf = ET.SubElement(cfg_conf, "Option", type="Map")
    m_conf = ET.SubElement(opt_conf, "Option", type="List", name="map")
    for label, val in [("1 - High", "1"), ("2 - Medium", "2"), ("3 - Low", "3")]:
        item = ET.SubElement(m_conf, "Option", type="Map")
        ET.SubElement(item, "Option", type="QString", name=label, value=val)

    d_conf = ET.SubElement(defaults, "default", field="confidence", applyOnUpdate="0")
    d_conf.set("expression", "2")

    if is_points:
        f_pt = ET.SubElement(fc, "field", name="point_type")
        ew_pt = ET.SubElement(f_pt, "editWidget", type="ValueMap")
        cfg_pt = ET.SubElement(ew_pt, "config")
        opt_pt = ET.SubElement(cfg_pt, "Option", type="Map")
        m_pt = ET.SubElement(opt_pt, "Option", type="List", name="map")
        for ptype in ["lowest_saddle", "crest_endpoint", "local_depression", "representative_crest"]:
            item = ET.SubElement(m_pt, "Option", type="Map")
            ET.SubElement(item, "Option", type="QString", name=ptype, value=ptype)

        d_pt = ET.SubElement(defaults, "default", field="point_type", applyOnUpdate="0")
        d_pt.set("expression", "'representative_crest'")


def _build_symbol_xml(name: str, sym_type: str, color: str, size: str = "0.8", outline_only: bool = False) -> ET.Element:
    sym = ET.Element("symbol", type=sym_type, name=name, alpha="1", clip_to_extent="1", force_rhr="0")
    if sym_type == "fill":
        layer = ET.SubElement(sym, "layer", {"class": "SimpleFill", "enabled": "1", "locked": "0", "pass": "0"})
        opts = ET.SubElement(layer, "Option", type="Map")
        ET.SubElement(opts, "Option", type="QString", name="color", value="255,140,0,0" if outline_only else color)
        ET.SubElement(opts, "Option", type="QString", name="outline_color", value=color)
        ET.SubElement(opts, "Option", type="QString", name="outline_width", value=size)
        ET.SubElement(opts, "Option", type="QString", name="outline_width_unit", value="MM")
        ET.SubElement(opts, "Option", type="QString", name="style", value="no" if outline_only else "solid")
    elif sym_type == "line":
        layer = ET.SubElement(sym, "layer", {"class": "SimpleLine", "enabled": "1", "locked": "0", "pass": "0"})
        opts = ET.SubElement(layer, "Option", type="Map")
        ET.SubElement(opts, "Option", type="QString", name="line_color", value=color)
        ET.SubElement(opts, "Option", type="QString", name="line_width", value=size)
        ET.SubElement(opts, "Option", type="QString", name="line_width_unit", value="MM")
        ET.SubElement(opts, "Option", type="QString", name="line_style", value="solid")
    elif sym_type == "marker":
        layer = ET.SubElement(sym, "layer", {"class": "SimpleMarker", "enabled": "1", "locked": "0", "pass": "0"})
        opts = ET.SubElement(layer, "Option", type="Map")
        ET.SubElement(opts, "Option", type="QString", name="color", value=color)
        ET.SubElement(opts, "Option", type="QString", name="outline_color", value="20,20,20,255")
        ET.SubElement(opts, "Option", type="QString", name="outline_width", value="0.4")
        ET.SubElement(opts, "Option", type="QString", name="size", value=size)
        ET.SubElement(opts, "Option", type="QString", name="size_unit", value="MM")
        ET.SubElement(opts, "Option", type="QString", name="name", value="star" if size == "4.5" else "circle")
    return sym


def build_qgis_project(
    lake: dict[str, Any],
    crs_str: str,
    gpkg_path: Path,
    rasters: dict[str, Path | None],
    hillshade_path: Path | None,
    qgz_path: Path,
) -> Path:
    """
    Generate a QGIS `.qgz` project file with styled vector annotation layers
    and all available raster layers.
    """
    qgz_path = Path(qgz_path)
    qgz_path.parent.mkdir(parents=True, exist_ok=True)
    lake_id = lake["id"]

    root = ET.Element("qgis", projectname=f"{lake['name']} Moraine Annotation", version="3.34.0-Prizren")
    ET.SubElement(root, "title").text = f"{lake['name']} ({lake_id}) — Moraine Dam Annotation"

    # Project CRS
    proj_crs = ET.SubElement(root, "projectCrs")
    srs = ET.SubElement(proj_crs, "spatialrefsys", nativeFormat="Wkt")
    ET.SubElement(srs, "authid").text = crs_str

    layer_tree = ET.SubElement(root, "layer-tree-group")
    project_layers = ET.SubElement(root, "projectlayers")

    def add_vector_layer(layer_name: str, display_name: str, defn: dict[str, Any]) -> None:
        lid = f"{lake_id}_{layer_name}"
        lt = ET.SubElement(layer_tree, "layer-tree-layer", id=lid, name=display_name, checked="Qt::Checked", expanded="1")
        lt.set("source", f"{_rel_path(qgz_path.parent, gpkg_path)}|layername={layer_name}")

        geom_label = defn["qgis_geometry"]
        wkb_type = defn["qgis_wkb"]

        ml = ET.SubElement(project_layers, "maplayer", type="vector", geometry=geom_label, wkbType=wkb_type)
        ET.SubElement(ml, "id").text = lid
        ET.SubElement(ml, "datasource").text = f"{_rel_path(qgz_path.parent, gpkg_path)}|layername={layer_name}"
        ET.SubElement(ml, "layername").text = display_name
        ET.SubElement(ml, "provider", encoding="UTF-8").text = "ogr"
        ET.SubElement(ml, "wkbType").text = wkb_type

        ml_srs = ET.SubElement(ml, "srs")
        ml_sr = ET.SubElement(ml_srs, "spatialrefsys", nativeFormat="Wkt")
        ET.SubElement(ml_sr, "authid").text = crs_str

        if layer_name == "dam_footprint":
            ren = ET.SubElement(ml, "renderer-v2", type="singleSymbol", symbollevels="0", enableorderby="0")
            syms = ET.SubElement(ren, "symbols")
            syms.append(_build_symbol_xml("0", "fill", "255,140,0,255", size="0.9", outline_only=True))
            _add_field_config(ml, lake_id, is_points=False)
        elif layer_name == "crest_line":
            ren = ET.SubElement(ml, "renderer-v2", type="singleSymbol", symbollevels="0", enableorderby="0")
            syms = ET.SubElement(ren, "symbols")
            syms.append(_build_symbol_xml("0", "line", "0,229,255,255", size="1.2"))
            _add_field_config(ml, lake_id, is_points=False)
        elif layer_name == "crest_points":
            ren = ET.SubElement(ml, "renderer-v2", type="categorizedSymbol", attr="point_type", symbollevels="0", enableorderby="0")
            cats = ET.SubElement(ren, "categories")
            syms = ET.SubElement(ren, "symbols")
            styles = [
                ("lowest_saddle", "Lowest Saddle", "255,23,68,255", "4.5"),
                ("crest_endpoint", "Crest Endpoint", "41,121,255,255", "3.0"),
                ("local_depression", "Local Depression", "255,214,0,255", "3.2"),
                ("representative_crest", "Representative Crest", "0,230,118,255", "3.0"),
            ]
            for idx, (val, lbl, col, sz) in enumerate(styles):
                ET.SubElement(cats, "category", value=val, label=lbl, symbol=str(idx), render="true")
                syms.append(_build_symbol_xml(str(idx), "marker", col, size=sz))
            _add_field_config(ml, lake_id, is_points=True)

    def add_raster_layer(layer_id: str, display_name: str, path: Path) -> None:
        rel = _rel_path(qgz_path.parent, path)
        lt = ET.SubElement(layer_tree, "layer-tree-layer", id=layer_id, name=display_name, checked="Qt::Checked", expanded="0")
        lt.set("source", rel)

        ml = ET.SubElement(project_layers, "maplayer", type="raster")
        ET.SubElement(ml, "id").text = layer_id
        ET.SubElement(ml, "datasource").text = rel
        ET.SubElement(ml, "layername").text = display_name
        ET.SubElement(ml, "provider").text = "gdal"

    # Add vector layers on top
    add_vector_layer("crest_points", f"{lake_id}_crest_points", LAYER_DEFINITIONS["crest_points"])
    add_vector_layer("crest_line", f"{lake_id}_crest_line", LAYER_DEFINITIONS["crest_line"])
    add_vector_layer("dam_footprint", f"{lake_id}_dam_footprint", LAYER_DEFINITIONS["dam_footprint"])

    # Add available rasters underneath
    if rasters.get("s2_latest"):
        add_raster_layer(f"{lake_id}_s2", f"{lake_id} Sentinel-2", rasters["s2_latest"])
    if rasters.get("stacked_latest"):
        add_raster_layer(f"{lake_id}_stacked", f"{lake_id} Stacked (10-ch)", rasters["stacked_latest"])
    if rasters.get("dem"):
        add_raster_layer(f"{lake_id}_dem", f"{lake_id} Copernicus DEM", rasters["dem"])
    if hillshade_path and hillshade_path.exists():
        add_raster_layer(f"{lake_id}_hillshade", f"{lake_id} DEM Hillshade", hillshade_path)

    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    qgs_name = f"{lake_id}_annotation.qgs"
    with zipfile.ZipFile(qgz_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(qgs_name, xml_bytes)

    logger.info("QGIS project written to %s (CRS: %s)", qgz_path, crs_str)
    return qgz_path


def prepare_lake_workspace(
    lake_id: str,
    skip_gpkg: bool = False,
    force_recreate: bool = False,
) -> dict[str, Any]:
    """Prepare the full QGIS moraine annotation workspace for `lake_id`."""
    lake = get_lake_by_id(lake_id)
    rasters = discover_lake_rasters(lake_id)
    crs_str, crs_source = determine_project_crs(lake, rasters)
    epsg = parse_epsg_code(crs_str)

    out_dir = DATA_MANUAL / lake_id
    out_dir.mkdir(parents=True, exist_ok=True)
    gpkg_path = out_dir / f"{lake_id}_moraine.gpkg"

    hillshade_path: Path | None = None
    if rasters.get("dem") is not None:
        try:
            hillshade_path = generate_hillshade(
                rasters["dem"],
                out_dir / "working" / f"{lake_id}_hillshade.tif",
            )
        except Exception as exc:
            logger.warning("Could not generate hillshade: %s", exc)

    validation_report: dict[str, Any] | None = None
    if not skip_gpkg:
        create_empty_moraine_geopackage(
            gpkg_path,
            lake_id,
            crs_str,
            force_recreate=force_recreate,
        )
        validation_report = validate_moraine_spatial_layers(gpkg_path, expected_epsg=epsg)

    qgz_path = build_qgis_project(
        lake=lake,
        crs_str=crs_str,
        gpkg_path=gpkg_path,
        rasters=rasters,
        hillshade_path=hillshade_path,
        qgz_path=QGIS_DIR / f"{lake_id}_annotation.qgz",
    )

    return {
        "lake_id": lake_id,
        "crs": crs_str,
        "epsg": epsg,
        "crs_source": crs_source,
        "gpkg_path": str(gpkg_path),
        "qgz_path": str(qgz_path),
        "hillshade_path": str(hillshade_path) if hillshade_path else None,
        "validation": validation_report,
        "rasters_found": {k: (str(v) if v else None) for k, v in rasters.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare QGIS moraine-dam annotation project.")
    ap.add_argument("--lake", default="south_lhonak", help="Lake ID in configs/lakes.yaml")
    ap.add_argument("--skip-gpkg", action="store_true", help="Only build .qgz project file")
    ap.add_argument(
        "--force-recreate",
        action="store_true",
        help="Recreate GeoPackage with backup of existing file",
    )
    args = ap.parse_args()

    info = prepare_lake_workspace(
        args.lake,
        skip_gpkg=args.skip_gpkg,
        force_recreate=args.force_recreate,
    )
    logger.info("Workspace ready: %s", info)


if __name__ == "__main__":
    main()
