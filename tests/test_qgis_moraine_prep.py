"""Tests for QGIS moraine project preparation and spatial layer validation."""

from __future__ import annotations

import sqlite3
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from src.labeling.qgis_moraine_prep import (
    build_qgis_project,
    create_empty_moraine_geopackage,
    validate_moraine_spatial_layers,
)
from src.utils.config_loader import get_lake_by_id


def test_create_moraine_gpkg_has_spatial_layers_and_not_nogeometry(tmp_path: Path) -> None:
    """Verify that GeoPackage creates concrete spatial layers and avoids NoGeometry."""
    gpkg_path = tmp_path / "south_lhonak_moraine.gpkg"
    create_empty_moraine_geopackage(gpkg_path, lake_id="south_lhonak", crs_str="EPSG:32645")

    assert gpkg_path.exists()

    con = sqlite3.connect(gpkg_path)
    cur = con.cursor()
    geom_cols = dict(cur.execute("SELECT table_name, geometry_type_name FROM gpkg_geometry_columns").fetchall())
    geom_srs = dict(cur.execute("SELECT table_name, srs_id FROM gpkg_geometry_columns").fetchall())
    contents = dict(cur.execute("SELECT table_name, data_type FROM gpkg_contents").fetchall())
    con.close()

    # Geometry types must be concrete and never GEOMETRY/NoGeometry
    assert geom_cols["dam_footprint"] == "POLYGON"
    assert geom_cols["crest_line"] == "LINESTRING"
    assert geom_cols["crest_points"] == "POINT"

    for layer_name, geom_type in geom_cols.items():
        assert geom_type not in (None, "GEOMETRY", "Unknown", ""), f"{layer_name} has invalid geometry type {geom_type}"
        assert geom_srs[layer_name] == 32645, f"{layer_name} has invalid SRS ID {geom_srs[layer_name]}"
        assert contents[layer_name] == "features", f"{layer_name} data_type is not 'features'"

    # Validation function must pass without errors
    report = validate_moraine_spatial_layers(gpkg_path, expected_epsg=32645)
    assert report["valid"] is True
    assert report["errors"] == []
    for lyr in ["dam_footprint", "crest_line", "crest_points"]:
        assert report["layers"][lyr]["can_add_features"] is True


def test_regression_catches_nogeometry_and_generic_geometry(tmp_path: Path) -> None:
    """Regression test: ensure validation detects and rejects NoGeometry/GEOMETRY layers."""
    gpkg_path = tmp_path / "corrupted_nogeometry.gpkg"
    create_empty_moraine_geopackage(gpkg_path, lake_id="south_lhonak", crs_str="EPSG:32645")

    # Simulate the bug where layers are registered as generic GEOMETRY (causing NoGeometry in QGIS)
    con = sqlite3.connect(gpkg_path)
    con.execute("UPDATE gpkg_geometry_columns SET geometry_type_name='GEOMETRY' WHERE table_name='dam_footprint'")
    con.commit()
    con.close()

    with pytest.raises(ValueError) as excinfo:
        validate_moraine_spatial_layers(gpkg_path, expected_epsg=32645)

    err_msg = str(excinfo.value)
    assert "generic/NoGeometry type 'GEOMETRY'" in err_msg or "dam_footprint" in err_msg


def test_backup_created_when_replacing_existing_gpkg(tmp_path: Path) -> None:
    """Verify that an explicit backup is created when replacing an existing GeoPackage."""
    gpkg_path = tmp_path / "south_lhonak_moraine.gpkg"
    create_empty_moraine_geopackage(gpkg_path, lake_id="south_lhonak", crs_str="EPSG:32645")

    # Force recreate to trigger backup
    create_empty_moraine_geopackage(gpkg_path, lake_id="south_lhonak", crs_str="EPSG:32645", force_recreate=True)

    backups = list(tmp_path.glob("south_lhonak_moraine.backup_*.gpkg"))
    assert len(backups) >= 1
    assert backups[0].exists()


def test_build_qgis_project_writes_valid_qgz(tmp_path: Path) -> None:
    lake = get_lake_by_id("south_lhonak")
    gpkg_path = tmp_path / "data" / "labels" / "manual" / "south_lhonak" / "south_lhonak_moraine.gpkg"
    gpkg_path.parent.mkdir(parents=True, exist_ok=True)
    qgz_path = tmp_path / "qgis" / "south_lhonak_annotation.qgz"

    out = build_qgis_project(
        lake=lake,
        crs_str="EPSG:32645",
        gpkg_path=gpkg_path,
        rasters={},
        hillshade_path=None,
        qgz_path=qgz_path,
    )

    assert out.exists()
    with zipfile.ZipFile(out, "r") as zf:
        names = zf.namelist()
        assert "south_lhonak_annotation.qgs" in names
        xml_bytes = zf.read("south_lhonak_annotation.qgs")

    root = ET.fromstring(xml_bytes)
    assert root.findtext("projectCrs/spatialrefsys/authid") == "EPSG:32645"
    layer_names = [el.text for el in root.findall("projectlayers/maplayer/layername")]
    assert "south_lhonak_dam_footprint" in layer_names
    assert "south_lhonak_crest_line" in layer_names
    assert "south_lhonak_crest_points" in layer_names

    # Check wkbType elements
    wkb_types = [el.text for el in root.findall("projectlayers/maplayer/wkbType")]
    assert "Polygon" in wkb_types
    assert "LineString" in wkb_types
    assert "Point" in wkb_types
