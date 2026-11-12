"""Tests for moraine GeoPackage validation and export (src/labeling/moraine_validation.py)."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point, Polygon

from src.labeling.moraine_validation import (
    apply_to_lakes_yaml,
    format_lakes_yaml_block,
    validate_moraine_geopackage,
)


def _write_moraine_gpkg(
    path: Path,
    dam_gdf: gpd.GeoDataFrame | None = None,
    line_gdf: gpd.GeoDataFrame | None = None,
    pts_gdf: gpd.GeoDataFrame | None = None,
) -> Path:
    crs = "EPSG:32645"
    if dam_gdf is None:
        # 100 m x 50 m rectangle -> area = 5000 m^2
        dam_gdf = gpd.GeoDataFrame(
            {
                "annotation_id": ["dam_1"],
                "lake_id": ["south_lhonak"],
                "confidence": [1],
                "notes": ["test dam"],
            },
            geometry=[Polygon([(617000, 3088000), (617100, 3088000), (617100, 3088050), (617000, 3088050)])],
            crs=crs,
        )
    if line_gdf is None:
        # 100 m line -> length = 100 m
        line_gdf = gpd.GeoDataFrame(
            {
                "annotation_id": ["crest_1"],
                "lake_id": ["south_lhonak"],
                "confidence": [1],
                "notes": ["test crest"],
            },
            geometry=[LineString([(617000, 3088025), (617100, 3088025)])],
            crs=crs,
        )
    if pts_gdf is None:
        pts_gdf = gpd.GeoDataFrame(
            {
                "point_id": ["p1", "p2", "p3"],
                "lake_id": ["south_lhonak", "south_lhonak", "south_lhonak"],
                "point_type": ["crest_endpoint", "lowest_saddle", "representative_crest"],
                "confidence": [1, 1, 2],
                "elevation_m": [5210.0, 5195.0, 5205.0],
                "notes": ["end", "saddle", "rep"],
            },
            geometry=[
                Point(617000, 3088025),
                Point(617050, 3088025),
                Point(617100, 3088025),
            ],
            crs=crs,
        )

    dam_gdf.to_file(path, layer="dam_footprint", driver="GPKG")
    line_gdf.to_file(path, layer="crest_line", driver="GPKG")
    pts_gdf.to_file(path, layer="crest_points", driver="GPKG")
    return path


def test_valid_moraine_gpkg_metrics_and_yaml(tmp_path: Path) -> None:
    gpkg = _write_moraine_gpkg(tmp_path / "valid.gpkg")
    report = validate_moraine_geopackage(gpkg, lake_id="south_lhonak")

    assert report["valid"] is True
    assert report["errors"] == []
    m = report["metrics"]
    assert m["dam_footprint_area_m2"] == pytest.approx(5000.0, abs=1.0)
    assert m["crest_line_length_m"] == pytest.approx(100.0, abs=0.5)
    assert m["crest_elevation_min_m"] == pytest.approx(5195.0)
    assert m["crest_elevation_max_m"] == pytest.approx(5210.0)
    assert m["crest_elevation_mean_m"] == pytest.approx(5203.333, abs=1e-2)
    assert m["lowest_saddle_elevation_m"] == pytest.approx(5195.0)
    assert m["num_crest_points"] == 3

    yaml_block = format_lakes_yaml_block(report)
    assert "crest_points:" in yaml_block
    assert "is_lowest: true" in yaml_block
    assert "lowest_saddle_elev_m: 5195.00" in yaml_block


def test_invalid_geometry_not_modified_and_flagged(tmp_path: Path) -> None:
    # Bowtie self-intersecting polygon
    bowtie = Polygon([(617000, 3088000), (617100, 3088050), (617100, 3088000), (617000, 3088050)])
    dam_gdf = gpd.GeoDataFrame(
        {"annotation_id": ["d1"], "lake_id": ["south_lhonak"], "confidence": [1], "notes": [""]},
        geometry=[bowtie],
        crs="EPSG:32645",
    )
    gpkg = _write_moraine_gpkg(tmp_path / "invalid_geom.gpkg", dam_gdf=dam_gdf)
    report = validate_moraine_geopackage(gpkg, lake_id="south_lhonak")

    assert report["valid"] is False
    assert any("invalid geometry" in e for e in report["errors"])


def test_duplicate_ids_missing_saddle_and_bad_elevation_flagged(tmp_path: Path) -> None:
    pts_gdf = gpd.GeoDataFrame(
        {
            "point_id": ["dup1", "dup1"],
            "lake_id": ["south_lhonak", "south_lhonak"],
            "point_type": ["crest_endpoint", "representative_crest"],
            "confidence": [1, 2],
            "elevation_m": [5200.0, float("nan")],
            "notes": ["", ""],
        },
        geometry=[Point(617000, 3088025), Point(617050, 3088025)],
        crs="EPSG:32645",
    )
    gpkg = _write_moraine_gpkg(tmp_path / "bad_pts.gpkg", pts_gdf=pts_gdf)
    report = validate_moraine_geopackage(gpkg, lake_id="south_lhonak")

    assert report["valid"] is False
    joined = "\n".join(report["errors"])
    assert "Duplicate point_id" in joined
    assert "Missing 'lowest_saddle'" in joined
    assert "Missing or invalid elevation_m" in joined


def test_apply_to_lakes_yaml_preserves_comments_and_other_lakes(tmp_path: Path) -> None:
    sample_yaml = tmp_path / "lakes.yaml"
    sample_yaml.write_text(
        "# Header comment\nlakes:\n"
        "  - id: south_lhonak\n"
        "    name: South Lhonak\n"
        "    crest_points: []          # fill in M7\n"
        "    dam:\n"
        "      height_m: null\n"
        "  - id: imja_tsho\n"
        "    name: Imja Tsho\n"
        "    crest_points: []\n",
        encoding="utf-8",
    )
    gpkg = _write_moraine_gpkg(tmp_path / "valid.gpkg")
    report = validate_moraine_geopackage(gpkg, lake_id="south_lhonak")
    apply_to_lakes_yaml(report, lakes_yaml_path=sample_yaml)

    updated = sample_yaml.read_text(encoding="utf-8")
    assert "# Header comment" in updated
    assert "point_type: lowest_saddle" in updated
    assert "id: imja_tsho" in updated
    assert "height_m: null" in updated
