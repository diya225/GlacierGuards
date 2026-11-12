"""Tests for DEM elevation sampling utility (src/labeling/dem_sampling.py)."""

from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from shapely.geometry import Point

from src.labeling.dem_sampling import sample_dem_at_points


def _write_test_dem(path: Path, crs: str = "EPSG:4326", nodata: float = -9999.0) -> Path:
    """Create a 10x10 synthetic DEM GeoTIFF with known elevations and one NoData cell."""
    data = np.full((10, 10), 5100.0, dtype=np.float32)
    data[0, 0] = 5150.5
    data[5, 5] = nodata  # NoData cell

    if crs == "EPSG:4326":
        # 0.001 deg (~100 m) pixels near 27.91N, 88.20E
        transform = from_origin(88.190, 27.920, 0.001, 0.001)
    else:
        transform = from_origin(617000.0, 3089000.0, 10.0, 10.0)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)
    return path


def test_sample_dem_transforms_crs_and_preserves_geometry(tmp_path: Path) -> None:
    dem_path = _write_test_dem(tmp_path / "dem_4326.tif", crs="EPSG:4326")

    # Pixel (row=0, col=0) centre in EPSG:4326 is (88.1905, 27.9195)
    # Pixel (row=2, col=3) centre in EPSG:4326 is (88.1935, 27.9175)
    t = Transformer.from_crs("EPSG:4326", "EPSG:32645", always_xy=True)
    x0, y0 = t.transform(88.1905, 27.9195)
    x1, y1 = t.transform(88.1935, 27.9175)

    pts = gpd.GeoDataFrame(
        {"point_id": ["p1", "p2"]},
        geometry=[Point(x0, y0), Point(x1, y1)],
        crs="EPSG:32645",
    )

    out, meta = sample_dem_at_points(pts, dem_path, dem_source="TEST_DEM")

    assert out.crs == pts.crs
    assert out.geometry.iloc[0].equals(pts.geometry.iloc[0])
    assert out.geometry.iloc[1].equals(pts.geometry.iloc[1])
    assert out["elevation_m"].iloc[0] == pytest.approx(5150.5, abs=1e-2)
    assert out["elevation_m"].iloc[1] == pytest.approx(5100.0, abs=1e-2)
    assert meta["valid_samples"] == 2
    assert meta["invalid_samples"] == []
    assert meta["units"] == "metres"
    assert meta["surveyed"] is False
    assert meta["dem_source"] == "TEST_DEM"
    assert meta["dem_resolution_m"][0] > 0


def test_sample_dem_reports_nodata_and_oob_without_zeroing(tmp_path: Path) -> None:
    dem_path = _write_test_dem(tmp_path / "dem_utm.tif", crs="EPSG:32645")

    # Pixel (row=5, col=5) is NoData; (600000, 3000000) is out of bounds
    pts = gpd.GeoDataFrame(
        {"point_id": ["ok", "nodata_pt", "oob_pt"]},
        geometry=[
            Point(617005.0, 3088995.0),
            Point(617055.0, 3088945.0),
            Point(600000.0, 3000000.0),
        ],
        crs="EPSG:32645",
    )

    out, meta = sample_dem_at_points(pts, dem_path, raise_on_invalid=False)

    assert out["elevation_m"].iloc[0] == pytest.approx(5150.5, abs=1e-2)
    assert math.isnan(out["elevation_m"].iloc[1])
    assert math.isnan(out["elevation_m"].iloc[2])
    assert meta["valid_samples"] == 1
    reasons = {r["point_id"]: r["reason"] for r in meta["invalid_samples"]}
    assert reasons["nodata_pt"] == "dem_nodata_or_nan"
    assert reasons["oob_pt"] == "outside_dem_bounds"

    with pytest.raises(ValueError):
        sample_dem_at_points(pts, dem_path, raise_on_invalid=True)
