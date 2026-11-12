"""
dem_sampling.py — Reusable DEM elevation extraction for manually annotated crest points.

Samples elevation in metres from a DEM GeoTIFF at each point in a GeoDataFrame:
- Transforms point coordinates into the DEM's CRS for sampling while preserving
  the original point geometries in the returned GeoDataFrame.
- Respects raster NoData values, NaNs, and out-of-bounds coordinates.
- Reports invalid samples explicitly rather than silently assigning zero.
- Records DEM source, CRS, resolution, and a non-surveyed disclaimer in metadata.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pyproj
import rasterio
from pyproj import Transformer

logger = logging.getLogger(__name__)

DEFAULT_DEM_SOURCE = "COPERNICUS/DEM/GLO30_2024_1"
DEM_DISCLAIMER = (
    "DEM-derived elevations are interpolated from remotely sensed elevation data "
    "(e.g. ~30 m native Copernicus GLO-30) and are NOT field-surveyed crest elevations."
)


def _estimate_dem_resolution_m(src: rasterio.DatasetReader) -> tuple[float | None, float | None]:
    """
    Return approximate (x_res_m, y_res_m) in metres for the DEM.
    If the DEM is in a geographic CRS, converts degree pixel size at the raster
    centre latitude to approximate metres.
    """
    res_x, res_y = abs(float(src.res[0])), abs(float(src.res[1]))
    if src.crs is None:
        return None, None
    if src.crs.is_projected:
        return round(res_x, 3), round(res_y, 3)

    # Geographic CRS (degrees) -> approximate metres at raster centre
    bounds = src.bounds
    mid_lat = 0.5 * (bounds.bottom + bounds.top)
    geod = pyproj.Geod(ellps="WGS84")
    _, _, dx_m = geod.inv(bounds.left, mid_lat, bounds.left + res_x, mid_lat)
    _, _, dy_m = geod.inv(bounds.left, mid_lat, bounds.left, mid_lat + res_y)
    return round(abs(dx_m), 3), round(abs(dy_m), 3)


def sample_dem_at_points(
    points_gdf: gpd.GeoDataFrame,
    dem_path: Path,
    elevation_col: str = "elevation_m",
    dem_source: str = DEFAULT_DEM_SOURCE,
    band: int = 1,
    raise_on_invalid: bool = False,
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """
    Sample DEM elevations (in metres) at each point geometry in `points_gdf`.

    Original point geometries and CRS are preserved in the returned GeoDataFrame.

    Args:
        points_gdf:       GeoDataFrame containing Point geometries.
        dem_path:         Path to the DEM GeoTIFF.
        elevation_col:    Output column name for sampled elevation in metres.
        dem_source:       Human-readable DEM source identifier for metadata.
        band:             1-based raster band index to sample (default 1).
        raise_on_invalid: If True, raise ValueError when any point fails sampling.

    Returns:
        (updated_gdf, metadata_dict)
    """
    dem_path = Path(dem_path)
    if not dem_path.exists():
        raise FileNotFoundError(f"DEM file not found: {dem_path}")
    if points_gdf.empty:
        raise ValueError("Cannot sample DEM: points GeoDataFrame is empty.")
    if points_gdf.crs is None:
        raise ValueError("points_gdf has no CRS defined; cannot transform to DEM CRS.")

    result = points_gdf.copy()
    invalid_records: list[dict[str, Any]] = []

    with rasterio.open(dem_path) as src:
        if src.crs is None:
            raise ValueError(f"DEM raster has no CRS defined: {dem_path}")
        if band < 1 or band > src.count:
            raise ValueError(f"Requested band {band} out of range [1, {src.count}] for {dem_path}")

        nodata = src.nodata
        bounds = src.bounds
        res_m_x, res_m_y = _estimate_dem_resolution_m(src)

        transformer = Transformer.from_crs(
            points_gdf.crs, src.crs, always_xy=True
        )

        coords_dem: list[tuple[float, float]] = []
        valid_geom_mask: list[bool] = []
        for idx, geom in zip(result.index, result.geometry):
            if geom is None or geom.is_empty:
                coords_dem.append((np.nan, np.nan))
                valid_geom_mask.append(False)
                invalid_records.append({
                    "index": str(idx),
                    "point_id": str(result.at[idx, "point_id"]) if "point_id" in result.columns else str(idx),
                    "reason": "empty_or_null_geometry",
                })
                continue
            if geom.geom_type != "Point":
                coords_dem.append((np.nan, np.nan))
                valid_geom_mask.append(False)
                invalid_records.append({
                    "index": str(idx),
                    "point_id": str(result.at[idx, "point_id"]) if "point_id" in result.columns else str(idx),
                    "reason": f"unexpected_geometry_type:{geom.geom_type}",
                })
                continue
            x_dem, y_dem = transformer.transform(geom.x, geom.y)
            coords_dem.append((x_dem, y_dem))
            valid_geom_mask.append(True)

        # Sample all valid coordinates in one pass
        sampled_values: list[float] = []
        valid_coords = [c for c, ok in zip(coords_dem, valid_geom_mask) if ok]
        raw_samples = list(src.sample(valid_coords, indexes=band)) if valid_coords else []
        raw_iter = iter(raw_samples)

        for idx, (x_dem, y_dem), ok in zip(result.index, coords_dem, valid_geom_mask):
            point_id = str(result.at[idx, "point_id"]) if "point_id" in result.columns else str(idx)
            if not ok:
                sampled_values.append(float("nan"))
                continue

            val_arr = next(raw_iter)
            val = float(val_arr[0])

            in_bounds = (
                bounds.left <= x_dem <= bounds.right
                and bounds.bottom <= y_dem <= bounds.top
            )
            if not in_bounds:
                sampled_values.append(float("nan"))
                invalid_records.append({
                    "index": str(idx),
                    "point_id": point_id,
                    "reason": "outside_dem_bounds",
                    "dem_x": x_dem,
                    "dem_y": y_dem,
                })
                continue

            is_nodata = (
                not np.isfinite(val)
                or (nodata is not None and np.isclose(val, float(nodata)))
                or np.isclose(val, -9999.0)
            )
            if is_nodata:
                sampled_values.append(float("nan"))
                invalid_records.append({
                    "index": str(idx),
                    "point_id": point_id,
                    "reason": "dem_nodata_or_nan",
                    "raw_value": None if not np.isfinite(val) else val,
                })
                continue

            sampled_values.append(round(val, 3))

        result[elevation_col] = sampled_values

        metadata: dict[str, Any] = {
            "dem_path": str(dem_path),
            "dem_source": dem_source,
            "dem_crs": str(src.crs),
            "dem_native_resolution": [float(src.res[0]), float(src.res[1])],
            "dem_resolution_m": [res_m_x, res_m_y],
            "dem_nodata": None if nodata is None else float(nodata),
            "band": band,
            "units": "metres",
            "total_points": int(len(result)),
            "valid_samples": int(len(result) - len(invalid_records)),
            "invalid_samples": invalid_records,
            "surveyed": False,
            "disclaimer": DEM_DISCLAIMER,
        }

    if invalid_records:
        logger.warning(
            "DEM sampling encountered %d invalid sample(s) out of %d points.",
            len(invalid_records),
            len(result),
        )
        if raise_on_invalid:
            raise ValueError(
                f"DEM sampling failed for {len(invalid_records)} point(s): {invalid_records}"
            )

    return result, metadata
