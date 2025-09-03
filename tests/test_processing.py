import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box
import geopandas as gpd
import zipfile
import fiona
import pytest

from utils.processing import (
    align_crop_to_depth,
    process_flood_damage,
    polygon_mask_to_depth_array,
    constant_depth_array,
    drawn_features_to_depth_array,
    run_monte_carlo,
)
from utils.crop_definitions import CROP_DEFINITIONS


def create_raster(path, array, crs, transform):
    profile = {
        "driver": "GTiff",
        "height": array.shape[0],
        "width": array.shape[1],
        "count": 1,
        "dtype": array.dtype,
        "crs": crs,
        "transform": transform,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)


def test_align_crop_to_depth_crs_and_shape(tmp_path):
    crop_arr = np.array([[1, 2], [3, 4]], dtype=np.uint16)
    crop_path = tmp_path / "crop.tif"
    crop_transform = from_origin(0, 2, 1, 1)
    create_raster(crop_path, crop_arr, "EPSG:4326", crop_transform)

    depth_arr = np.ones((3, 3), dtype=np.float32)
    depth_path = tmp_path / "depth.tif"
    depth_transform = from_origin(0, 3000, 1000, 1000)
    create_raster(depth_path, depth_arr, "EPSG:3857", depth_transform)

    aligned, profile = align_crop_to_depth(str(crop_path), str(depth_path))

    assert aligned.shape == crop_arr.shape
    assert str(profile["crs"]) == "EPSG:3857"


def test_process_flood_damage_generates_outputs(tmp_path):
    crop = np.array([[1, 1, 2], [1, 2, 2], [1, 1, 1]], dtype=np.uint16)
    crop_path = tmp_path / "crop.tif"
    create_raster(crop_path, crop, "EPSG:4326", from_origin(0, 3, 1, 1))

    depth_arr = np.full((3, 3), 6.0, dtype=float)

    crop_inputs = {
        1: {"Value": 10, "GrowingSeason": [6]},
        2: {"Value": 20, "GrowingSeason": [6]},
    }
    flood_metadata = {"floodA": {"return_period": 10, "flood_month": 6}}

    out_dir = tmp_path / "out"
    excel_path, summaries, diagnostics, rasters = process_flood_damage(
        str(crop_path),
        [("floodA", depth_arr)],
        str(out_dir),
        100,
        crop_inputs,
        flood_metadata,
    )

    assert os.path.exists(excel_path)
    assert (out_dir / "damage_floodA.tif").exists()
    assert "floodA" in summaries
    df = summaries["floodA"]
    assert len(df) == 2
    assert "FloodedPixels" in df.columns
    assert diagnostics == []
    assert rasters["floodA"]["ratio"].shape == crop.shape
    assert set(np.unique(rasters["floodA"]["crop"])) == {1, 2}


def test_process_flood_damage_with_labeled_path(tmp_path):
    crop = np.array([[1, 2], [2, 1]], dtype=np.uint16)
    crop_path = tmp_path / "crop.tif"
    create_raster(crop_path, crop, "EPSG:4326", from_origin(0, 2, 1, 1))

    depth_arr = np.full((2, 2), 6.0, dtype=float)
    depth_path = tmp_path / "depthA.tif"
    create_raster(depth_path, depth_arr, "EPSG:4326", from_origin(0, 2, 1, 1))

    crop_inputs = {
        1: {"Value": 10, "GrowingSeason": [6]},
        2: {"Value": 20, "GrowingSeason": [6]},
    }
    flood_metadata = {"depthA": {"return_period": 10, "flood_month": 6}}

    out_dir = tmp_path / "out"
    excel_path, summaries, diagnostics, rasters = process_flood_damage(
        str(crop_path),
        [("depthA", str(depth_path))],
        str(out_dir),
        100,
        crop_inputs,
        flood_metadata,
    )

    assert (out_dir / "damage_depthA.tif").exists()
    assert "depthA" in summaries


def test_process_flood_damage_reports_all_crops(tmp_path):
    crop = np.array([[1, 1], [1, 1]], dtype=np.uint16)
    crop_path = tmp_path / "crop.tif"
    create_raster(crop_path, crop, "EPSG:4326", from_origin(0, 2, 1, 1))

    depth_arr = np.full((2, 2), 6.0, dtype=float)
    crop_inputs = {
        1: {"Value": 10, "GrowingSeason": [6]},
        2: {"Value": 20, "GrowingSeason": [7]},  # out of season and absent
    }
    flood_metadata = {"floodA": {"return_period": 10, "flood_month": 6}}

    out_dir = tmp_path / "out"
    excel_path, summaries, diagnostics, rasters = process_flood_damage(
        str(crop_path),
        [("floodA", depth_arr)],
        str(out_dir),
        100,
        crop_inputs,
        flood_metadata,
    )

    df = summaries["floodA"]
    assert set(df["CropCode"]) == {1, 2}
    row2 = df[df["CropCode"] == 2].iloc[0]
    assert row2["FloodedAcres"] == 0
    assert row2["FloodedPixels"] == 0
    assert row2["DollarsLost"] == 0
    assert any(d["CropCode"] == 2 for d in diagnostics)


def test_process_flood_damage_includes_names(tmp_path):
    crop = np.array([[1]], dtype=np.uint16)
    crop_path = tmp_path / "crop.tif"
    create_raster(crop_path, crop, "EPSG:4326", from_origin(0, 1, 1, 1))

    depth_arr = np.full((1, 1), 6.0, dtype=float)
    crop_inputs = {1: {"Value": 10, "GrowingSeason": [6]}}
    flood_metadata = {"floodA": {"return_period": 10, "flood_month": 6}}

    out_dir = tmp_path / "out"
    _, summaries, _, _ = process_flood_damage(
        str(crop_path), [("floodA", depth_arr)], str(out_dir), 100, crop_inputs, flood_metadata
    )

    df = summaries["floodA"]
    assert "CropName" in df.columns
    name = df[df["CropCode"] == 1]["CropName"].iloc[0]
    assert name == CROP_DEFINITIONS[1][0]


def test_process_flood_damage_includes_unlisted_crops(tmp_path):
    # Include a crop code (999) that's absent from CROP_DEFINITIONS to exercise
    # the fallback naming logic.
    crop = np.array([[1, 999]], dtype=np.uint16)
    crop_path = tmp_path / "crop.tif"
    create_raster(crop_path, crop, "EPSG:3857", from_origin(0, 1, 1, 1))

    depth_arr = np.full((1, 2), 6.0, dtype=float)
    crop_inputs = {1: {"Value": 10, "GrowingSeason": [6]}}  # crop 999 omitted
    flood_metadata = {"floodA": {"return_period": 10, "flood_month": 6}}

    out_dir = tmp_path / "out"
    _, summaries, _, _ = process_flood_damage(
        str(crop_path),
        [("floodA", depth_arr)],
        str(out_dir),
        100,
        crop_inputs,
        flood_metadata,
    )

    df = summaries["floodA"]
    assert set(df["CropCode"]) == {1, 999}
    # Crop code 999 was not supplied in crop_inputs or the definitions, so its
    # name should default to the string representation of the code rather than
    # being blank.
    name = df[df["CropCode"] == 999]["CropName"].iloc[0]
    assert name == "999"


def test_process_flood_damage_respects_user_values(tmp_path):
    crop = np.array([[3]], dtype=np.uint16)
    crop_path = tmp_path / "crop.tif"
    create_raster(crop_path, crop, "EPSG:4326", from_origin(0, 1, 1, 1))

    depth_arr = np.full((1, 1), 6.0, dtype=float)
    # Provide a custom value that should be preserved
    crop_inputs = {3: {"Value": 1200, "GrowingSeason": [6]}}
    flood_metadata = {"floodA": {"return_period": 10, "flood_month": 6}}

    out_dir = tmp_path / "out"
    _, summaries, _, _ = process_flood_damage(
        str(crop_path), [("floodA", depth_arr)], str(out_dir), 100, crop_inputs, flood_metadata
    )

    df = summaries["floodA"]
    val = df[df["CropCode"] == 3]["ValuePerAcre"].iloc[0]
    assert val == 1200


def test_pixel_to_acre_conversion(tmp_path):
    crop = np.array([[1]], dtype=np.uint16)
    pixel_size = 30  # meters
    crop_path = tmp_path / "crop.tif"
    create_raster(
        crop_path, crop, "EPSG:3857", from_origin(0, pixel_size, pixel_size, pixel_size)
    )

    depth_arr = np.full((1, 1), 6.0, dtype=float)
    crop_inputs = {1: {"Value": 10, "GrowingSeason": [6]}}
    flood_metadata = {"floodA": {"return_period": 10, "flood_month": 6}}

    out_dir = tmp_path / "out"
    _, summaries, _, _ = process_flood_damage(
        str(crop_path),
        [("floodA", depth_arr)],
        str(out_dir),
        100,
        crop_inputs,
        flood_metadata,
    )

    df = summaries["floodA"]
    expected = (pixel_size * pixel_size) / 4046.8564224
    flooded_pixels = df[df["CropCode"] == 1]["FloodedPixels"].iloc[0]
    flooded_acres = df[df["CropCode"] == 1]["FloodedAcres"].iloc[0]
    assert flooded_pixels == 1
    assert flooded_acres == pytest.approx(flooded_pixels * expected, rel=1e-3)


def test_pixel_to_acre_conversion_feet(tmp_path):
    crop = np.array([[1]], dtype=np.uint16)
    pixel_size_ft = 98.4252  # 30 meters expressed in US survey feet
    crop_path = tmp_path / "crop.tif"
    create_raster(
        crop_path, crop, "EPSG:2277", from_origin(0, pixel_size_ft, pixel_size_ft, pixel_size_ft)
    )

    depth_arr = np.full((1, 1), 6.0, dtype=float)
    crop_inputs = {1: {"Value": 10, "GrowingSeason": [6]}}
    flood_metadata = {"floodA": {"return_period": 10, "flood_month": 6}}

    out_dir = tmp_path / "out"
    _, summaries, _, _ = process_flood_damage(
        str(crop_path), [("floodA", depth_arr)], str(out_dir), 100, crop_inputs, flood_metadata
    )

    df = summaries["floodA"]
    unit_factor = rasterio.crs.CRS.from_epsg(2277).linear_units_factor[1]
    expected = ((pixel_size_ft * unit_factor) ** 2) / 4046.8564224
    flooded_pixels = df[df["CropCode"] == 1]["FloodedPixels"].iloc[0]
    flooded_acres = df[df["CropCode"] == 1]["FloodedAcres"].iloc[0]
    assert flooded_pixels == 1
    assert flooded_acres == pytest.approx(flooded_pixels * expected, rel=1e-3)


def test_process_flood_damage_excludes_zero_code(tmp_path):
    crop = np.array([[0, 1], [0, 1]], dtype=np.uint16)
    crop_path = tmp_path / "crop.tif"
    create_raster(crop_path, crop, "EPSG:4326", from_origin(0, 2, 1, 1))

    depth_arr = np.full((2, 2), 6.0, dtype=float)
    flood_metadata = {"flood": {"return_period": 10, "flood_month": 6}}

    out_dir = tmp_path / "out"
    _, summaries, _, rasters = process_flood_damage(
        str(crop_path), [("flood", depth_arr)], str(out_dir), 100, None, flood_metadata
    )

    df = summaries["flood"]
    assert set(df["CropCode"]) == {1}
    assert np.all(rasters["flood"]["ratio"][crop == 0] == 0)

def test_rasterize_polygon_zipped_shapefile(tmp_path):
    crop = np.zeros((10, 10), dtype=np.uint16)
    crop_path = tmp_path / "crop.tif"
    create_raster(crop_path, crop, "EPSG:4326", from_origin(0, 10, 1, 1))

    gdf = gpd.GeoDataFrame({"geometry": [box(2, 2, 5, 5)]}, crs="EPSG:4326")
    shp_dir = tmp_path / "shp"
    shp_dir.mkdir()
    gdf.to_file(shp_dir / "poly.shp")

    zip_path = tmp_path / "poly.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        for ext in ["shp", "shx", "dbf", "cpg", "prj"]:
            f = shp_dir / f"poly.{ext}"
            if f.exists():
                z.write(f, arcname=f"poly.{ext}")

    arr = polygon_mask_to_depth_array(str(zip_path), str(crop_path))
    assert arr.shape == crop.shape
    assert arr.max() == 0.5
    assert arr.sum() > 0


def test_rasterize_polygon_geojson(tmp_path):
    crop = np.zeros((10, 10), dtype=np.uint16)
    crop_path = tmp_path / "crop.tif"
    create_raster(crop_path, crop, "EPSG:4326", from_origin(0, 10, 1, 1))

    gdf = gpd.GeoDataFrame({"geometry": [box(1, 1, 4, 4)]}, crs="EPSG:4326")
    geojson_path = tmp_path / "poly.geojson"
    gdf.to_file(geojson_path, driver="GeoJSON")

    arr = polygon_mask_to_depth_array(str(geojson_path), str(crop_path))
    assert arr.shape == crop.shape
    assert arr.max() == 0.5
    assert arr.sum() > 0


def test_rasterize_polygon_kml(tmp_path):
    if "KML" not in fiona.supported_drivers:
        pytest.skip("KML driver not available")

    crop = np.zeros((10, 10), dtype=np.uint16)
    crop_path = tmp_path / "crop.tif"
    create_raster(crop_path, crop, "EPSG:4326", from_origin(0, 10, 1, 1))

    gdf = gpd.GeoDataFrame({"geometry": [box(3, 3, 6, 6)]}, crs="EPSG:4326")
    kml_path = tmp_path / "poly.kml"
    gdf.to_file(kml_path, driver="KML")

    arr = polygon_mask_to_depth_array(str(kml_path), str(crop_path))
    assert arr.shape == crop.shape
    assert arr.max() == 0.5
    assert arr.sum() > 0


def test_constant_depth_array(tmp_path):
    crop = np.zeros((5, 5), dtype=np.uint16)
    crop_path = tmp_path / "crop.tif"
    create_raster(crop_path, crop, "EPSG:4326", from_origin(0, 5, 1, 1))

    arr = constant_depth_array(str(crop_path), depth_value=0.5)

    assert arr.shape == crop.shape
    assert np.allclose(arr, 0.5)


def test_drawn_features_to_depth_array(tmp_path):
    crop = np.zeros((10, 10), dtype=np.uint16)
    crop_path = tmp_path / "crop.tif"
    create_raster(crop_path, crop, "EPSG:4326", from_origin(0, 10, 1, 1))

    feature = {
        "type": "Feature",
        "properties": {"depth": 1.0},
        "geometry": box(2, 2, 5, 5).__geo_interface__,
    }

    arr = drawn_features_to_depth_array([feature], str(crop_path))

    assert arr.shape == crop.shape
    assert arr.max() == 1.0
    assert arr.sum() > 0


def test_process_flood_damage_uses_default_season(tmp_path):
    crop = np.array([[1]], dtype=np.uint16)
    crop_path = tmp_path / "crop.tif"
    create_raster(crop_path, crop, "EPSG:4326", from_origin(0, 1, 1, 1))

    depth_arr = np.full((1, 1), 6.0, dtype=float)
    flood_metadata = {"floodA": {"return_period": 10, "flood_month": 1}}

    out_dir = tmp_path / "out"
    _, summaries, diagnostics, _ = process_flood_damage(
        str(crop_path), [("floodA", depth_arr)], str(out_dir), 100, None, flood_metadata
    )

    df = summaries["floodA"]
    assert df.iloc[0]["EAD"] == 0
    assert any(d["Issue"] == "Out of season" for d in diagnostics)


def test_run_monte_carlo_month_uncertainty(tmp_path):
    crop = np.array([[1]], dtype=np.uint16)
    crop_path = tmp_path / "crop.tif"
    create_raster(crop_path, crop, "EPSG:4326", from_origin(0, 1, 1, 1))

    depth_arr = np.full((1, 1), 6.0, dtype=float)
    crop_inputs = {1: {"Value": 10, "GrowingSeason": [6]}}
    flood_metadata = {"floodA": {"return_period": 10, "flood_month": 6}}

    out_dir = tmp_path / "out"
    _, summaries, _, _ = process_flood_damage(
        str(crop_path), [("floodA", depth_arr)], str(out_dir), 100, crop_inputs, flood_metadata
    )

    original_ead = summaries["floodA"].iloc[0]["EAD"]
    mc = run_monte_carlo(summaries, flood_metadata, 1000, 0, 0, month_uncertainty=True)
    mc_mean = mc["floodA"].iloc[0]["EAD_MC_Mean"]
    expected = round(original_ead / 12, 2)
    assert mc_mean == pytest.approx(expected, rel=0.2)
