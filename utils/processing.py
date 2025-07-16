import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, calculate_default_transform, aligned_target
from rasterio.io import MemoryFile
from rasterio.transform import Affine
from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference

def reproject_and_align_raster(src_raster_path, target_profile):
    with rasterio.open(src_raster_path) as src:
        dst_transform, width, height = calculate_default_transform(
            src.crs, target_profile["crs"], target_profile["width"], target_profile["height"], *src.bounds)
        dst_shape = (height, width)
        dst_array = np.zeros(dst_shape, dtype=src.meta['dtype'])

        reproject(
            source=rasterio.band(src, 1),
            destination=dst_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=target_profile["crs"],
            resampling=Resampling.nearest
        )
    return dst_array

def process_flood_damage(crop_raster_path, depth_raster_paths, output_dir, period_years, crop_inputs, flood_metadata):
    os.makedirs(output_dir, exist_ok=True)
    summaries = {}
    diagnostics = []
    damage_rasters = {}

    for depth_path in depth_raster_paths:
        label = os.path.splitext(os.path.basename(depth_path))[0]
        metadata = flood_metadata.get(os.path.basename(depth_path), {"return_period": 100, "flood_month": 6})
        return_period = metadata["return_period"]
        flood_month = metadata["flood_month"]

        # Open depth raster (target for alignment)
        with rasterio.open(depth_path) as depth_src:
            depth_profile = depth_src.profile.copy()
            depth_array = depth_src.read(1)
            depth_transform = depth_src.transform
            depth_crs = depth_src.crs

        # Reproject and align crop raster to depth raster
        with rasterio.open(crop_raster_path) as crop_src:
            crop_aligned = np.zeros(depth_array.shape, dtype=crop_src.meta["dtype"])
            reproject(
                source=rasterio.band(crop_src, 1),
                destination=crop_aligned,
                src_transform=crop_src.transform,
                src_crs=crop_src.crs,
                dst_transform=depth_transform,
                dst_crs=depth_crs,
                resampling=Resampling.nearest
            )
            crop_profile = crop_src.profile.copy()
            crop_profile.update({
                "height": depth_array.shape[0],
                "width": depth_array.shape[1],
                "transform": depth_transform,
                "crs": depth_crs
            })

        damage_arr = np.zeros_like(depth_array, dtype=float)
        summary_rows = []

        for code, props in crop_inputs.items():
            mask = (crop_aligned == code)
            if not np.any(mask):
                diagnostics.append({"Flood": label, "CropCode": code, "Issue": "Not present"})
                continue
            if flood_month not in props["GrowingSeason"]:
                diagnostics.append({"Flood": label, "CropCode": code, "Issue": "Out of season"})
                continue

            value = props["Value"]
            flooded_mask = (depth_array > 0) & mask
            damage_ratio = np.clip(depth_array / 6.0, 0, 1)
            crop_damage = value * damage_ratio * mask
            avg_damage = crop_damage[flooded_mask].sum()
            ead = avg_damage * (1 / return_period)

            damage_arr = np.where(mask, damage_ratio, damage_arr)

            summary_rows.append({
                "CropCode": code,
                "FloodedAcres": int(flooded_mask.sum()),
                "ValuePerAcre": value,
                "DollarsLost": round(avg_damage, 2),
                "EAD": round(ead, 2),
            })

        df = pd.DataFrame(summary_rows)
        summaries[label] = df
        damage_rasters[label] = damage_arr

        with rasterio.open(os.path.join(output_dir, f"damage_{label}.tif"), "w", **crop_profile) as dst:
            dst.write(damage_arr, 1)

    excel_path = os.path.join(output_dir, "ag_damage_summary.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for flood, df in summaries.items():
            df.to_excel(writer, sheet_name=flood, index=False)
        pd.DataFrame(diagnostics).to_excel(writer, sheet_name="Diagnostics", index=False)

    return excel_path, summaries, diagnostics, damage_rasters
