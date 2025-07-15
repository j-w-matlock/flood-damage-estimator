import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, calculate_default_transform
from rasterio.io import MemoryFile
from scipy.interpolate import interp1d
from openpyxl import load_workbook
from openpyxl.chart import BarChart, Reference
import random
from collections import Counter


def process_flood_damage(crop_raster_path, depth_raster_paths, output_dir, period_years, samples, crop_inputs, flood_metadata):
    os.makedirs(output_dir, exist_ok=True)
    all_summaries = {}
    diagnostics = []

    for depth_path in depth_raster_paths:
        label = os.path.splitext(os.path.basename(depth_path))[0]
        filename = os.path.basename(depth_path)
        metadata = flood_metadata.get(filename, {"return_period": 100, "flood_month": 6})
        return_period = metadata["return_period"]
        flood_month = metadata["flood_month"]

        with rasterio.open(crop_raster_path) as src_crop:
            crop_data = src_crop.read(1)
            crop_transform = src_crop.transform
            crop_crs = src_crop.crs

        with rasterio.open(depth_path) as src_depth:
            depth_data = src_depth.read(1, resampling=Resampling.nearest)
            depth_meta = src_depth.meta.copy()
            depth_crs = src_depth.crs
            depth_transform = src_depth.transform

        if crop_crs != depth_crs or src_crop.res != src_depth.res:
            dst_transform, width, height = calculate_default_transform(
                crop_crs, depth_crs, src_crop.width, src_crop.height, *src_crop.bounds)
            reprojected = np.zeros((height, width), dtype=crop_data.dtype)
            reproject(
                source=crop_data,
                destination=reprojected,
                src_transform=crop_transform,
                src_crs=crop_crs,
                dst_transform=dst_transform,
                dst_crs=depth_crs,
                resampling=Resampling.nearest)
            crop_data = reprojected
            crop_transform = dst_transform

        min_rows = min(crop_data.shape[0], depth_data.shape[0])
        min_cols = min(crop_data.shape[1], depth_data.shape[1])
        crop_data = crop_data[:min_rows, :min_cols]
        depth_data = depth_data[:min_rows, :min_cols]

        damage = np.zeros_like(depth_data, dtype=np.float32)
        flooded_acres = {}
        pixel_area = abs(depth_transform[0] * depth_transform[4]) * 0.000247105

        for code, params in crop_inputs.items():
            mask = crop_data == code
            if not np.any(mask):
                diagnostics.append({"Flood": label, "CropCode": code, "Note": "No pixels of this crop"})
                continue
            if flood_month not in params["GrowingSeason"]:
                diagnostics.append({"Flood": label, "CropCode": code, "Note": "Out of growing season"})
                continue
            f = interp1d([0, 0.01, 6], [0, 0.9, 1.0], bounds_error=False, fill_value=(0, 1))
            damage[mask] = f(depth_data[mask])
            flooded_acres[code] = np.sum(mask) * pixel_area

        out_raster_path = os.path.join(output_dir, f"damage_{label}.tif")
        meta = depth_meta
        meta.update(dtype=rasterio.float32, count=1)
        with rasterio.open(out_raster_path, "w", **meta) as dst:
            dst.write(damage, 1)

        summary = []
        for code, acres in flooded_acres.items():
            mask = crop_data == code
            avg_damage = float(np.mean(damage[mask]))
            value = crop_inputs[code]["Value"]
            loss = avg_damage * acres * value
            summary.append({
                "CropCode": code,
                "FloodedAcres": round(acres, 2),
                "AvgDamage": round(avg_damage, 3),
                "DollarsLost": round(loss, 2),
                "EAD": round(loss / period_years, 2)
            })

        df = pd.DataFrame(summary)
        all_summaries[label] = df

    excel_path = os.path.join(output_dir, "ag_damage_summary.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for flood, df in all_summaries.items():
            df.to_excel(writer, sheet_name=f"Summary_{flood[:25]}", index=False)

        summary_rows = []
        for flood, df in all_summaries.items():
            for _, row in df.iterrows():
                summary_rows.append({
                    "Flood": flood,
                    "Crop": row["CropCode"],
                    "EAD": row["EAD"]
                })
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_excel(writer, sheet_name="EAD_Summary", index=False)

    wb = load_workbook(excel_path)
    ws = wb["EAD_Summary"]
    chart = BarChart()
    chart.title = "Expected Annual Damages by Crop"
    chart.y_axis.title = "EAD ($)"
    chart.x_axis.title = "Crop"
    data = Reference(ws, min_col=3, min_row=1, max_row=ws.max_row)
    cats = Reference(ws, min_col=2, min_row=2, max_row=ws.max_row)
    chart.add_data(data, titles_from_data=True)
    chart.set_categories(cats)
    ws.add_chart(chart, "E2")
    wb.save(excel_path)

    return excel_path, all_summaries, diagnostics
