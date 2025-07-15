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
    mc_rows = []

    for depth_path in depth_raster_paths:
        label = os.path.splitext(os.path.basename(depth_path))[0]
        filename = os.path.basename(depth_path)
        metadata = flood_metadata.get(filename, {"return_period": 100, "flood_month": 6})
        return_period = metadata["return_period"]
        flood_month = metadata["flood_month"]
        freq = 1.0 / return_period

        print(f"\n🌊 Processing {label} (RP={return_period}, Month={flood_month})")

        # Align rasters
        with rasterio.open(crop_raster_path) as crop_src:
            crop_arr = crop_src.read(1)
            crop_profile = crop_src.profile

        with rasterio.open(depth_path) as depth_src:
            depth_arr = depth_src.read(1, out_shape=(1, crop_arr.shape[0], crop_arr.shape[1]), resampling=Resampling.bilinear)

        damage_arr = np.zeros_like(crop_arr, dtype=float)
        summary_rows = []

        for crop_code, props in crop_inputs.items():
            mask = crop_arr == crop_code
            if not np.any(mask):
                diagnostics.append({"Flood": label, "CropCode": crop_code, "Issue": "Crop not present in raster"})
                continue

            value_per_acre = props["Value"]
            growing_season = props["GrowingSeason"]
            if flood_month not in growing_season:
                diagnostics.append({"Flood": label, "CropCode": crop_code, "Issue": "Out of growing season"})
                continue

            pixel_area = crop_profile["transform"][0] * -crop_profile["transform"][4]  # assumes square pixels
            acres_per_pixel = pixel_area * 0.000247105
            total_pixels = np.sum(mask)
            total_acres = total_pixels * acres_per_pixel
            flooded_pixels = np.sum((depth_arr > 0.01) & mask)
            flooded_acres = flooded_pixels * acres_per_pixel

            for s in range(samples):
                total_loss = 0
                for year in range(1, period_years + 1):
                    occurs = random.uniform(0, 1) < freq
                    if occurs:
                        perturbed = depth_arr + np.random.normal(0, 0.1, size=depth_arr.shape)
                        crop_depth = np.where(mask, perturbed, 0)
                        damage_ratio = np.clip(crop_depth / 6.0, 0, 1)
                        avg_dam = np.sum(damage_ratio[mask]) / total_pixels
                        loss = avg_dam * total_acres * value_per_acre
                        total_loss += loss
                mc_rows.append({
                    "Flood": label,
                    "Crop": crop_code,
                    "Sim": s + 1,
                    "TotalLoss": total_loss,
                    "MeanAnnualLoss": total_loss / period_years
                })

            sim_df = pd.DataFrame([r for r in mc_rows if r["Flood"] == label and r["Crop"] == crop_code])
            mean_loss = sim_df["MeanAnnualLoss"].mean()
            p5 = sim_df["MeanAnnualLoss"].quantile(0.05)
            p95 = sim_df["MeanAnnualLoss"].quantile(0.95)

            summary_rows.append({
                "CropCode": crop_code,
                "ValuePerAcre": value_per_acre,
                "MeanAnnualLoss": round(mean_loss, 2),
                "Loss_5th": round(p5, 2),
                "Loss_95th": round(p95, 2),
                "Acres": round(total_acres, 2),
                "FloodedAcres": round(flooded_acres, 2),
                "Pixels": int(total_pixels)
            })

        summary_df = pd.DataFrame(summary_rows)
        all_summaries[label] = summary_df

        # Save damage raster (last simulated year’s pattern)
        damage_output_path = os.path.join(output_dir, f"damage_{label}.tif")
        with rasterio.open(damage_output_path, "w", **crop_profile) as dst:
            dst.write(damage_arr, 1)

    # Export Excel summary
    excel_path = os.path.join(output_dir, "ag_damage_summary.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for flood, df in all_summaries.items():
            df.to_excel(writer, sheet_name=flood[:31], index=False)
        pd.DataFrame(diagnostics).to_excel(writer, sheet_name="Diagnostics", index=False)
        pd.DataFrame(mc_rows).to_excel(writer, sheet_name="MonteCarlo", index=False)

        # Add summary chart
        workbook = writer.book
        for flood in all_summaries:
            sheet = workbook[flood[:31]]
            chart = BarChart()
            chart.title = "Mean Annual Crop Loss"
            chart.x_axis.title = "CropCode"
            chart.y_axis.title = "$ per year"
            data = Reference(sheet, min_col=3, min_row=1, max_row=sheet.max_row)
            cats = Reference(sheet, min_col=1, min_row=2, max_row=sheet.max_row)
            chart.add_data(data, titles_from_data=True)
            chart.set_categories(cats)
            sheet.add_chart(chart, "J2")

    return excel_path, all_summaries, diagnostics
