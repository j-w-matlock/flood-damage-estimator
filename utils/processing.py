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

            # Simulate damage using Monte Carlo
            for s in range(samples):
                perturbed_depth = depth_arr + np.random.normal(0, 0.1, size=depth_arr.shape)
                crop_depth = np.where(mask, perturbed_depth, 0)
                damage_ratio = np.clip(crop_depth / 6.0, 0, 1)
                loss = damage_ratio * value_per_acre
                total_loss = np.sum(loss)
                mc_rows.append({
                    "Flood": label,
                    "Crop": crop_code,
                    "Sim": s + 1,
                    "Loss": total_loss
                })

            mean_loss = np.mean([r["Loss"] for r in mc_rows if r["Flood"] == label and r["Crop"] == crop_code])
            p5 = np.percentile([r["Loss"] for r in mc_rows if r["Flood"] == label and r["Crop"] == crop_code], 5)
            p95 = np.percentile([r["Loss"] for r in mc_rows if r["Flood"] == label and r["Crop"] == crop_code], 95)

            damage_arr = np.where(mask, damage_ratio, damage_arr)

            summary_rows.append({
                "CropCode": crop_code,
                "ValuePerAcre": value_per_acre,
                "MeanLoss": round(mean_loss, 2),
                "Loss_5th": round(p5, 2),
                "Loss_95th": round(p95, 2),
                "DollarsLost": round(mean_loss, 2),
                "EAD": round((1.0 / return_period) * mean_loss, 2)
            })

        summary_df = pd.DataFrame(summary_rows)
        all_summaries[label] = summary_df

        # Save damage raster
        damage_output_path = os.path.join(output_dir, f"damage_{label}.tif")
        with rasterio.open(damage_output_path, "w", **crop_profile) as dst:
            dst.write(damage_arr, 1)

    # Export Excel summary
    excel_path = os.path.join(output_dir, "ag_damage_summary.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for flood, df in all_summaries.items():
            df.to_excel(writer, sheet_name=flood, index=False)
        pd.DataFrame(diagnostics).to_excel(writer, sheet_name="Diagnostics", index=False)
        pd.DataFrame(mc_rows).to_excel(writer, sheet_name="MonteCarlo", index=False)

        # Add EAD summary chart
        workbook = writer.book
        for flood in all_summaries:
            sheet = workbook[flood]
            chart = BarChart()
            chart.title = "Expected Annual Damage by Crop"
            chart.x_axis.title = "CropCode"
            chart.y_axis.title = "EAD ($/yr)"
            data = Reference(sheet, min_col=8, min_row=1, max_row=sheet.max_row)
            cats = Reference(sheet, min_col=1, min_row=2, max_row=sheet.max_row)
            chart.add_data(data, titles_from_data=True)
            chart.set_categories(cats)
            sheet.add_chart(chart, "K2")

    return excel_path, all_summaries, diagnostics
