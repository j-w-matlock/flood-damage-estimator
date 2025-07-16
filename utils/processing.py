import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from openpyxl import load_workbook
from openpyxl.chart import BarChart, Reference
from collections import Counter
import random

def process_flood_damage(crop_raster_path, depth_raster_paths, output_dir, period_years, crop_inputs, flood_metadata):
    os.makedirs(output_dir, exist_ok=True)
    all_summaries = {}
    diagnostics = []

    for depth_path in depth_raster_paths:
        label = os.path.splitext(os.path.basename(depth_path))[0]
        filename = os.path.basename(depth_path)
        metadata = flood_metadata.get(filename, {"return_period": 100, "flood_month": 6})
        return_period = metadata["return_period"]
        flood_month = metadata["flood_month"]

        with rasterio.open(crop_raster_path) as crop_src:
            crop_arr = crop_src.read(1)
            crop_profile = crop_src.profile

        with rasterio.open(depth_path) as depth_src:
            depth_arr = depth_src.read(
                1,
                out_shape=(1, crop_arr.shape[0], crop_arr.shape[1]),
                resampling=Resampling.bilinear
            )

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
                diagnostics.append({"Flood": label, "CropCode": crop_code, "Issue": "Flood outside growing season"})
                continue

            # Direct damage (linear: 6 ft = 100%)
            crop_depth = np.where(mask, depth_arr, 0)
            damage_ratio = np.clip(crop_depth / 6.0, 0, 1)
            loss = damage_ratio * value_per_acre
            total_loss = np.sum(loss)
            flooded_acres = np.sum(mask)

            damage_arr = np.where(mask, damage_ratio, damage_arr)

            ead = total_loss * (1 / return_period)
            ead_annualized = ead * period_years

            summary_rows.append({
                "CropCode": crop_code,
                "FloodedAcres": int(flooded_acres),
                "ValuePerAcre": value_per_acre,
                "DirectDamage": round(total_loss, 2),
                "EAD": round(ead, 2),
                "EAD_Annualized": round(ead_annualized, 2)
            })

        summary_df = pd.DataFrame(summary_rows)
        all_summaries[label] = summary_df

        damage_output_path = os.path.join(output_dir, f"damage_{label}.tif")
        with rasterio.open(damage_output_path, "w", **crop_profile) as dst:
            dst.write(damage_arr, 1)

    # Excel export
    excel_path = os.path.join(output_dir, "ag_damage_summary.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for flood, df in all_summaries.items():
            df.to_excel(writer, sheet_name=flood, index=False)

        pd.DataFrame(diagnostics).to_excel(writer, sheet_name="Diagnostics", index=False)

        workbook = writer.book
        for flood in all_summaries:
            sheet = workbook[flood]

            # EAD Chart
            chart = BarChart()
            chart.title = "Expected Annual Damage by Crop"
            chart.x_axis.title = "Crop Code"
            chart.y_axis.title = "EAD ($/yr)"
            data = Reference(sheet, min_col=5, min_row=1, max_row=sheet.max_row)
            cats = Reference(sheet, min_col=1, min_row=2, max_row=sheet.max_row)
            chart.add_data(data, titles_from_data=True)
            chart.set_categories(cats)
            sheet.add_chart(chart, "K2")

    return excel_path, all_summaries, diagnostics

def run_monte_carlo(all_summaries, samples=100):
    mc_results = {}

    for flood, df in all_summaries.items():
        rows = []
        for _, row in df.iterrows():
            damage_mean = row["DirectDamage"]
            std_dev = 0.2 * damage_mean  # Placeholder 20% uncertainty
            draws = [random.gauss(damage_mean, std_dev) for _ in range(samples)]
            draws = [max(0, d) for d in draws]

            rows.append({
                "CropCode": row["CropCode"],
                "EAD_Mean": round(np.mean(draws), 2),
                "EAD_5th": round(np.percentile(draws, 5), 2),
                "EAD_95th": round(np.percentile(draws, 95), 2)
            })

        mc_results[flood] = pd.DataFrame(rows)

    return mc_results
