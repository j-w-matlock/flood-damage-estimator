import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from openpyxl import load_workbook
from openpyxl.chart import BarChart, Reference
from collections import Counter
import random

def process_flood_damage(crop_raster_path, depth_raster_paths, output_dir, period_years, samples, crop_inputs, flood_metadata):
    os.makedirs(output_dir, exist_ok=True)
    all_summaries = {}
    diagnostics = []
    per_pixel_damage_store = {}

    for depth_path in depth_raster_paths:
        label = os.path.splitext(os.path.basename(depth_path))[0]
        filename = os.path.basename(depth_path)
        metadata = flood_metadata.get(filename, {"return_period": 100, "flood_month": 6})
        return_period = metadata["return_period"]
        flood_month = metadata["flood_month"]

        print(f"\n🌊 Processing {label} (RP={return_period}, Month={flood_month})")

        with rasterio.open(crop_raster_path) as crop_src:
            crop_arr = crop_src.read(1)
            crop_profile = crop_src.profile
            transform = crop_src.transform

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

            crop_depth = np.where(mask, depth_arr, 0)
            damage_ratio = np.clip(crop_depth / 6.0, 0, 1)
            direct_loss = damage_ratio * value_per_acre
            total_direct_loss = np.sum(direct_loss)
            damage_arr = np.where(mask, damage_ratio, damage_arr)

            per_pixel_damage_store[(label, crop_code)] = direct_loss[mask]

            summary_rows.append({
                "CropCode": crop_code,
                "FloodedAcres": int(np.sum(mask)),
                "ValuePerAcre": value_per_acre,
                "MeanLoss": round(total_direct_loss, 2),
                "Loss_5th": 0,
                "Loss_95th": 0,
                "DirectDamage": round(total_direct_loss, 2),
                "DollarsLost": round(total_direct_loss, 2),
                "EAD": round(total_direct_loss * (1 / return_period), 2),
                "EAD_Annualized": round((total_direct_loss * (1 / return_period)) * period_years, 2)
            })

        summary_df = pd.DataFrame(summary_rows)
        all_summaries[label] = summary_df

        damage_output_path = os.path.join(output_dir, f"damage_{label}.tif")
        with rasterio.open(damage_output_path, "w", **crop_profile) as dst:
            dst.write(damage_arr, 1)

    excel_path = os.path.join(output_dir, "ag_damage_summary.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for flood, df in all_summaries.items():
            df.to_excel(writer, sheet_name=flood, index=False)
        pd.DataFrame(diagnostics).to_excel(writer, sheet_name="Diagnostics", index=False)

        workbook = writer.book
        for flood in all_summaries:
            sheet = workbook[flood]
            chart1 = BarChart()
            chart1.title = "Expected Annual Damage by Crop"
            chart1.x_axis.title = "Crop Code"
            chart1.y_axis.title = "EAD ($/yr)"
            data1 = Reference(sheet, min_col=10, min_row=1, max_row=sheet.max_row)
            cats1 = Reference(sheet, min_col=1, min_row=2, max_row=sheet.max_row)
            chart1.add_data(data1, titles_from_data=True)
            chart1.set_categories(cats1)
            sheet.add_chart(chart1, "K2")

            chart2 = BarChart()
            chart2.title = "Direct Flood Damage by Crop"
            chart2.y_axis.title = "Damage ($)"
            data2 = Reference(sheet, min_col=7, min_row=1, max_row=sheet.max_row)
            chart2.add_data(data2, titles_from_data=True)
            chart2.set_categories(cats1)
            sheet.add_chart(chart2, "K18")

    return excel_path, all_summaries, diagnostics, per_pixel_damage_store


def run_monte_carlo_analysis(original_summary, per_pixel_damage_store, flood_metadata, period_years, samples):
    monte_results = {}

    for (flood_label, crop_code), original_losses in per_pixel_damage_store.items():
        metadata = flood_metadata.get(flood_label + ".tif", {"return_period": 100})
        return_period = metadata["return_period"]
        std_dev = np.std(original_losses)
        mean = np.mean(original_losses)
        mc_samples = []

        for _ in range(samples):
            synthetic = np.random.normal(mean, std_dev, size=len(original_losses))
            mc_samples.append(np.sum(np.clip(synthetic, 0, None)))

        mean_loss = np.mean(mc_samples)
        p5 = np.percentile(mc_samples, 5)
        p95 = np.percentile(mc_samples, 95)

        if flood_label not in monte_results:
            monte_results[flood_label] = []

        monte_results[flood_label].append({
            "CropCode": crop_code,
            "MC_MeanLoss": round(mean_loss, 2),
            "MC_Loss_5th": round(p5, 2),
            "MC_Loss_95th": round(p95, 2),
            "MC_EAD": round(mean_loss * (1 / return_period), 2),
            "MC_EAD_Annualized": round((mean_loss * (1 / return_period)) * period_years, 2)
        })

    monte_summaries = {k: pd.DataFrame(v) for k, v in monte_results.items()}
    return monte_summaries
