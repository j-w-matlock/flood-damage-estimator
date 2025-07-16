import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from openpyxl import load_workbook
from openpyxl.chart import BarChart, Reference
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

        print(f"\n🌊 Processing {label} (RP={return_period}, Month={flood_month})")

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

            crop_depth = np.where(mask, depth_arr, 0)
            damage_ratio = np.clip(crop_depth / 6.0, 0, 1)
            direct_loss = damage_ratio * value_per_acre
            direct_damage = np.sum(direct_loss)

            damage_arr = np.where(mask, damage_ratio, damage_arr)

            summary_rows.append({
                "CropCode": crop_code,
                "FloodedAcres": int(np.sum(mask)),
                "ValuePerAcre": value_per_acre,
                "DirectDamage": round(direct_damage, 2),
                "DollarsLost": round(direct_damage, 2),
                "EAD": round(direct_damage * 1 / return_period, 2),
                "EAD_Annualized": round(direct_damage * 1 / return_period * period_years, 2),
                "StdDevEstimate": round(0.1 * direct_damage, 2)  # placeholder for Monte Carlo
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

            ead_chart = BarChart()
            ead_chart.title = "Expected Annual Damage by Crop"
            ead_chart.x_axis.title = "Crop Code"
            ead_chart.y_axis.title = "EAD ($/yr)"
            data = Reference(sheet, min_col=6, min_row=1, max_row=sheet.max_row)
            cats = Reference(sheet, min_col=1, min_row=2, max_row=sheet.max_row)
            ead_chart.add_data(data, titles_from_data=True)
            ead_chart.set_categories(cats)
            sheet.add_chart(ead_chart, "K2")

            direct_chart = BarChart()
            direct_chart.title = "Direct Flood Damage by Crop"
            direct_chart.y_axis.title = "Damage ($)"
            data = Reference(sheet, min_col=4, min_row=1, max_row=sheet.max_row)
            direct_chart.add_data(data, titles_from_data=True)
            direct_chart.set_categories(cats)
            sheet.add_chart(direct_chart, "K18")

    return excel_path, all_summaries, diagnostics


def run_monte_carlo(summary_df, period_years, samples=1000):
    results = []
    for _, row in summary_df.iterrows():
        mean = row["DirectDamage"]
        stddev = row.get("StdDevEstimate", 0.1 * mean)
        simulated = np.random.normal(loc=mean, scale=stddev, size=samples)
        simulated = np.clip(simulated, 0, None)

        mc_mean = simulated.mean()
        mc_5th = np.percentile(simulated, 5)
        mc_95th = np.percentile(simulated, 95)

        results.append({
            "CropCode": row["CropCode"],
            "MC_EAD_Mean": round(mc_mean / period_years, 2),
            "MC_EAD_5th": round(mc_5th / period_years, 2),
            "MC_EAD_95th": round(mc_95th / period_years, 2),
            "MC_EAD_Annualized_Mean": round(mc_mean, 2)
        })

    return pd.DataFrame(results)
