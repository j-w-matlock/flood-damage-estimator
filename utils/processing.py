import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference
from collections import defaultdict

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

        with rasterio.open(crop_raster_path) as crop_src:
            crop_arr = crop_src.read(1)
            profile = crop_src.profile

        with rasterio.open(depth_path) as depth_src:
            depth_arr = depth_src.read(1, out_shape=(crop_arr.shape), resampling=Resampling.bilinear)

        damage_arr = np.zeros_like(crop_arr, dtype=float)
        rows = []

        for code, props in crop_inputs.items():
            mask = (crop_arr == code)
            if not np.any(mask):
                diagnostics.append({"Flood": label, "CropCode": code, "Issue": "Not present"})
                continue
            if flood_month not in props["GrowingSeason"]:
                diagnostics.append({"Flood": label, "CropCode": code, "Issue": "Out of season"})
                continue

            value = props["Value"]
            flooded = np.where(mask, depth_arr > 0, 0)
            damage_ratio = np.clip(depth_arr / 6.0, 0, 1)
            crop_damage = value * damage_ratio * mask
            avg_damage = crop_damage.sum()
            ead = avg_damage * (1 / return_period)

            damage_arr = np.where(mask, damage_ratio, damage_arr)

            rows.append({
                "CropCode": code,
                "FloodedAcres": int(mask.sum()),
                "ValuePerAcre": value,
                "DollarsLost": round(avg_damage, 2),
                "EAD": round(ead, 2),
            })

        df = pd.DataFrame(rows)
        summaries[label] = df
        damage_rasters[label] = damage_arr

        with rasterio.open(os.path.join(output_dir, f"damage_{label}.tif"), "w", **profile) as dst:
            dst.write(damage_arr, 1)

    excel_path = os.path.join(output_dir, "ag_damage_summary.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for flood, df in summaries.items():
            df.to_excel(writer, sheet_name=flood, index=False)
        pd.DataFrame(diagnostics).to_excel(writer, sheet_name="Diagnostics", index=False)

    return excel_path, summaries, diagnostics, damage_rasters


def run_monte_carlo(summaries, flood_metadata, samples, value_uncertainty_pct, depth_uncertainty_ft):
    results = {}
    for flood, df in summaries.items():
        return_period = flood_metadata[flood]["return_period"]
        mc_rows = []
        for _, row in df.iterrows():
            code = row["CropCode"]
            value = row["ValuePerAcre"]
            base_ead = row["EAD"]
            mean_damage = row["DollarsLost"]
            sim_losses = []
            for _ in range(samples):
                perturbed_val = np.random.normal(value, value * (value_uncertainty_pct / 100))
                perturbed_depth = np.random.normal(1.0, depth_uncertainty_ft / 6.0)  # scale damage ratio
                simulated_loss = perturbed_val * perturbed_depth * row["FloodedAcres"]
                sim_losses.append(simulated_loss * (1 / return_period))
            mc_rows.append({
                "CropCode": code,
                "EAD_MC_Mean": round(np.mean(sim_losses), 2),
                "EAD_MC_5th": round(np.percentile(sim_losses, 5), 2),
                "EAD_MC_95th": round(np.percentile(sim_losses, 95), 2),
                "Original_EAD": base_ead
            })
        results[flood] = pd.DataFrame(mc_rows)
    return results
