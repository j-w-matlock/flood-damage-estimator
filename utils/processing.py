import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference

def align_crop_to_depth(crop_path, depth_path):
    with rasterio.open(depth_path) as depth_src:
        dst_crs = depth_src.crs
        dst_transform, width, height = calculate_default_transform(
            rasterio.open(crop_path).crs,
            dst_crs,
            rasterio.open(crop_path).width,
            rasterio.open(crop_path).height,
            *rasterio.open(crop_path).bounds
        )
        profile = rasterio.open(crop_path).profile.copy()
        profile.update({
            'crs': dst_crs,
            'transform': dst_transform,
            'width': width,
            'height': height
        })

        reprojected = np.zeros((height, width), dtype=np.uint16)
        reproject(
            source=rasterio.open(crop_path).read(1),
            destination=reprojected,
            src_transform=rasterio.open(crop_path).transform,
            src_crs=rasterio.open(crop_path).crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest
        )

        return reprojected, profile

def process_flood_damage(crop_path, depth_paths, output_dir, period_years, crop_inputs, flood_metadata):
    os.makedirs(output_dir, exist_ok=True)
    summaries, diagnostics, damage_rasters = {}, [], {}

    for depth_path in depth_paths:
        label = os.path.splitext(os.path.basename(depth_path))[0]
        meta = flood_metadata.get(os.path.basename(depth_path), {})
        return_period = meta.get("return_period", 100)
        flood_month = meta.get("flood_month", 6)

        aligned_crop, crop_profile = align_crop_to_depth(crop_path, depth_path)

        with rasterio.open(depth_path) as depth_src:
            depth_arr = depth_src.read(1, out_shape=(aligned_crop.shape), resampling=Resampling.bilinear)

        damage_arr = np.zeros_like(aligned_crop, dtype=float)
        rows = []

        for code, props in crop_inputs.items():
            if flood_month not in props["GrowingSeason"]:
                diagnostics.append({"Flood": label, "CropCode": code, "Issue": "Out of season"})
                continue

            mask = aligned_crop == code
            if not np.any(mask):
                diagnostics.append({"Flood": label, "CropCode": code, "Issue": "Not present"})
                continue

            value = props["Value"]
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

        with rasterio.open(os.path.join(output_dir, f"damage_{label}.tif"), "w", **crop_profile) as dst:
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
        rows = []
        for _, row in df.iterrows():
            sim = []
            for _ in range(samples):
                value = np.random.normal(row["ValuePerAcre"], row["ValuePerAcre"] * value_uncertainty_pct / 100)
                depth_ratio = np.random.normal(1.0, depth_uncertainty_ft / 6.0)
                sim_loss = value * depth_ratio * row["FloodedAcres"]
                sim.append(sim_loss * (1 / return_period))
            rows.append({
                "CropCode": row["CropCode"],
                "EAD_MC_Mean": round(np.mean(sim), 2),
                "EAD_MC_5th": round(np.percentile(sim, 5), 2),
                "EAD_MC_95th": round(np.percentile(sim, 95), 2),
                "Original_EAD": row["EAD"]
            })
        results[flood] = pd.DataFrame(rows)
    return results
