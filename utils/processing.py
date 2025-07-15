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

        print(f"\n🌊 Processing {label} (RP {return_period}, Month {flood_month})")

        try:
            with rasterio.open(crop_raster_path) as crop_src:
                crop_shape = (crop_src.height, crop_src.width)
                crop_scale = 0.1 if max(crop_src.width, crop_src.height) > 3000 else 0.25 if max(crop_src.width, crop_src.height) > 2000 else 1.0
                crop_out_shape = (int(crop_src.height * crop_scale), int(crop_src.width * crop_scale))
                crop_data = crop_src.read(1, out_shape=crop_out_shape, resampling=Resampling.nearest)
                crop_meta = crop_src.meta.copy()
                crop_meta.update({"height": crop_out_shape[0], "width": crop_out_shape[1]})
                crop_meta = crop_src.meta.copy()

            with rasterio.open(depth_path) as depth_src:
                depth_shape = (depth_src.height, depth_src.width)
                depth_scale = 0.1 if max(depth_src.width, depth_src.height) > 3000 else 0.25 if max(depth_src.width, depth_src.height) > 2000 else 1.0
                depth_out_shape = (int(depth_src.height * depth_scale), int(depth_src.width * depth_scale))
                depth_data = depth_src.read(1, out_shape=depth_out_shape, resampling=Resampling.average)

            if crop_data.shape != depth_data.shape:
                raise ValueError(f"Mismatched raster dimensions: crop={crop_data.shape}, depth={depth_data.shape}")

            unique_crops = np.unique(crop_data)
            summary = []

            for crop_code in unique_crops:
                if crop_code == 0 or crop_code not in crop_inputs:
                    diagnostics.append({"Flood": label, "CropCode": int(crop_code), "Note": "Skipped - not in crop inputs or code is 0"})
                    continue

                mask = crop_data == crop_code
                depths = depth_data[mask]
                if depths.size == 0 or np.all(depths <= 0):
                    diagnostics.append({"Flood": label, "CropCode": int(crop_code), "Note": "Skipped - no flooding detected"})
                    continue

                # Simulate uncertainty
                all_estimates = []
                for _ in range(samples):
                    perturb_depth = depths + np.random.normal(0, 0.1, size=depths.shape)
                    perturb_depth = np.clip(perturb_depth, 0, None)
                    damage_pct = np.clip(perturb_depth / 6.0, 0, 1)  # Linear placeholder
                    avg_pct = damage_pct.mean()
                    all_estimates.append(avg_pct)

                mean_pct = np.mean(all_estimates)
                p05 = np.percentile(all_estimates, 5)
                p95 = np.percentile(all_estimates, 95)

                acres = mask.sum() * 0.222394  # 30m pixels → acres
                value = crop_inputs[crop_code]["Value"]
                season = crop_inputs[crop_code]["GrowingSeason"]
                season_factor = 1.0 if flood_month in season else 0.0

                loss = acres * value * mean_pct * season_factor

                summary.append({
                    "CropCode": crop_code,
                    "Acres": acres,
                    "MeanPctDamage": mean_pct,
                    "LossPerAcre": value * mean_pct,
                    "DollarsLost": loss,
                    "SeasonalFactor": season_factor,
                    "P05": p05,
                    "P95": p95
                })

            df = pd.DataFrame(summary)
            all_summaries[label] = df

            damage_pct = np.clip(depth_data / 6.0, 0, 1)
            damage_raster = damage_pct * np.isin(crop_data, list(crop_inputs.keys()))

            out_meta = crop_meta.copy()
            out_meta.update({"dtype": "float32", "count": 1})
            damage_path = os.path.join(output_dir, f"damage_{label}.tif")

            with rasterio.open(damage_path, "w", **out_meta) as dst:
                dst.write(damage_raster.astype(np.float32), 1)

        except Exception as e:
            diagnostics.append({"Flood": label, "Error": str(e), "CropSize": crop_shape, "DepthSize": depth_shape})
            continue

    excel_path = os.path.join(output_dir, "ag_damage_summary.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for name, df in all_summaries.items():
            df.to_excel(writer, sheet_name=name, index=False)
        pd.DataFrame(diagnostics).to_excel(writer, sheet_name="Diagnostics", index=False)

    return excel_path, all_summaries, diagnostics
