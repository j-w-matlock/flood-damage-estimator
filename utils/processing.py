import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
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

        with rasterio.open(depth_path) as depth_src:
            depth_arr = depth_src.read(1, resampling=Resampling.bilinear)
            depth_crs = depth_src.crs
            depth_transform = depth_src.transform
            depth_shape = depth_src.shape
            pixel_area = abs(depth_transform.a * depth_transform.e)

        with rasterio.open(crop_raster_path) as crop_src:
            crop_arr = crop_src.read(1)
            aligned_crop = np.zeros(depth_shape, dtype=np.uint16)
            reproject(
                source=crop_arr,
                destination=aligned_crop,
                src_transform=crop_src.transform,
                src_crs=crop_src.crs,
                dst_transform=depth_transform,
                dst_crs=depth_crs,
                resampling=Resampling.nearest
            )

        crop_arr = aligned_crop

        overlap_mask = (crop_arr > 0) & (depth_arr > 0)
        crop_pixels = np.sum(crop_arr > 0)
        depth_pixels = np.sum(depth_arr > 0)
        overlap_pixels = np.sum(overlap_mask)

        diagnostics.append({
            "Flood": label,
            "Crop": "All",
            "Reason": "Overlap check",
            "CropPixels": int(crop_pixels),
            "DepthPixels": int(depth_pixels),
            "OverlapPixels": int(overlap_pixels),
            "PctCropOverlap": round(100 * overlap_pixels / crop_pixels, 2) if crop_pixels else 0,
            "PctDepthOverlap": round(100 * overlap_pixels / depth_pixels, 2) if depth_pixels else 0
        })

        damage = np.zeros_like(depth_arr, dtype=np.float32)
        summary = []

        for code, data in crop_inputs.items():
            mask = crop_arr == int(code)
            if not np.any(mask):
                diagnostics.append({"Flood": label, "Crop": code, "Reason": "No crop pixels found"})
                continue
            f = interp1d([0, 0.01, 6], [0, 0.9, 1.0], bounds_error=False, fill_value=(0, 1))
            dvals = f(depth_arr[mask])
            damage[mask] = dvals

            acres = np.sum(mask) * pixel_area * 0.000247105
            avg = np.mean(dvals)
            loss = avg * acres * data["Value"]

            summary.append({
                "CropCode": code,
                "Pixels": int(np.sum(mask)),
                "Acres": acres,
                "AvgDamage": avg,
                "DollarsLost": loss
            })

        df = pd.DataFrame(summary)
        df = df[df["DollarsLost"] > 0]
        df.to_csv(os.path.join(output_dir, f"summary_{label}.csv"), index=False)
        all_summaries[label] = df

        profile = {
            "driver": "GTiff",
            "height": damage.shape[0],
            "width": damage.shape[1],
            "count": 1,
            "dtype": rasterio.float32,
            "crs": depth_crs,
            "transform": depth_transform
        }
        damage_path = os.path.join(output_dir, f"damage_{label}.tif")
        with rasterio.open(damage_path, 'w', **profile) as dst:
            dst.write(damage.astype(rasterio.float32), 1)

        if df.empty:
            diagnostics.append({"Flood": label, "Crop": "All", "Reason": "No damage detected"})

    mc_rows = []
    for depth_path in depth_raster_paths:
        label = os.path.splitext(os.path.basename(depth_path))[0]
        filename = os.path.basename(depth_path)
        rp = flood_metadata.get(filename, {}).get("return_period", 100)
        freq = 1.0 / rp
        df = all_summaries.get(label)
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            code = row["CropCode"]
            acres = row["Acres"]
            base = row["AvgDamage"]
            cv = crop_inputs[code]["Value"]
            for s in range(samples):
                for year in range(1, period_years + 1):
                    occurs = random.uniform(0, 1) < freq
                    perturbed = np.clip(random.gauss(base, 0.1 * base), 0, 1) if occurs else 0
                    loss = perturbed * acres * cv if occurs else 0
                    mc_rows.append({
                        "Flood": label,
                        "Crop": code,
                        "Sim": s + 1,
                        "Year": year,
                        "Damage": perturbed,
                        "Loss": loss
                    })

    summary_rows = []
    annual_rows = []
    if mc_rows:
        mc_df = pd.DataFrame(mc_rows)
        g = mc_df.groupby(["Flood", "Crop", "Sim"])
        for (flood, code, sim), grp in g:
            total_loss = grp["Loss"].sum()
            mean_annual = total_loss / period_years
            summary_rows.append({
                "Flood": flood,
                "Crop": code,
                "Sim": sim,
                "TotalLoss": total_loss,
                "MeanAnnualLoss": mean_annual
            })

        for (flood, code), grp in mc_df.groupby(["Flood", "Crop"]):
            rp = flood_metadata.get(f"{flood}.tif", {}).get("return_period", 100)
            freq = 1.0 / rp
            mean_loss = grp["Loss"].mean()
            annual_rows.append({
                "Flood": flood,
                "Crop": code,
                "RP": rp,
                "Mean Loss": mean_loss,
                "Annualized": freq * mean_loss
            })

    excel_path = os.path.join(output_dir, "ag_damage_summary.xlsx")
    with pd.ExcelWriter(excel_path) as w:
        for lbl, df in all_summaries.items():
            df.to_excel(w, sheet_name=f"Summary_{lbl[:25]}", index=False)
        if mc_rows:
            pd.DataFrame(mc_rows).to_excel(w, sheet_name="MonteCarlo", index=False)
            pd.DataFrame(summary_rows).to_excel(w, sheet_name="PeriodAnnualized", index=False)
            pd.DataFrame(annual_rows).to_excel(w, sheet_name="TraditionalAnnualized", index=False)
        pd.DataFrame(diagnostics).to_excel(w, sheet_name="Diagnostics", index=False)

    if mc_rows:
        wb = load_workbook(excel_path)
        ws = wb["TraditionalAnnualized"]
        chart = BarChart()
        chart.title = "Traditional Annualized Loss"
        chart.y_axis.title = "$"
        chart.x_axis.title = "Flood"
        data = Reference(ws, min_col=5, min_row=2, max_row=ws.max_row)
        cats = Reference(ws, min_col=1, min_row=2, max_row=ws.max_row)
        chart.add_data(data, titles_from_data=False)
        chart.set_categories(cats)
        ws.add_chart(chart, "H2")
        wb.save(excel_path)

    return excel_path, all_summaries, diagnostics
