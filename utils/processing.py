import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference


def align_crop_to_depth(crop_path, depth_path):
    """
    Aligns cropland raster to match the flood depth raster's CRS and resolution.
    Returns the reprojected crop array and updated raster profile.
    """
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
            'height': height,
            'dtype': 'uint16',
            'count': 1
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


def compute_trapezoidal_ead(probabilities, damages):
    """
    Computes USACE-compliant trapezoidal EAD integration over a damage-probability curve.
    """
    sorted_pairs = sorted(zip(probabilities, damages))
    probs, dmg = zip(*sorted_pairs)
    ead = 0.0
    for i in range(len(probs) - 1):
        dp = probs[i + 1] - probs[i]
        avg_dmg = (dmg[i] + dmg[i + 1]) / 2
        ead += avg_dmg * dp
    return round(ead, 2)


def process_flood_damage(crop_path, depth_paths, output_dir, period_years, crop_inputs, flood_metadata):
    """
    Processes flood damage per raster, estimating crop-specific losses and event-based EAD.
    Optionally integrates multiple events to compute full trapezoidal EAD across return periods.
    Returns summary tables, diagnostics, raster outputs, and an Excel report path.
    """
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

        damage_arr = np.zeros_like(aligned_crop, dtype=np.float32)
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
            ead_event = avg_damage * (1 / return_period)

            damage_arr = np.where(mask, damage_ratio, damage_arr)

            rows.append({
                "CropCode": code,
                "FloodedAcres": int(mask.sum()),
                "ValuePerAcre": value,
                "DollarsLost": round(avg_damage, 2),
                "EAD_Event": round(ead_event, 2)
            })

        df = pd.DataFrame(rows)
        summaries[label] = df
        damage_rasters[label] = damage_arr

        out_profile = crop_profile.copy()
        out_profile.update({
            'dtype': 'float32',
            'compress': 'lzw'
        })

        with rasterio.open(os.path.join(output_dir, f"damage_{label}.tif"), "w", **out_profile) as dst:
            dst.write(damage_arr, 1)

    # Optional USACE-compliant EAD Integration if multiple floods exist
    if len(depth_paths) > 1:
        crop_ead_table = {}

        for label, df in summaries.items():
            prob = 1 / flood_metadata[label + ".tif"]["return_period"]
            for _, row in df.iterrows():
                code = row["CropCode"]
                if code not in crop_ead_table:
                    crop_ead_table[code] = []
                crop_ead_table[code].append((prob, row["DollarsLost"]))

        full_ead_rows = []
        for code, pairs in crop_ead_table.items():
            probs, damages = zip(*pairs)
            integrated_ead = compute_trapezoidal_ead(probs, damages)
            full_ead_rows.append({
                "CropCode": code,
                "Integrated_EAD": integrated_ead
            })

        summaries["Integrated_EAD"] = pd.DataFrame(full_ead_rows)

    # Export Excel report including MC placeholder (if later added)
    excel_path = os.path.join(output_dir, "ag_damage_summary.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for label, df in summaries.items():
            df.to_excel(writer, sheet_name=label, index=False)
        if diagnostics:
            pd.DataFrame(diagnostics).to_excel(writer, sheet_name="Diagnostics", index=False)

    return excel_path, summaries, diagnostics, damage_rasters


def run_monte_carlo(summaries, flood_metadata, samples, value_uncertainty_pct, depth_uncertainty_ft):
    """
    Runs Monte Carlo simulation to estimate uncertainty around EAD values.
    Exports MC results to a separate summary that can be saved to Excel.
    """
    results = {}

    for flood, df in summaries.items():
        if flood == "Integrated_EAD":
            continue  # skip full integration sheet

        # Try to resolve flood metadata key with or without .tif extension
        possible_keys = [flood, flood + ".tif"]
        key = next((k for k in possible_keys if k in flood_metadata), None)
        if not key:
            raise ValueError(f"Flood metadata missing for: {flood}")

        return_period = flood_metadata[key]["return_period"]
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
                "Original_EAD": row["EAD_Event"]
            })

        results[flood] = pd.DataFrame(rows)

    return results
