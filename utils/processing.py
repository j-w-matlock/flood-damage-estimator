import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from rasterio import features
import geopandas as gpd
from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference

# Depth in feet assumed to result in 100% crop damage
FULL_DAMAGE_DEPTH_FT = 6.0

def align_crop_to_depth(crop_path, depth_path):
    with rasterio.open(depth_path) as depth_src, rasterio.open(crop_path) as crop_src:
        dst_crs = depth_src.crs
        dst_transform, width, height = calculate_default_transform(
            crop_src.crs,
            dst_crs,
            crop_src.width,
            crop_src.height,
            *crop_src.bounds
        )
        profile = crop_src.profile.copy()
        profile.update({
            'crs': dst_crs,
            'transform': dst_transform,
            'width': width,
            'height': height
        })

        reprojected = np.zeros((height, width), dtype=np.uint16)
        reproject(
            source=crop_src.read(1),
            destination=reprojected,
            src_transform=crop_src.transform,
            src_crs=crop_src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest
        )

        return reprojected, profile

def polygon_mask_to_depth_array(polygon_path, crop_path, depth_value=0.5):
    """Rasterize polygons to the crop raster grid."""

    with rasterio.open(crop_path) as src:
        height, width = src.height, src.width
        transform = src.transform
        crs = src.crs

    if polygon_path.lower().endswith(".zip"):
        gdf = gpd.read_file(f"zip://{polygon_path}")
    else:
        gdf = gpd.read_file(polygon_path)

    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    shapes = [(geom, depth_value) for geom in gdf.geometry]
    burned = features.rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0.0,
        dtype="float32",
    )
    return burned

def process_flood_damage(crop_path, depth_inputs, output_dir, period_years, crop_inputs, flood_metadata):
    """Compute deterministic flood damages for each event.

    Damage ratios scale linearly up to :data:`FULL_DAMAGE_DEPTH_FT` feet of
    inundation, which represents total crop loss (6 ft by default).
    """

    os.makedirs(output_dir, exist_ok=True)
    summaries, diagnostics, damage_rasters = {}, [], {}

    with rasterio.open(crop_path) as base_crop_src:
        base_crop_arr = base_crop_src.read(1)
        base_crop_profile = base_crop_src.profile.copy()

    for item in depth_inputs:
        if isinstance(item, tuple):
            label, data = item
            meta = flood_metadata.get(label, {})
            return_period = meta.get("return_period", 100)
            flood_month = meta.get("flood_month", 6)

            if isinstance(data, np.ndarray):
                depth_arr = data
                aligned_crop = base_crop_arr
                crop_profile = base_crop_profile
            else:
                depth_path = data
                aligned_crop, crop_profile = align_crop_to_depth(crop_path, depth_path)
                with rasterio.open(depth_path) as depth_src:
                    depth_arr = depth_src.read(1, out_shape=(aligned_crop.shape), resampling=Resampling.bilinear)
        else:
            depth_path = item
            label = os.path.splitext(os.path.basename(depth_path))[0]
            meta = flood_metadata.get(label, {})
            return_period = meta.get("return_period", 100)
            flood_month = meta.get("flood_month", 6)

            aligned_crop, crop_profile = align_crop_to_depth(crop_path, depth_path)

            with rasterio.open(depth_path) as depth_src:
                depth_arr = depth_src.read(1, out_shape=(aligned_crop.shape), resampling=Resampling.bilinear)

        damage_arr = np.zeros_like(aligned_crop, dtype=float)
        rows = []

        damage_ratio = np.clip(depth_arr / FULL_DAMAGE_DEPTH_FT, 0, 1)

        for code, props in crop_inputs.items():
            if flood_month not in props["GrowingSeason"]:
                diagnostics.append({"Flood": label, "CropCode": code, "Issue": "Out of season"})
                continue

            mask = aligned_crop == code
            if not np.any(mask):
                diagnostics.append({"Flood": label, "CropCode": code, "Issue": "Not present"})
                continue

            value = props["Value"]
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
                "ReturnPeriod": return_period,
                "FloodMonth": flood_month
            })

        df = pd.DataFrame(rows)
        summaries[label] = df
        damage_rasters[label] = damage_arr

        with rasterio.open(os.path.join(output_dir, f"damage_{label}.tif"), "w", **crop_profile) as dst:
            dst.write(damage_arr, 1)

    # Optional trapezoidal integration
    trapezoid_rows = []
    if len(summaries) > 1:
        combined = pd.concat([
            df.assign(Flood=label) for label, df in summaries.items()
        ])
        grouped = combined.groupby("CropCode")

        for code, group in grouped:
            sorted_group = group.sort_values("ReturnPeriod", ascending=False)
            x = 1 / sorted_group["ReturnPeriod"].values
            y = sorted_group["EAD"].values
            trapezoidal_ead = np.trapz(y, x)
            trapezoid_rows.append({
                "CropCode": code,
                "TrapezoidalEAD": round(trapezoidal_ead, 2),
                "FloodsUsed": len(group)
            })

    excel_path = os.path.join(output_dir, "ag_damage_summary.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for label, df in summaries.items():
            df.to_excel(writer, sheet_name=label, index=False)
        if trapezoid_rows:
            pd.DataFrame(trapezoid_rows).to_excel(writer, sheet_name="Integrated_EAD", index=False)
        pd.DataFrame(diagnostics).to_excel(writer, sheet_name="Diagnostics", index=False)

    return excel_path, summaries, diagnostics, damage_rasters

def run_monte_carlo(summaries, flood_metadata, samples, value_uncertainty_pct, depth_uncertainty_ft):
    """Perform Monte Carlo EAD calculations.

    The standard deviation of depth error is expressed relative to
    :data:`FULL_DAMAGE_DEPTH_FT` feet (6 ft by default).
    """

    results = {}
    for flood, df in summaries.items():
        meta = flood_metadata.get(flood, {})
        return_period = meta.get("return_period", 100)
        rows = []
        for _, row in df.iterrows():
            value_sd = row["ValuePerAcre"] * value_uncertainty_pct / 100
            value_samples = np.random.normal(row["ValuePerAcre"], value_sd, samples)
            depth_samples = np.random.normal(1.0, depth_uncertainty_ft / FULL_DAMAGE_DEPTH_FT, samples)
            losses = value_samples * depth_samples * row["FloodedAcres"] * (1 / return_period)

            rows.append({
                "CropCode": row["CropCode"],
                "EAD_MC_Mean": round(float(np.mean(losses)), 2),
                "EAD_MC_5th": round(float(np.percentile(losses, 5)), 2),
                "EAD_MC_95th": round(float(np.percentile(losses, 95)), 2),
                "Original_EAD": row["EAD"]
            })
        results[flood] = pd.DataFrame(rows)
    return results
