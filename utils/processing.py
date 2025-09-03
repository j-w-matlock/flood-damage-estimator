import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio import features
import geopandas as gpd
from shapely.geometry import shape
from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference
import re

from .crop_definitions import (
    CROP_DEFINITIONS,
    CROP_GROWING_SEASONS,
    DEFAULT_GROWING_SEASON,
)

# Depth in feet assumed to result in 100% crop damage
FULL_DAMAGE_DEPTH_FT = 6.0

# Conversion factor from square meters to acres
SQ_METERS_TO_ACRES = 0.000247105


INVALID_SHEET_CHARS = r"[\[\]:\*\?/\\]"


def sanitize_label(label, max_length=31):
    r"""Return a filesystem and Excel-safe version of *label*.

    Excel sheet names cannot contain ``[]:*?/\`` and must be <=31 characters.
    This helper replaces forbidden characters with underscores and truncates
    the result so that downstream file writes do not crash.
    """

    cleaned = re.sub(INVALID_SHEET_CHARS, "_", str(label))
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length]
    return cleaned or "sheet"


def align_crop_to_depth(crop_path, depth_path):
    with rasterio.open(depth_path) as depth_src, rasterio.open(crop_path) as crop_src:
        dst_crs = depth_src.crs
        dst_transform = depth_src.transform
        width = depth_src.width
        height = depth_src.height

        profile = crop_src.profile.copy()
        profile.update(
            {
                "crs": dst_crs,
                "transform": dst_transform,
                "width": width,
                "height": height,
            }
        )

        reprojected = np.zeros((height, width), dtype=np.uint16)
        reproject(
            source=crop_src.read(1),
            destination=reprojected,
            src_transform=crop_src.transform,
            src_crs=crop_src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
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


def constant_depth_array(crop_path, depth_value=0.5):
    """Create a depth raster matching the crop raster filled with a constant value."""

    with rasterio.open(crop_path) as src:
        arr_shape = src.read(1).shape

    return np.full(arr_shape, depth_value, dtype="float32")


def drawn_features_to_depth_array(features_list, crop_path, default_value=0.0):
    """Rasterize drawn polygon features with per-feature depth values."""

    if not features_list:
        with rasterio.open(crop_path) as src:
            arr_shape = src.read(1).shape
        return np.full(arr_shape, default_value, dtype="float32")

    with rasterio.open(crop_path) as src:
        height, width = src.height, src.width
        transform = src.transform
        crs = src.crs

    geoms = []
    for feat in features_list:
        geom = shape(feat["geometry"])
        depth = float(feat.get("properties", {}).get("depth", default_value))
        geoms.append((geom, depth))

    burned = features.rasterize(
        geoms,
        out_shape=(height, width),
        transform=transform,
        fill=default_value,
        dtype="float32",
    )
    return burned


def process_flood_damage(
    crop_path,
    depth_inputs,
    output_dir,
    period_years,
    crop_inputs=None,
    flood_metadata=None,
):
    """Compute deterministic flood damages for each event.

    Damage ratios scale linearly up to :data:`FULL_DAMAGE_DEPTH_FT` feet of
    inundation, which represents total crop loss (6 ft by default).
    """

    flood_metadata = flood_metadata or {}

    os.makedirs(output_dir, exist_ok=True)
    summaries, diagnostics, damage_raster_paths = {}, [], {}

    with rasterio.open(crop_path) as base_crop_src:
        base_crop_arr = base_crop_src.read(1)
        base_crop_profile = base_crop_src.profile.copy()
        crop_codes_present = [c for c in np.unique(base_crop_arr) if c != 0]

    if crop_inputs is None:
        crop_inputs = {
            code: {
                "Name": CROP_DEFINITIONS.get(code, (str(code), 0))[0],
                "Value": CROP_DEFINITIONS.get(code, (str(code), 0))[1],
                "GrowingSeason": CROP_GROWING_SEASONS.get(
                    code, DEFAULT_GROWING_SEASON
                ),
            }
            for code in crop_codes_present
        }
    else:
        crop_inputs = {k: v for k, v in crop_inputs.items() if k != 0}
        for code, props in crop_inputs.items():
            default_name, default_value = CROP_DEFINITIONS.get(code, (str(code), 0))
            props.setdefault("Name", default_name)
            props.setdefault("Value", default_value)
            props.setdefault(
                "GrowingSeason",
                CROP_GROWING_SEASONS.get(code, DEFAULT_GROWING_SEASON),
            )
        for code in crop_codes_present:
            if code not in crop_inputs:
                name, value = CROP_DEFINITIONS.get(code, (str(code), 0))
                crop_inputs[code] = {
                    "Name": name,
                    "Value": value,
                    "GrowingSeason": CROP_GROWING_SEASONS.get(
                        code, DEFAULT_GROWING_SEASON
                    ),
                }

    for item in depth_inputs:
        if isinstance(item, tuple):
            label, data = item
            label = sanitize_label(label)
            meta = flood_metadata.get(label, {})
            return_period = meta.get("return_period", 100)
            flood_month = meta.get("flood_month", 6)

            if isinstance(data, np.ndarray):
                depth_arr = data.astype("float32")
                aligned_crop = base_crop_arr
                crop_profile = base_crop_profile
            else:
                depth_path = data
                aligned_crop, crop_profile = align_crop_to_depth(crop_path, depth_path)
                with rasterio.open(depth_path) as depth_src:
                    depth_arr = depth_src.read(
                        1,
                        out_shape=(aligned_crop.shape),
                        resampling=Resampling.nearest,
                    ).astype("float32")
        else:
            depth_path = item
            label = sanitize_label(
                os.path.splitext(os.path.basename(depth_path))[0]
            )
            meta = flood_metadata.get(label, {})
            return_period = meta.get("return_period", 100)
            flood_month = meta.get("flood_month", 6)

            aligned_crop, crop_profile = align_crop_to_depth(crop_path, depth_path)

            with rasterio.open(depth_path) as depth_src:
                depth_arr = depth_src.read(
                    1,
                    out_shape=(aligned_crop.shape),
                    resampling=Resampling.nearest,
                ).astype("float32")

        damage_arr = np.zeros_like(aligned_crop, dtype=float)
        rows = []

        damage_ratio = np.clip(depth_arr / FULL_DAMAGE_DEPTH_FT, 0, 1)

        crs = crop_profile["crs"]
        unit_factor = 1.0
        if crs and crs.is_projected:
            try:
                lf = crs.linear_units_factor
                unit_factor = lf[1] if isinstance(lf, tuple) else float(lf)
            except Exception:
                pass
        pixel_area_acres = (
            abs(crop_profile["transform"][0] * crop_profile["transform"][4])
            * (unit_factor ** 2)
            * SQ_METERS_TO_ACRES
        )

        for code, props in crop_inputs.items():
            value = props["Value"]
            name = props.get("Name", CROP_DEFINITIONS.get(code, (str(code), 0))[0])

            mask = aligned_crop == code
            out_of_season = flood_month not in props["GrowingSeason"]
            not_present = not np.any(mask)

            if out_of_season or not_present:
                issue = "Out of season" if out_of_season else "Not present"
                diagnostics.append(
                    {"Flood": label, "CropCode": code, "Issue": issue}
                )
                rows.append(
                    {
                        "CropCode": code,
                        "CropName": name,
                        "FloodedPixels": 0,
                        "FloodedAcres": 0.0,
                        "ValuePerAcre": value,
                        "DollarsLost": 0.0,
                        "EAD": 0.0,
                        "ReturnPeriod": return_period,
                        "FloodMonth": flood_month,
                        "GrowingSeason": props["GrowingSeason"],
                    }
                )
                continue

            flooded_pixels = int(mask.sum())
            flooded_acres = flooded_pixels * pixel_area_acres
            crop_damage = value * damage_ratio * mask * pixel_area_acres
            avg_damage = crop_damage.sum()
            ead = avg_damage * (1 / return_period)
            damage_arr = np.where(mask, damage_ratio, damage_arr)

            rows.append(
                {
                    "CropCode": code,
                    "CropName": name,
                    "FloodedPixels": flooded_pixels,
                    "FloodedAcres": flooded_acres,
                    "ValuePerAcre": value,
                    "DollarsLost": round(avg_damage, 2),
                    "EAD": round(ead, 2),
                    "ReturnPeriod": return_period,
                    "FloodMonth": flood_month,
                    "GrowingSeason": props["GrowingSeason"],
                }
            )

        df = pd.DataFrame(rows)
        summaries[label] = df

        damage_arr = damage_arr.astype(np.float32)
        damage_crop_arr = np.where(damage_arr > 0, aligned_crop, 0).astype(
            aligned_crop.dtype
        )

        ratio_profile = crop_profile.copy()
        ratio_profile["dtype"] = "float32"
        ratio_path = os.path.join(output_dir, f"damage_{label}.tif")
        with rasterio.open(ratio_path, "w", **ratio_profile) as dst:
            dst.write(damage_arr, 1)

        crop_profile_out = crop_profile.copy()
        crop_profile_out["dtype"] = aligned_crop.dtype
        crop_path = os.path.join(output_dir, f"damage_crops_{label}.tif")
        with rasterio.open(crop_path, "w", **crop_profile_out) as dst:
            dst.write(damage_crop_arr, 1)

        damage_raster_paths[label] = {"ratio": ratio_path, "crop": crop_path}

    # Optional trapezoidal integration
    trapezoid_rows = []
    if len(summaries) > 1:
        combined = pd.concat(
            [df.assign(Flood=label) for label, df in summaries.items()]
        )
        grouped = combined.groupby("CropCode")

        for code, group in grouped:
            sorted_group = group.sort_values("ReturnPeriod", ascending=False)
            x = 1 / sorted_group["ReturnPeriod"].values
            y = sorted_group["EAD"].values
            trapezoidal_ead = np.trapz(y, x)
            trapezoid_rows.append(
                {
                    "CropCode": code,
                    "TrapezoidalEAD": round(trapezoidal_ead, 2),
                    "FloodsUsed": len(group),
                }
            )

    excel_path = os.path.join(output_dir, "ag_damage_summary.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for label, df in summaries.items():
            df.to_excel(writer, sheet_name=label, index=False)
        if trapezoid_rows:
            pd.DataFrame(trapezoid_rows).to_excel(
                writer, sheet_name="Integrated_EAD", index=False
            )
        pd.DataFrame(diagnostics).to_excel(
            writer, sheet_name="Diagnostics", index=False
        )

    return excel_path, summaries, diagnostics, damage_raster_paths


def run_monte_carlo(
    summaries,
    flood_metadata,
    samples,
    value_uncertainty_pct,
    depth_uncertainty_ft,
    month_uncertainty=False,
):
    """Perform Monte Carlo EAD calculations.

    The standard deviation of depth error is expressed relative to
    :data:`FULL_DAMAGE_DEPTH_FT` feet (6 ft by default).
    When ``month_uncertainty`` is ``True``, flood months are sampled uniformly
    from all 12 months and compared against each crop's growing season.
    """

    results = {}
    for flood, df in summaries.items():
        meta = flood_metadata.get(flood, {})
        return_period = meta.get("return_period", 100)
        rows = []
        for _, row in df.iterrows():
            value_sd = row["ValuePerAcre"] * value_uncertainty_pct / 100
            value_samples = np.random.normal(row["ValuePerAcre"], value_sd, samples)
            depth_samples = np.random.normal(
                1.0, depth_uncertainty_ft / FULL_DAMAGE_DEPTH_FT, samples
            )

            if month_uncertainty:
                month_samples = np.random.randint(1, 13, samples)
            else:
                month_samples = np.full(samples, row["FloodMonth"])
            in_season = np.isin(month_samples, row.get("GrowingSeason", []))

            losses = (
                value_samples
                * depth_samples
                * row["FloodedAcres"]
                * (1 / return_period)
                * in_season
            )

            rows.append(
                {
                    "CropCode": row["CropCode"],
                    "EAD_MC_Mean": round(float(np.mean(losses)), 2),
                    "EAD_MC_5th": round(float(np.percentile(losses, 5)), 2),
                    "EAD_MC_95th": round(float(np.percentile(losses, 95)), 2),
                    "Original_EAD": row["EAD"],
                }
            )
        results[flood] = pd.DataFrame(rows)
    return results
