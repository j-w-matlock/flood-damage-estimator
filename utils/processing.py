import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import reproject
from rasterio import features
from contextlib import ExitStack
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

# Characters disallowed in Excel sheet names: []:*?/\
INVALID_SHEET_CHARS = r"[\[\]:\*\?/\\]"


def sanitize_label(label, max_length=31):
    r"""Return a filesystem and Excel-safe version of *label*.

    Excel sheet names cannot contain the characters []:*?/\ and must be
    <=31 characters. This helper replaces forbidden characters with
    underscores and truncates the result so that downstream file writes do
    not crash.
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
        base_crop_profile = base_crop_src.profile.copy()
        base_crop_shape = base_crop_src.read(1).shape
        unique_codes = set()
        for _, window in base_crop_src.block_windows(1):
            block = base_crop_src.read(1, window=window)
            unique_codes.update(np.unique(block))
        crop_codes_present = [c for c in unique_codes if c != 0]

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
            with ExitStack() as stack:
                if isinstance(item, tuple):
                    label, data = item
                    label = sanitize_label(label)
                    meta = flood_metadata.get(label, {})
                    return_period = meta.get("return_period", 100)
                    flood_month = meta.get("flood_month", 6)

                    if isinstance(data, np.ndarray):
                        depth_arr = data.astype("float32")
                        if depth_arr.shape != base_crop_shape:
                            raise ValueError(
                                f"Depth array shape {depth_arr.shape} does not match crop raster shape {base_crop_shape}. "
                                "Please align inputs before processing."
                            )
                        crop_profile = base_crop_profile

                        def read_depth(window, arr=depth_arr):
                            r0, c0 = window.row_off, window.col_off
                            return arr[r0 : r0 + window.height, c0 : c0 + window.width]

                        crop_reader = base_crop_src
                        window_src = base_crop_src
                    else:
                        depth_path = data
                        depth_src = stack.enter_context(rasterio.open(depth_path))
                        crop_reader = stack.enter_context(
                            WarpedVRT(
                                base_crop_src,
                                crs=depth_src.crs,
                                transform=depth_src.transform,
                                width=depth_src.width,
                                height=depth_src.height,
                                resampling=Resampling.nearest,
                            )
                        )
                        crop_profile = crop_reader.profile.copy()

                        def read_depth(window, src=depth_src):
                            return src.read(1, window=window, out_dtype="float32")

                        window_src = depth_src
                else:
                    depth_path = item
                    label = sanitize_label(
                        os.path.splitext(os.path.basename(depth_path))[0]
                    )
                    meta = flood_metadata.get(label, {})
                    return_period = meta.get("return_period", 100)
                    flood_month = meta.get("flood_month", 6)

                    depth_src = stack.enter_context(rasterio.open(depth_path))
                    crop_reader = stack.enter_context(
                        WarpedVRT(
                            base_crop_src,
                            crs=depth_src.crs,
                            transform=depth_src.transform,
                            width=depth_src.width,
                            height=depth_src.height,
                            resampling=Resampling.nearest,
                        )
                    )
                    crop_profile = crop_reader.profile.copy()

                    def read_depth(window, src=depth_src):
                        return src.read(1, window=window, out_dtype="float32")

                    window_src = depth_src

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

                ratio_profile = crop_profile.copy()
                ratio_profile.update({"driver": "GTiff", "dtype": "float32"})
                ratio_path = os.path.join(output_dir, f"damage_{label}.tif")
                crop_profile_out = crop_profile.copy()
                crop_profile_out.update({"driver": "GTiff", "dtype": crop_reader.dtypes[0]})
                crop_path_out = os.path.join(output_dir, f"damage_crops_{label}.tif")

                in_season_codes = {
                    code: props
                    for code, props in crop_inputs.items()
                    if flood_month in props["GrowingSeason"]
                }
                stats = {code: {"pixels": 0, "sum_ratio": 0.0} for code in in_season_codes}
                out_of_season_codes = [
                    code for code in crop_inputs if code not in in_season_codes
                ]

                with rasterio.open(ratio_path, "w", **ratio_profile) as ratio_dst, rasterio.open(
                    crop_path_out, "w", **crop_profile_out
                ) as crop_dst:
                    for _, window in window_src.block_windows(1):
                        crop_block = crop_reader.read(1, window=window)
                        depth_block = read_depth(window).astype("float32")
                        damage_ratio = depth_block / FULL_DAMAGE_DEPTH_FT
                        np.clip(damage_ratio, 0, 1, out=damage_ratio)
                        if out_of_season_codes:
                            oos_mask = np.isin(crop_block, out_of_season_codes)
                            damage_ratio[oos_mask] = 0
                        damage_ratio[crop_block == 0] = 0

                        for code in in_season_codes:
                            mask = crop_block == code
                            if mask.any():
                                flood_mask = mask & (damage_ratio > 0)
                                stats[code]["pixels"] += int(flood_mask.sum())
                                stats[code]["sum_ratio"] += damage_ratio[flood_mask].sum()

                        damage_crop_block = crop_block.copy()
                        damage_crop_block[damage_ratio <= 0] = 0

                        ratio_dst.write(damage_ratio.astype("float32"), 1, window=window)
                        crop_dst.write(damage_crop_block, 1, window=window)

                rows = []
                for code, props in crop_inputs.items():
                    value = props["Value"]
                    name = props.get("Name", CROP_DEFINITIONS.get(code, (str(code), 0))[0])
                    if code in in_season_codes:
                        pixels = stats[code]["pixels"]
                        flooded_acres = pixels * pixel_area_acres
                        avg_damage = stats[code]["sum_ratio"] * value * pixel_area_acres
                        ead = avg_damage * (1 / return_period)
                        if pixels == 0:
                            diagnostics.append({"Flood": label, "CropCode": code, "Issue": "Not present"})
                        rows.append(
                            {
                                "CropCode": code,
                                "CropName": name,
                                "FloodedPixels": pixels,
                                "FloodedAcres": flooded_acres,
                                "ValuePerAcre": value,
                                "DollarsLost": round(avg_damage, 2),
                                "EAD": round(ead, 2),
                                "ReturnPeriod": return_period,
                                "FloodMonth": flood_month,
                                "GrowingSeason": props["GrowingSeason"],
                            }
                        )
                    else:
                        diagnostics.append({"Flood": label, "CropCode": code, "Issue": "Out of season"})
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

                df = pd.DataFrame(rows)
                summaries[label] = df
                damage_raster_paths[label] = {"ratio": ratio_path, "crop": crop_path_out}
    
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
                    "CropName": row["CropName"],
                    "EAD_MC_Mean": round(float(np.mean(losses)), 2),
                    "EAD_MC_5th": round(float(np.percentile(losses, 5)), 2),
                    "EAD_MC_95th": round(float(np.percentile(losses, 95)), 2),
                    "Original_EAD": row["EAD"],
                }
            )
        results[flood] = pd.DataFrame(rows)
    return results
