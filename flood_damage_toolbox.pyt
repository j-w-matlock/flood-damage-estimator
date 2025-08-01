"""ArcGIS Pro toolbox interface for flood damage processing."""

from pathlib import Path

import arcpy
import pandas as pd

try:
    import rasterio  # noqa: F401
    from utils.processing import process_flood_damage, run_monte_carlo
    RASTERIO_AVAILABLE = True
except Exception:  # pragma: no cover - ArcGIS Pro may lack rasterio
    RASTERIO_AVAILABLE = False
    from utils.processing import run_monte_carlo, FULL_DAMAGE_DEPTH_FT
    def process_flood_damage(crop_raster, depth_inputs, output_dir, period_years, crop_inputs, flood_metadata):
        """Fallback processing using arcpy when rasterio is unavailable."""
        import os
        import numpy as np

        os.makedirs(output_dir, exist_ok=True)

        crop_arr = arcpy.RasterToNumPyArray(crop_raster)
        desc = arcpy.Describe(crop_raster)
        cell_x = desc.meanCellWidth
        cell_y = desc.meanCellHeight
        lower_left = desc.extent.lowerLeft

        summaries, diagnostics = {}, []

        for label, depth_path in depth_inputs:
            meta = flood_metadata.get(label, {})
            return_period = meta.get("return_period", 100)
            flood_month = meta.get("flood_month", 6)

            depth_arr = arcpy.RasterToNumPyArray(depth_path)
            if depth_arr.shape != crop_arr.shape:
                tmp = arcpy.env.scratchGDB + f"/resamp_{label}"
                arcpy.management.ProjectRaster(depth_path, tmp, crop_raster, "BILINEAR", f"{cell_x} {cell_y}")
                depth_arr = arcpy.RasterToNumPyArray(tmp)
                arcpy.management.Delete(tmp)

            damage_arr = np.zeros_like(depth_arr, dtype=float)
            rows = []
            damage_ratio = np.clip(depth_arr / FULL_DAMAGE_DEPTH_FT, 0, 1)

            for code, props in crop_inputs.items():
                if flood_month not in props["GrowingSeason"]:
                    diagnostics.append({"Flood": label, "CropCode": code, "Issue": "Out of season"})
                    continue

                mask = crop_arr == code
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
                    "DollarsLost": round(float(avg_damage), 2),
                    "EAD": round(float(ead), 2),
                    "ReturnPeriod": return_period,
                    "FloodMonth": flood_month,
                })

            summaries[label] = pd.DataFrame(rows)
            out_ras = arcpy.NumPyArrayToRaster(damage_arr, lower_left, cell_x, cell_y, -9999)
            out_ras.save(os.path.join(output_dir, f"damage_{label}.tif"))

        trapezoid_rows = []
        if len(summaries) > 1:
            combined = pd.concat([df.assign(Flood=l) for l, df in summaries.items()])
            grouped = combined.groupby("CropCode")
            for code, group in grouped:
                sorted_group = group.sort_values("ReturnPeriod", ascending=False)
                x = 1 / sorted_group["ReturnPeriod"].values
                y = sorted_group["EAD"].values
                trapezoidal_ead = np.trapz(y, x)
                trapezoid_rows.append({
                    "CropCode": code,
                    "TrapezoidalEAD": round(float(trapezoidal_ead), 2),
                    "FloodsUsed": len(group),
                })

        excel_path = os.path.join(output_dir, "ag_damage_summary.xlsx")
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            for label, df in summaries.items():
                df.to_excel(writer, sheet_name=label, index=False)
            if trapezoid_rows:
                pd.DataFrame(trapezoid_rows).to_excel(writer, sheet_name="Integrated_EAD", index=False)
            pd.DataFrame(diagnostics).to_excel(writer, sheet_name="Diagnostics", index=False)

        return excel_path, summaries, diagnostics, {}
from utils.processing import process_flood_damage, run_monte_carlo


class Toolbox(object):
    def __init__(self):
        self.label = "Flood Damage Toolbox"
        self.alias = "flooddamage"
        self.tools = [FloodDamageTool]


class FloodDamageTool(object):
    def __init__(self):
        self.label = "Flood Damage Estimator"
        self.description = "Estimate agricultural flood damages from raster inputs."
        self.canRunInBackground = False

    def getParameterInfo(self):
        params = [
            arcpy.Parameter(
                displayName="Crop Raster",
                name="crop_raster",
                datatype="DERasterDataset",
                parameterType="Required",
                direction="Input",
            ),
            arcpy.Parameter(
                displayName="Flood Depth Rasters",
                name="flood_rasters",
                datatype="DERasterDataset",
                parameterType="Required",
                direction="Input",
                multiValue=True,
            ),
            arcpy.Parameter(
                displayName="Return Periods (years)",
                name="return_periods",
                datatype="GPLong",
                parameterType="Required",
                direction="Input",
                multiValue=True,
            ),
            arcpy.Parameter(
                displayName="Flood Months (1-12)",
                name="flood_months",
                datatype="GPLong",
                parameterType="Required",
                direction="Input",
                multiValue=True,
            ),
            arcpy.Parameter(
                displayName="Crop Table CSV",
                name="crop_table",
                datatype="DEFile",
                parameterType="Required",
                direction="Input",
            ),
            arcpy.Parameter(
                displayName="Output Folder",
                name="output_folder",
                datatype="DEFolder",
                parameterType="Required",
                direction="Input",
            ),
            arcpy.Parameter(
                displayName="Run Monte Carlo",
                name="run_mc",
                datatype="GPBoolean",
                parameterType="Optional",
                direction="Input",
            ),
            arcpy.Parameter(
                displayName="MC Samples",
                name="mc_samples",
                datatype="GPLong",
                parameterType="Optional",
                direction="Input",
            ),
            arcpy.Parameter(
                displayName="Value Uncertainty (%)",
                name="value_sd",
                datatype="GPDouble",
                parameterType="Optional",
                direction="Input",
            ),
            arcpy.Parameter(
                displayName="Depth Uncertainty (ft)",
                name="depth_sd",
                datatype="GPDouble",
                parameterType="Optional",
                direction="Input",
            ),
        ]
        return params

    def execute(self, parameters, messages):
        crop_raster = parameters[0].valueAsText
        flood_rasters = parameters[1].valueAsText.split(";")
        return_periods = [int(p) for p in parameters[2].valueAsText.split(";")]
        flood_months = [int(p) for p in parameters[3].valueAsText.split(";")]
        crop_table = parameters[4].valueAsText
        output_folder = parameters[5].valueAsText
        run_mc = parameters[6].value or False
        mc_samples = int(parameters[7].value) if parameters[7].value else 100
        value_sd = float(parameters[8].value) if parameters[8].value else 10.0
        depth_sd = float(parameters[9].value) if parameters[9].value else 0.1

        crop_df = pd.read_csv(crop_table)
        crop_inputs = {}
        for _, row in crop_df.iterrows():
            months = [int(m) for m in str(row["GrowingSeason"]).split(',') if m]
            crop_inputs[int(row["CropCode"])] = {
                "Value": float(row["Value"]),
                "GrowingSeason": months,
            }

        if not (
            len(flood_rasters)
            == len(return_periods)
            == len(flood_months)
        ):
            raise ValueError(
                "Mismatch among depth rasters, return periods and months"
            )

        depth_inputs = []
        flood_metadata = {}
        for idx, path in enumerate(flood_rasters):
            label = Path(path).stem
            depth_inputs.append((label, path))
            flood_metadata[label] = {
                "return_period": return_periods[idx],
                "flood_month": flood_months[idx],
            }

        excel_path, summaries, diagnostics, _ = process_flood_damage(
            crop_raster,
            depth_inputs,
            output_folder,
            100,
            crop_inputs,
            flood_metadata,
        )

        if run_mc:
            mc_results = run_monte_carlo(
                summaries,
                flood_metadata,
                mc_samples,
                value_sd,
                depth_sd,
            )
            with pd.ExcelWriter(excel_path, mode="a", engine="openpyxl") as writer:
                for label, df in mc_results.items():
                    df.to_excel(writer, sheet_name=f"MC_{label}", index=False)

        messages.addMessage(f"Results written to {excel_path}")

