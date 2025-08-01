"""ArcGIS Pro toolbox interface for flood damage processing."""

from pathlib import Path

import arcpy
import pandas as pd

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

