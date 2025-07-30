import streamlit as st
import os
import tempfile
import pandas as pd
import rasterio
from utils.processing import (
    process_flood_damage,
    run_monte_carlo,
    rasterize_polygon_to_array,
)
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

st.set_page_config(layout="wide")
st.title("🌾 Agricultural Flood Damage Estimator")

# Session state init
for key in [
    "result_path",
    "summaries",
    "diagnostics",
    "crop_path",
    "depth_inputs",
    "damage_rasters",
    "label_map",
    "label_metadata",
    "temp_files",
    "temp_dir",
]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "temp_files" else None


    """Remove temporary files tracked in the session."""
    files = st.session_state.get("temp_files") or []
    for path in files:
        if path and os.path.exists(path):
            os.remove(path)
    st.session_state["temp_files"] = []


def cleanup_temp_dir():
    """Remove the temporary directory used for outputs."""
    tmp = st.session_state.get("temp_dir")
    if tmp:
        tmp.cleanup()
    st.session_state["temp_dir"] = None

# Reset
if st.sidebar.button("🔁 Reset App"):
    cleanup_temp_dir()
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# Sidebar Inputs
st.sidebar.header("🛠️ Settings")
mode = st.sidebar.radio(
    "Select Analysis Mode:",
    ["Direct Damages", "Monte Carlo Simulation"],
    help="Choose whether to run a straightforward flood loss calculation (Direct Damages) or include uncertainty using random simulations (Monte Carlo Simulation)"
)
crop_file = st.sidebar.file_uploader(
    "🌾 USDA CropScape Raster", type=["tif", "img"],
    help="Upload a CropScape raster that defines crop type per pixel"
)
depth_files = st.sidebar.file_uploader(
    "🌊 Flood Depth Grids", type=["tif"], accept_multiple_files=True,
    help="Upload one or more flood depth raster files (in feet)"
)
polygon_file = st.sidebar.file_uploader(
    "📐 Flood Extent Polygon", type=["zip", "geojson", "kml"],
    help="Upload a polygon defining flooded areas (zipped Shapefile, GeoJSON, or KML)"
)
period_years = st.sidebar.number_input(
    "📆 Analysis Period (Years)", min_value=1, value=50,
    help="Used for context in planning studies; not required for EAD computation"
)
samples = st.sidebar.number_input(
    "🎲 Monte Carlo Iterations", min_value=10, value=100,
    help="Number of random simulations per crop type to estimate uncertainty"
)
depth_sd = st.sidebar.number_input(
    "± Depth Uncertainty (ft)", value=0.1,
    help="Assumed standard deviation of flood depth error (used only in Monte Carlo)"
)
value_sd = st.sidebar.number_input(
    "± Crop Value Uncertainty (%)", value=10,
    help="Assumed variability in per-acre crop values (used only in Monte Carlo)"
)

crop_inputs, label_to_filename, label_to_metadata = {}, {}, {}

# Process cropland raster
if crop_file:
    crop_path = tempfile.NamedTemporaryFile(delete=False, suffix=".tif").name
    with open(crop_path, "wb") as f:
        f.write(crop_file.read())
    st.session_state.crop_path = crop_path
    st.session_state.temp_files.append(crop_path)

    with rasterio.open(crop_path) as src:
        arr = src.read(1)
    counts = Counter(arr.flatten())
    codes = [c for c, _ in counts.most_common(10) if c != 0]

    st.markdown("### 🌱 Crop Values and Growing Seasons")
    for code in codes:
        val = st.number_input(
            f"Crop {code} – $/Acre", value=5500, step=100,
            key=f"val_{code}",
            help="Enter average crop value per acre for this code"
        )
        season = st.multiselect(
            f"Crop {code} – Growing Months",
            list(range(1, 13)),
            default=list(range(4, 10)),
            key=f"season_{code}",
            help="Choose the active growing months when this crop is vulnerable to flooding"
        )
        crop_inputs[code] = {"Value": val, "GrowingSeason": season}

# Process flood rasters or polygon
if depth_files or polygon_file:
    st.markdown("### ⚙️ Flood Raster Settings")
    depth_inputs = []

    if depth_files:
        for i, f in enumerate(depth_files):
            path = tempfile.NamedTemporaryFile(delete=False, suffix=".tif").name
            with open(path, "wb") as out:
                out.write(f.read())
            depth_inputs.append(path)
            st.session_state.temp_files.append(path)
            rp = st.number_input(
                f"Return Period: {f.name}", min_value=1, value=100,
                key=f"rp_{i}",
                help="How often this flood event is expected to occur (e.g., 100 for 1-in-100 year flood)"
            )
            mo = st.number_input(
                f"Flood Month: {f.name}", min_value=1, max_value=12,
                value=6, key=f"mo_{i}",
                help="Month of flood to compare against crop growing season"
            )
            label = os.path.splitext(os.path.basename(path))[0]
            label_to_filename[label] = f.name
            label_to_metadata[label] = {"return_period": rp, "flood_month": mo}

    if polygon_file and crop_file:
        poly_ext = os.path.splitext(polygon_file.name)[1]
        poly_path = tempfile.NamedTemporaryFile(delete=False, suffix=poly_ext).name
        with open(poly_path, "wb") as out:
            out.write(polygon_file.read())
        st.session_state.temp_files.append(poly_path)

        rp = st.number_input(
            f"Return Period: {polygon_file.name}", min_value=1, value=100,
            key="rp_polygon",
            help="How often this flood event is expected to occur"
        )
        mo = st.number_input(
            f"Flood Month: {polygon_file.name}", min_value=1, max_value=12,
            value=6, key="mo_polygon",
            help="Month of flood to compare against crop growing season"
        )

        depth_arr = rasterize_polygon_to_array(poly_path, st.session_state.crop_path)
        label = os.path.splitext(os.path.basename(poly_path))[0]
        depth_inputs.append((label, depth_arr))
        label_to_filename[label] = polygon_file.name
        label_to_metadata[label] = {"return_period": rp, "flood_month": mo}

    st.session_state.depth_inputs = depth_inputs
    st.session_state.label_map = label_to_filename
    st.session_state.label_metadata = label_to_metadata

# Direct Damages Mode
if mode == "Direct Damages":
    if st.button("🚀 Run Flood Damage Estimator"):
        if not (crop_file and (depth_files or polygon_file)):
            st.error("❌ Please upload a cropland raster and at least one flood source.")
        else:
            cleanup_temp_dir()
            st.session_state.temp_dir = tempfile.TemporaryDirectory()
            with st.spinner("🔄 Processing flood damages..."):
                try:
                    result_path, summaries, diagnostics, damage_rasters = process_flood_damage(
                        st.session_state.crop_path,
                        st.session_state.depth_inputs,
                        st.session_state.temp_dir.name,
                        period_years,
                        crop_inputs,
                        st.session_state.label_metadata
                    )
                    st.session_state.result_path = result_path
                    st.session_state.summaries = summaries
                    st.session_state.diagnostics = diagnostics
                    st.session_state.damage_rasters = damage_rasters
                    st.success("✅ Direct damage analysis complete.")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

    if st.session_state.result_path and "damage_rasters" in st.session_state:
        st.markdown("---")
        st.header("📈 Results & Visualizations")

        for label, df in st.session_state.summaries.items():
            st.subheader(f"📋 Summary for {label}")
            with st.expander("ℹ️ Column Definitions"):
                st.markdown("""
                - **CropCode**: CropScape code for crop type.
                - **FloodedAcres**: Area affected (1 pixel ≈ 0.222 acres).
                - **ValuePerAcre**: Input value per acre for the crop.
                - **DollarsLost**: Total crop damage.
                - **EAD**: Expected Annual Damage = DollarsLost ÷ ReturnPeriod.
                """)
            st.dataframe(df)

        if st.session_state.diagnostics:
            st.subheader("🛠️ Diagnostics")
            st.dataframe(pd.DataFrame(st.session_state.diagnostics))

        for label, arr in st.session_state.damage_rasters.items():
            st.subheader(f"🗺️ Damage Raster – {label}")
            fig, ax = plt.subplots()
            cax = ax.imshow(arr, cmap='YlOrRd')
            fig.colorbar(cax, ax=ax, label="Damage Ratio")
            ax.set_title(f"Damage Raster: {label}")
            st.pyplot(fig)

        with open(st.session_state.result_path, "rb") as file:
            st.download_button(
                label="📥 Download Results Excel File",
                data=file,
                file_name=os.path.basename(st.session_state.result_path),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

# Monte Carlo Mode
elif mode == "Monte Carlo Simulation":
    if st.button("🧪 Run Monte Carlo Simulation"):
        if not (crop_file and (depth_files or polygon_file)):
            st.error("❌ Please upload a cropland raster and at least one flood source.")
        else:
            cleanup_temp_dir()
            st.session_state.temp_dir = tempfile.TemporaryDirectory()
            with st.spinner("🔬 Running Monte Carlo..."):
                try:
                    result_path, summaries, diagnostics, damage_rasters = process_flood_damage(
                        st.session_state.crop_path,
                        st.session_state.depth_inputs,
                        st.session_state.temp_dir.name,
                        period_years,
                        crop_inputs,
                        st.session_state.label_metadata
                    )
                    mc_results = run_monte_carlo(
                        summaries,
                        st.session_state.label_metadata,
                        samples,
                        value_sd,
                        depth_sd
                    )
                    st.session_state.result_path = result_path

                    st.markdown("---")
                    st.header("📉 Monte Carlo EAD Results")

                    for label, df in mc_results.items():
                        st.subheader(f"🧪 MC Summary for {label}")
                        with st.expander("ℹ️ Column Definitions"):
                            st.markdown("""
                            - **CropCode**: CropScape code for crop type.
                            - **EAD_MC_Mean**: Mean simulated EAD from Monte Carlo.
                            - **EAD_MC_5th / 95th**: Uncertainty bounds (percentiles).
                            - **Original_EAD**: Deterministic EAD value for comparison.
                            """)
                        st.dataframe(df)

                    with pd.ExcelWriter(result_path, mode="a", engine="openpyxl") as writer:
                        for label, df in mc_results.items():
                            df.to_excel(writer, sheet_name=f"MC_{label}", index=False)

                    with open(result_path, "rb") as file:
                        st.download_button(
                            label="📥 Download Monte Carlo Excel File",
                            data=file,
                            file_name=os.path.basename(result_path),
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )

                    st.success("✅ Monte Carlo analysis complete.")

                except Exception as e:
                    st.error(f"⚠️ Monte Carlo error: {e}")
