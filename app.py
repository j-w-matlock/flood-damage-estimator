import streamlit as st
import os
import tempfile
import pandas as pd
import rasterio
from utils.processing import process_flood_damage, run_monte_carlo
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

st.set_page_config(layout="wide")
st.title("🌾 Agricultural Flood Damage Estimator")

# Session state init
for key in ["result_path", "summaries", "diagnostics", "crop_path", "depth_paths", "damage_rasters", "label_map", "label_metadata"]:
    if key not in st.session_state:
        st.session_state[key] = None

# Reset
if st.sidebar.button("🔁 Reset App"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# Sidebar Inputs
st.sidebar.header("🛠️ Settings")
crop_file = st.sidebar.file_uploader("🌾 USDA Cropland Raster", type=["tif", "img"])
depth_files = st.sidebar.file_uploader("🌊 Flood Depth Grids", type=["tif"], accept_multiple_files=True)
period_years = st.sidebar.number_input("📆 Analysis Period (Years)", min_value=1, value=50)
samples = st.sidebar.number_input("🎲 Monte Carlo Iterations", min_value=10, value=100)
depth_sd = st.sidebar.number_input("± Depth Uncertainty (ft)", value=0.1)
value_sd = st.sidebar.number_input("± Crop Value Uncertainty (%)", value=10)

crop_inputs, label_to_filename, label_to_metadata = {}, {}, {}

# Process cropland raster
if crop_file:
    crop_path = tempfile.NamedTemporaryFile(delete=False, suffix=".tif").name
    with open(crop_path, "wb") as f:
        f.write(crop_file.read())
    st.session_state.crop_path = crop_path

    with rasterio.open(crop_path) as src:
        arr = src.read(1)
    counts = Counter(arr.flatten())
    codes = [c for c, _ in counts.most_common(10) if c != 0]

    st.markdown("### 🌱 Crop Values and Growing Seasons")
    for code in codes:
        val = st.number_input(f"Crop {code} – $/Acre", value=5500, step=100, key=f"val_{code}")
        season = st.multiselect(f"Crop {code} – Growing Months", list(range(1, 13)), default=list(range(4, 10)), key=f"season_{code}")
        crop_inputs[code] = {"Value": val, "GrowingSeason": season}

# Process flood rasters
if depth_files:
    st.markdown("### ⚙️ Flood Raster Settings")
    depth_paths = []
    for i, f in enumerate(depth_files):
        path = tempfile.NamedTemporaryFile(delete=False, suffix=".tif").name
        with open(path, "wb") as out:
            out.write(f.read())
        depth_paths.append(path)
        rp = st.number_input(f"Return Period: {f.name}", min_value=1, value=100, key=f"rp_{i}")
        mo = st.number_input(f"Flood Month: {f.name}", min_value=1, max_value=12, value=6, key=f"mo_{i}")
        label = os.path.splitext(os.path.basename(path))[0]
        label_to_filename[label] = f.name
        label_to_metadata[label] = {"return_period": rp, "flood_month": mo}

    st.session_state.depth_paths = depth_paths
    st.session_state.label_map = label_to_filename
    st.session_state.label_metadata = label_to_metadata

# Run direct damage estimate
if st.button("🚀 Run Flood Damage Estimator"):
    if not (crop_file and depth_files):
        st.error("❌ Please upload both cropland and flood rasters.")
    else:
        with st.spinner("🔄 Processing flood damages..."):
            try:
                result_path, summaries, diagnostics, damage_rasters = process_flood_damage(
                    st.session_state.crop_path,
                    st.session_state.depth_paths,
                    tempfile.mkdtemp(),
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

# Results section
if st.session_state.result_path and "damage_rasters" in st.session_state:
    st.markdown("---")
    st.header("📈 Results & Visualizations")

    # Summary tables
    for label, df in st.session_state.summaries.items():
        st.subheader(f"📋 Summary for {label}")
        st.dataframe(df)

    # Diagnostics
    if st.session_state.diagnostics:
        st.subheader("🛠️ Diagnostics")
        st.dataframe(pd.DataFrame(st.session_state.diagnostics))

    # Damage raster visualizations
    for label, arr in st.session_state.damage_rasters.items():
        st.subheader(f"🗺️ Damage Raster – {label}")
        fig, ax = plt.subplots()
        cax = ax.imshow(arr, cmap='YlOrRd')
        fig.colorbar(cax, ax=ax, label="Damage Ratio")
        ax.set_title(f"Damage Raster: {label}")
        st.pyplot(fig)

    # Download Excel
    with open(st.session_state.result_path, "rb") as file:
        st.download_button(
            label="📥 Download Results Excel File",
            data=file,
            file_name=os.path.basename(st.session_state.result_path),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# Monte Carlo (optional)
if st.session_state.summaries and st.button("🧪 Run Monte Carlo Simulation"):
    with st.spinner("🔬 Running Monte Carlo..."):
        try:
            mc_results = run_monte_carlo(
                st.session_state.summaries,
                st.session_state.label_metadata,
                samples,
                value_sd,
                depth_sd
            )

            st.markdown("---")
            st.header("📉 Monte Carlo EAD Results")

            for label, df in mc_results.items():
                st.subheader(f"🧪 MC Summary for {label}")
                st.dataframe(df)

            # Export to Excel (append to existing file)
            with pd.ExcelWriter(st.session_state.result_path, mode="a", engine="openpyxl") as writer:
                for label, df in mc_results.items():
                    df.to_excel(writer, sheet_name=f"MC_{label}", index=False)

            st.success("✅ Monte Carlo results added to Excel.")

        except Exception as e:
            st.error(f"⚠️ Monte Carlo error: {e}")
