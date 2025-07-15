import streamlit as st
import os
import json
import tempfile
import shutil
import pandas as pd
from utils.processing import process_flood_damage
import rasterio
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import folium
from folium.plugins import Fullscreen
from streamlit_folium import st_folium
from matplotlib.cm import Reds
from matplotlib.colors import Normalize

st.set_page_config(layout="wide")
st.title("🌾 Agricultural Flood Damage Estimator")

# Session state init
for key in ["result_path", "summaries", "diagnostics", "crop_path", "depth_paths"]:
    if key not in st.session_state:
        st.session_state[key] = None

# Reset button
if st.button("🔁 Reset App"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# File uploads and settings
crop_file = st.file_uploader("🌾 Upload USDA Cropland Raster (GeoTIFF)", type=["tif", "img"])
depth_files = st.file_uploader("🌊 Upload One or More Flood Depth Grids (GeoTIFF)", type="tif", accept_multiple_files=True)
period_years = st.number_input("📆 Analysis Period (Years)", value=50, min_value=1)
samples = st.number_input("🎲 Monte Carlo Samples", value=100, min_value=10)

crop_inputs = {}
flood_metadata = {}

# Crop setup
if crop_file:
    crop_path = tempfile.NamedTemporaryFile(delete=False, suffix=".tif").name
    with open(crop_path, "wb") as f:
        f.write(crop_file.read())
    st.session_state["crop_path"] = crop_path

    with rasterio.open(crop_path) as src:
        arr = src.read(1)
    counts = Counter(arr.flatten())
    most_common = [c for c, _ in counts.most_common(10) if c != 0]

    st.markdown("### 🌱 Define Crop Values and Seasons")
    for code in most_common:
        val = st.number_input(f"Crop {code} — Value per Acre ($)", value=5500, step=100, key=f"val_{code}")
        months = st.multiselect(f"Crop {code} — Growing Season (months 1–12)", options=list(range(1, 13)), default=list(range(4, 10)), key=f"grow_{code}")
        if not months:
            st.warning(f"⚠️ No growing season months selected for crop {code}.")
        crop_inputs[code] = {"Value": val, "GrowingSeason": months}

# Flood metadata
if depth_files:
    st.markdown("### ⚙️ Flood Raster Settings")
    depth_paths = []
    for i, f in enumerate(depth_files):
        temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".tif").name
        with open(temp_path, "wb") as out:
            out.write(f.read())
        depth_paths.append(temp_path)
        rp = st.number_input(f"Return Period for {f.name} (years)", min_value=1, value=100, key=f"rp_{i}")
        mo = st.number_input(f"Flood Month for {f.name} (1–12)", min_value=1, max_value=12, value=6, key=f"mo_{i}")
        flood_metadata[f.name] = {"return_period": rp, "flood_month": mo}
    st.session_state["depth_paths"] = depth_paths

# Run processing
if st.button("🚀 Run Flood Damage Estimator"):
    if not crop_file or not depth_files:
        st.error("Please upload both cropland and depth raster files.")
    elif not crop_inputs:
        st.error("Please specify crop values and growing seasons.")
    elif not flood_metadata:
        st.error("Please provide return period and flood month for each raster.")
    else:
        with st.spinner("Processing flood damage estimates..."):
            temp_dir = tempfile.mkdtemp()
            try:
                result_path, summaries, diagnostics = process_flood_damage(
                    st.session_state.crop_path,
                    st.session_state.depth_paths,
                    temp_dir,
                    period_years,
                    samples,
                    crop_inputs,
                    flood_metadata
                )
                st.session_state.result_path = result_path
                st.session_state.summaries = summaries
                st.session_state.diagnostics = diagnostics
                st.success("✅ Damage estimates complete!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error during processing: {e}")

# Show results if available
if st.session_state.result_path and st.session_state.summaries:
    st.download_button("📥 Download Excel Summary", data=open(st.session_state.result_path, "rb"), file_name="ag_damage_summary.xlsx")

    st.markdown("## 🧪 Diagnostics Log")
    if st.session_state.diagnostics:
        st.dataframe(pd.DataFrame(st.session_state.diagnostics))
    else:
        st.info("✅ No issues detected in damage calculation.")

    for flood, df in st.session_state.summaries.items():
        st.subheader(f"📊 {flood} Summary")

        if not df.empty:
            df["EAD"] = (df["DollarsLost"] / period_years).round(2)

        st.dataframe(df)

        if "CropCode" in df.columns and "EAD" in df.columns:
            fig, ax = plt.subplots()
            df.plot(kind="bar", x="CropCode", y="EAD", ax=ax, legend=False)
            ax.set_ylabel("Expected Annual Damage ($)")
            ax.set_title(f"EAD per Crop for {flood}")
            st.pyplot(fig)

