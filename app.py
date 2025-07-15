import streamlit as st
import os
import json
import tempfile
import shutil
import pandas as pd
from utils.processing import process_flood_damage
import rasterio
import matplotlib.pyplot as plt
import numpy as np
import folium
from folium.plugins import Fullscreen
from streamlit_folium import st_folium

# ✅ App started
st.set_page_config(layout="wide")
st.title("🌾 Agricultural Flood Damage Estimator")

# 🧹 Reset at the top
if st.button("🔄 Reset App"):
    st.session_state.clear()
    st.rerun()

# 🔁 Session state
if "result_path" not in st.session_state:
    st.session_state.result_path = None
    st.session_state.summaries = None
    st.session_state.diagnostics = None

# File uploads
crop_file = st.file_uploader("🌾 Upload USDA Cropland Raster (GeoTIFF)", type=["tif", "img"])
depth_files = st.file_uploader("🌊 Upload One or More Flood Depth Grids (GeoTIFF)", type="tif", accept_multiple_files=True)
period_years = st.number_input("📆 Analysis Period (Years)", value=50, min_value=1)
samples = st.number_input("🎲 Monte Carlo Samples", value=500, min_value=10)

crop_inputs = {}
flood_metadata = {}

# -------------------------------------
# 🌱 Crop Setup
# -------------------------------------
if crop_file:
    crop_path = tempfile.mktemp(suffix=".tif")
    with open(crop_path, "wb") as f:
        f.write(crop_file.read())
    with rasterio.open(crop_path) as src:
        arr = src.read(1)
    codes = pd.Series(arr.flatten()).value_counts().head(10)
    st.markdown("### 🌱 Define Crop Values and Growing Seasons")
    for code in codes.index:
        val = st.number_input(f"Crop {code} — Value per Acre ($)", value=5500, step=100, key=f"val_{code}")
        months = st.multiselect(f"Crop {code} — Growing Season (months 1–12)", options=list(range(1, 13)), default=list(range(4, 10)), key=f"grow_{code}")
        crop_inputs[code] = {"Value": val, "GrowingSeason": months}

# -------------------------------------
# 🌊 Flood Metadata Input
# -------------------------------------
if depth_files:
    st.markdown("### ⚙️ Flood Raster Settings")
    for i, f in enumerate(depth_files):
        col1, col2 = st.columns(2)
        with col1:
            rp = st.number_input(f"Return Period for {f.name} (years)", min_value=1, value=100, key=f"rp_{i}")
        with col2:
            mo = st.number_input(f"Flood Month for {f.name} (1–12)", min_value=1, max_value=12, value=6, key=f"mo_{i}")
        flood_metadata[f.name] = {"return_period": rp, "flood_month": mo}

# -------------------------------------
# 🚀 Run Estimator
# -------------------------------------
if st.button("🚀 Run Flood Damage Estimator"):
    if not crop_file or not depth_files:
        st.error("Please upload cropland and depth raster files.")
    elif not crop_inputs:
        st.error("Please specify crop values and seasons.")
    else:
        with st.spinner("Processing flood damage estimates..."):
            temp_dir = tempfile.mkdtemp()
            depth_paths = []
            for f in depth_files:
                dp = os.path.join(temp_dir, f.name)
                with open(dp, "wb") as out:
                    out.write(f.read())
                depth_paths.append(dp)

            try:
                result_path, summaries, diagnostics = process_flood_damage(
                    crop_path, depth_paths, temp_dir, period_years, samples, crop_inputs, flood_metadata
                )
                st.session_state.result_path = result_path
                st.session_state.summaries = summaries
                st.session_state.diagnostics = diagnostics
                st.success("✅ Damage estimates complete!")
            except Exception as e:
                st.error(f"❌ Error during processing: {e}")

# -------------------------------------
# 📤 Show Results
# -------------------------------------
if st.session_state.result_path and st.session_state.summaries:
    st.download_button("📥 Download Excel Summary", data=open(st.session_state.result_path, "rb"), file_name="ag_damage_summary.xlsx")

    st.markdown("## 🧪 Diagnostics Log")
    if st.session_state.diagnostics:
        st.dataframe(pd.DataFrame(st.session_state.diagnostics))
    else:
        st.info("✅ No diagnostics warnings.")

    for flood, df in st.session_state.summaries.items():
        st.subheader(f"📊 {flood} Summary")
        st.dataframe(df)

        if "CropCode" in df.columns and "DollarsLost" in df.columns:
            fig, ax = plt.subplots()
            df.plot(kind="bar", x="CropCode", y="DollarsLost", ax=ax, legend=False)
            ax.set_ylabel("Total Loss ($)")
            ax.set_title(f"Crop Losses for {flood}")
            st.pyplot(fig)

        damage_path = os.path.join(os.path.dirname(st.session_state.result_path), f"damage_{flood}.tif")
        if os.path.exists(damage_path):
            with rasterio.open(damage_path) as dsrc:
                damage = dsrc.read(1)
                transform = dsrc.transform
            fig, ax = plt.subplots()
            ax.imshow(np.where(damage > 0.01, damage, np.nan), cmap="Reds", vmin=0, vmax=1)
            ax.set_title("Damage Footprint")
            ax.axis("off")
            st.pyplot(fig)

            bounds = rasterio.transform.array_bounds(damage.shape[0], damage.shape[1], transform)
            center_lat = (bounds[1] + bounds[3]) / 2
            center_lon = (bounds[0] + bounds[2]) / 2
            m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
            Fullscreen().add_to(m)
            folium.raster_layers.ImageOverlay(
                image=(damage * 255).astype(np.uint8),
                bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
                opacity=0.6,
                name="Damage",
            ).add_to(m)
            folium.LayerControl().add_to(m)
            st.markdown("### 🗺️ Interactive Damage Map")
            st_folium(m, width=800, height=500)
