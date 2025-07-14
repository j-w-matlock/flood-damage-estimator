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

print("✅ Streamlit app.py started")

@st.cache_data
def save_uploaded_file(uploadedfile, filename):
    tmp_dir = os.path.join("temp_data")
    os.makedirs(tmp_dir, exist_ok=True)
    path = os.path.join(tmp_dir, filename)
    with open(path, "wb") as f:
        f.write(uploadedfile.read())
    return path

st.set_page_config(layout="wide")
st.title("🌾 Agricultural Flood Damage Estimator")

crop_file = st.file_uploader("🌾 Upload USDA Cropland Raster (GeoTIFF)", type=["tif", "img"])
depth_files = st.file_uploader("🌊 Upload One or More Flood Depth Grids (GeoTIFF)", type="tif", accept_multiple_files=True)

period_years = st.number_input("📆 Analysis Period (Years)", value=50, min_value=1)
samples = st.number_input("🎲 Monte Carlo Samples", value=100, min_value=10)

crop_inputs = {}
flood_metadata = {}

# -------------------------------------
# 🌱 Crop Setup
# -------------------------------------
if crop_file:
    crop_path = save_uploaded_file(crop_file, "crop.tif")
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
        st.error("Please upload both cropland and depth raster files.")
    elif not crop_inputs:
        st.error("Please specify crop values and growing seasons.")
    elif not flood_metadata:
        st.error("Please provide return period and flood month for each raster.")
    else:
        with st.spinner("Processing flood damage estimates..."):
            temp_dir = tempfile.mkdtemp()
            depth_paths = [save_uploaded_file(f, f.name) for f in depth_files]
            try:
                result_path, summaries, diagnostics = process_flood_damage(
                    crop_path, depth_paths, temp_dir, period_years, samples, crop_inputs, flood_metadata
                )
            except Exception as e:
                st.error(f"❌ Error during processing: {e}")
                st.stop()

        st.success("✅ Damage estimates complete!")
        st.download_button("📥 Download Excel Summary", data=open(result_path, "rb"), file_name="ag_damage_summary.xlsx")

        # Diagnostics Table
        st.markdown("## 🧪 Diagnostics Log")
        if diagnostics:
            st.dataframe(pd.DataFrame(diagnostics))
        else:
            st.info("✅ No issues detected in damage calculation.")

        # Results and Visualization
        for flood, df in summaries.items():
            st.subheader(f"📊 {flood} Summary")
            if df.empty:
                st.info("ℹ️ No crop damage estimated for this flood scenario.")
                continue

            st.dataframe(df)

            if "CropCode" in df.columns and "DollarsLost" in df.columns:
                fig, ax = plt.subplots()
                df.plot(kind="bar", x="CropCode", y="DollarsLost", ax=ax, legend=False)
                ax.set_ylabel("Total Loss ($)")
                ax.set_title(f"Crop Losses for {flood}")
                st.pyplot(fig)

            # 🌄 PNG Visualization
            st.markdown("### 🖼️ Overlap Visualization (Crop / Depth / Damage)")
            damage_path = os.path.join(temp_dir, f"damage_{flood}.tif")
            depth_file_matches = [f for f in depth_paths if flood in os.path.basename(f)]
            if not depth_file_matches:
                st.warning(f"No matching depth file found for {flood}")
                continue
            depth_file = depth_file_matches[0]

            with rasterio.open(damage_path) as dsrc:
                damage = dsrc.read(1)
                transform = dsrc.transform
                crs = dsrc.crs

            with rasterio.open(depth_file) as depth_src:
                depth = depth_src.read(1)

            crop_arr = (damage > 0).astype(int)

            fig, axs = plt.subplots(1, 3, figsize=(16, 5))
            axs[0].imshow(np.where(crop_arr > 0, crop_arr, np.nan), cmap="Greens", interpolation="none")
            axs[0].set_title("🌾 Crop Presence")
            axs[1].imshow(np.where(depth > 0.01, depth, np.nan), cmap="Blues", interpolation="none")
            axs[1].set_title("🌊 Flood Depth")
            axs[2].imshow(np.where(damage > 0.01, damage, np.nan), cmap="Reds", vmin=0, vmax=1, interpolation="none")
            axs[2].set_title("💥 Damage Estimate")
            for ax in axs:
                ax.axis("off")

            overlap_path = os.path.join(temp_dir, f"overlap_{flood}.png")
            fig.savefig(overlap_path, bbox_inches="tight")
            st.pyplot(fig)
            with open(overlap_path, "rb") as f:
                st.download_button(f"📷 Download Overlap PNG for {flood}", f, file_name=f"overlap_{flood}.png")

            # 🗺️ Folium Map
            st.markdown("### 🌍 Interactive Map")
            bounds = rasterio.transform.array_bounds(damage.shape[0], damage.shape[1], transform)
            center_lat = (bounds[1] + bounds[3]) / 2
            center_lon = (bounds[0] + bounds[2]) / 2

            m = folium.Map(location=[center_lat, center_lon], zoom_start=12, control_scale=True)
            Fullscreen().add_to(m)

            norm = Normalize(vmin=0, vmax=1)
            damage_uint8 = (np.clip(damage, 0, 1) * 255).astype(np.uint8)

            folium.raster_layers.ImageOverlay(
                image=damage_uint8,
                bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
                opacity=0.6,
                colormap=lambda x: tuple(Reds(norm(x))),
                name="Damage",
                interactive=True,
            ).add_to(m)

            folium.LayerControl().add_to(m)
            st_folium(m, width=800, height=500)

        # Cleanup (optional, avoids buildup)
        shutil.rmtree(temp_dir, ignore_errors=True)
