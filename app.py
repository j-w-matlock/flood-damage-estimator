import streamlit as st
import os
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils.processing import process_flood_damage

st.set_page_config(layout="wide")
st.title("🌾 Agricultural Flood Damage Estimator")

# Session state init
for key in ["result_path", "summaries", "diagnostics", "crop_file", "depth_files"]:
    if key not in st.session_state:
        st.session_state[key] = None

# File upload UI
crop_file = st.file_uploader("🌾 Upload USDA CropScape Raster (GeoTIFF)", type=["tif", "img"])
depth_files = st.file_uploader("🌊 Upload One or More Flood Depth Grids", type=["tif"], accept_multiple_files=True)

# Reset app button
if st.button("🔄 Reset App"):
    st.session_state.clear()
    st.rerun()

# Analysis settings
period_years = st.number_input("📆 Period of Analysis (Years)", value=50, min_value=1)
samples = st.number_input("🎲 Monte Carlo Samples (not currently used)", value=100, min_value=10)

# Crop inputs
crop_inputs = {}
if crop_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
        tmp.write(crop_file.read())
        crop_path = tmp.name
    st.session_state.crop_file = crop_path
    st.markdown("### 🌱 Define Crop Value & Season")
    arr = None
    try:
        import rasterio
        from collections import Counter
        with rasterio.open(crop_path) as src:
            arr = src.read(1)
        common = [c for c, _ in Counter(arr.flatten()).most_common(10) if c != 0]
        for code in common:
            val = st.number_input(f"Crop {code} – Value per Acre ($)", value=5500, step=100, key=f"val_{code}")
            months = st.multiselect(f"Crop {code} – Growing Season (1–12)", options=list(range(1, 13)),
                                    default=list(range(4, 10)), key=f"grow_{code}")
            crop_inputs[code] = {"Value": val, "GrowingSeason": months}
    except Exception as e:
        st.error(f"Raster error: {e}")

# Flood metadata
flood_metadata = {}
if depth_files:
    st.markdown("### ⚙️ Flood Raster Settings")
    depth_paths = []
    for i, f in enumerate(depth_files):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
            tmp.write(f.read())
            depth_paths.append(tmp.name)
        rp = st.number_input(f"Return Period for {f.name} (years)", min_value=1, value=100, key=f"rp_{i}")
        mo = st.number_input(f"Flood Month for {f.name} (1–12)", min_value=1, max_value=12, value=6, key=f"mo_{i}")
        flood_metadata[f.name] = {"return_period": rp, "flood_month": mo}
    st.session_state.depth_files = depth_paths

# Run processing
if st.button("🚀 Run Damage Estimator"):
    if not st.session_state.crop_file or not st.session_state.depth_files:
        st.error("❌ Upload both Crop and Depth rasters.")
    elif not crop_inputs:
        st.error("❌ Define crop values and seasons.")
    elif not flood_metadata:
        st.error("❌ Provide flood metadata for each raster.")
    else:
        with st.spinner("⏳ Calculating damages..."):
            temp_dir = tempfile.mkdtemp()
            try:
                path, summaries, diagnostics = process_flood_damage(
                    st.session_state.crop_file,
                    st.session_state.depth_files,
                    temp_dir,
                    period_years,
                    samples,
                    crop_inputs,
                    flood_metadata
                )
                st.session_state.result_path = path
                st.session_state.summaries = summaries
                st.session_state.diagnostics = diagnostics
                st.success("✅ Damage estimates complete!")
            except Exception as e:
                st.error(f"❌ Error during processing: {e}")

# Show results
if st.session_state.result_path and st.session_state.summaries:
    st.download_button("📥 Download Excel Summary", data=open(st.session_state.result_path, "rb"),
                       file_name="ag_damage_summary.xlsx")

    st.markdown("## 🧪 Diagnostics Log")
    if st.session_state.diagnostics:
        st.dataframe(pd.DataFrame(st.session_state.diagnostics))
    else:
        st.info("✅ No issues detected.")

    for flood, df in st.session_state.summaries.items():
        st.subheader(f"📊 {flood} Summary")
        if df.empty:
            st.warning("No crop damage detected.")
            continue

        df["EAD ($)"] = df["EAD"].round(2)
        df["Total Loss ($)"] = df["DollarsLost"].round(2)
        df["Flooded Acres"] = df["FloodedAcres"].round(2)
        st.dataframe(df[["CropCode", "Flooded Acres", "AvgDamage", "Total Loss ($)", "EAD ($)"]])

        if "CropCode" in df.columns and "DollarsLost" in df.columns:
            fig, ax = plt.subplots()
            df.plot(kind="bar", x="CropCode", y="EAD", ax=ax, legend=False)
            ax.set_ylabel("EAD ($)")
            ax.set_title(f"EAD by Crop – {flood}")
            st.pyplot(fig)
