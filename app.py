import streamlit as st
import os
import numpy as np
import pandas as pd
import rasterio
from utils.processing import process_flood_damage

# 🔐 Safe tkinter import
try:
    from tkinter import Tk, filedialog
    tk_available = True
except ImportError:
    tk_available = False

st.set_page_config(layout="wide")

if "crop_path" not in st.session_state:
    st.session_state.crop_path = None
if "depth_paths" not in st.session_state:
    st.session_state.depth_paths = []
if "output_dir" not in st.session_state:
    st.session_state.output_dir = None
if "summaries" not in st.session_state:
    st.session_state.summaries = {}
if "result_path" not in st.session_state:
    st.session_state.result_path = None
if "diagnostics" not in st.session_state:
    st.session_state.diagnostics = []
if "crop_inputs" not in st.session_state:
    st.session_state.crop_inputs = {}
if "flood_metadata" not in st.session_state:
    st.session_state.flood_metadata = {}
if "detected_crop_codes" not in st.session_state:
    st.session_state.detected_crop_codes = []

# Reset button at top
if st.button("🔄 Reset App"):
    st.session_state.crop_path = None
    st.session_state.depth_paths = []
    st.session_state.output_dir = None
    st.session_state.summaries = {}
    st.session_state.result_path = None
    st.session_state.diagnostics = []
    st.session_state.crop_inputs = {}
    st.session_state.flood_metadata = {}
    st.session_state.detected_crop_codes = []
    st.rerun()

st.title("🌾 Agricultural Flood Damage Estimator")

st.markdown("### 📁 Step 1: Select Input Files")
if tk_available:
    try:
        Tk().withdraw()
    except:
        pass

if tk_available and st.button("📂 Select CropScape Raster"):
    try:
        Tk().withdraw()
        crop_path = filedialog.askopenfilename(title="Select CropScape Raster", filetypes=[("GeoTIFF", "*.tif *.img")])
        if crop_path:
            st.session_state.crop_path = crop_path
            with rasterio.open(crop_path) as src:
                crop_data = src.read(1)
                unique_codes = list(np.unique(crop_data))
                st.session_state.detected_crop_codes = sorted([int(c) for c in unique_codes if c != 0])
            st.rerun()
    except:
        st.error("Could not open file dialog. Tkinter may not be supported.")

if st.session_state.crop_path:
    st.success(f"✅ CropScape Raster: {os.path.basename(st.session_state.crop_path)}")

if tk_available and st.button("🌊 Select Flood Depth Raster(s)"):
    try:
        Tk().withdraw()
        st.session_state.depth_paths = list(filedialog.askopenfilenames(title="Select Depth Rasters", filetypes=[("GeoTIFF", "*.tif *.img")]))
        st.rerun()
    except:
        st.error("Could not open file dialog. Tkinter may not be supported.")

if st.session_state.depth_paths:
    for path in st.session_state.depth_paths:
        st.markdown(f"✔️ {os.path.basename(path)}")

if tk_available and st.button("📁 Select Output Folder"):
    try:
        Tk().withdraw()
        st.session_state.output_dir = filedialog.askdirectory(title="Select Output Folder")
        st.rerun()
    except:
        st.error("Could not open folder dialog. Tkinter may not be supported.")

if st.session_state.output_dir:
    st.success(f"📤 Output Folder: {st.session_state.output_dir}")

if st.session_state.crop_path and st.session_state.depth_paths and st.session_state.detected_crop_codes:
    st.markdown("### 🌱 Step 2: Crop Values and Seasons")
    for code in st.session_state.detected_crop_codes:
        val = st.number_input(f"💵 Value per acre for crop {code}", min_value=0.0, value=5500.0, key=f"val_{code}")
        months = st.text_input(f"🌿 Growing season (comma-separated months 1-12) for crop {code}", value="4,5,6,7,8,9", key=f"months_{code}")
        st.session_state.crop_inputs[code] = {
            "Value": val,
            "GrowingSeason": [int(m.strip()) for m in months.split(",") if m.strip().isdigit() and 1 <= int(m.strip()) <= 12]
        }

    st.markdown("### 📅 Step 3: Flood Metadata")
    for path in st.session_state.depth_paths:
        fname = os.path.basename(path)
        month = st.number_input(f"Flood Month (1-12) for {fname}", min_value=1, max_value=12, value=6, key=f"month_{fname}")
        rp = st.number_input(f"Return Period (years) for {fname}", min_value=1, value=100, key=f"rp_{fname}")
        st.session_state.flood_metadata[fname] = {"flood_month": month, "return_period": rp}

    st.markdown("### 🧮 Step 4: Run Damage Estimator")
    period_years = st.number_input("Period of Analysis (years)", min_value=1, value=50)
    samples = st.number_input("Monte Carlo Samples", min_value=1, value=500)
    if st.button("🚀 Compute Damage Estimates"):
        with st.spinner("Running damage calculations..."):
            try:
                result_path, summaries, diagnostics = process_flood_damage(
                    st.session_state.crop_path,
                    st.session_state.depth_paths,
                    st.session_state.output_dir,
                    period_years,
                    samples,
                    st.session_state.crop_inputs,
                    st.session_state.flood_metadata
                )
                st.session_state.summaries = summaries
                st.session_state.result_path = result_path
                st.session_state.diagnostics = diagnostics
                st.success("✅ Damage estimates complete!")
            except Exception as e:
                st.error(f"❌ Error during processing: {e}")

if st.session_state.result_path and st.session_state.summaries:
    st.download_button("📥 Download Excel Summary", data=open(st.session_state.result_path, "rb"), file_name="ag_damage_summary.xlsx")

    st.markdown("## 🧪 Diagnostics Log")
    if st.session_state.diagnostics:
        st.dataframe(pd.DataFrame(st.session_state.diagnostics))
    else:
        st.info("✅ No issues detected in damage calculation.")

    for flood, df in st.session_state.summaries.items():
        st.subheader(f"📊 {flood} Summary")
        if df.empty:
            st.info("ℹ️ No crop damage estimated for this flood scenario.")
            continue
        st.dataframe(df)
        if "CropCode" in df.columns and "EAD" in df.columns:
            chart_df = df[["CropCode", "EAD"]].set_index("CropCode")
            st.bar_chart(chart_df, use_container_width=True)
        if "CropCode" in df.columns and "DirectDamage" in df.columns:
            chart_df2 = df[["CropCode", "DirectDamage"]].set_index("CropCode")
            st.bar_chart(chart_df2, use_container_width=True)
