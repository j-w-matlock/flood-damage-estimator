import streamlit as st
import os
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from collections import Counter
from utils.processing import process_flood_damage, run_monte_carlo_analysis

st.set_page_config(layout="wide")
st.title("🌾 Agricultural Flood Damage Estimator")

# Initialize session state
for key in ["result_path", "summaries", "diagnostics", "crop_path", "depth_paths", "mc_results"]:
    if key not in st.session_state:
        st.session_state[key] = None

# 🔁 Reset button
if st.button("🔁 Reset App"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# 📁 Upload rasters
crop_file = st.file_uploader("🌾 Upload USDA Cropland Raster", type=["tif", "img"])
depth_files = st.file_uploader("🌊 Upload Flood Depth Raster(s)", type="tif", accept_multiple_files=True)

# 📆 Analysis settings
period_years = st.number_input("🗓️ Period of Analysis (Years)", value=50, min_value=1)
samples = st.number_input("🎲 Monte Carlo Samples (for optional step)", value=1000, min_value=10)

# 🌱 Crop inputs
crop_inputs = {}
flood_metadata = {}

if crop_file:
    crop_path = tempfile.NamedTemporaryFile(delete=False, suffix=".tif").name
    with open(crop_path, "wb") as f:
        f.write(crop_file.read())
    st.session_state.crop_path = crop_path

    with rasterio.open(crop_path) as src:
        arr = src.read(1)
    counts = Counter(arr.flatten())
    most_common = [c for c, _ in counts.most_common(10) if c != 0]

    st.markdown("### 🌽 Define Crop Values and Growing Seasons")
    for code in most_common:
        val = st.number_input(f"Crop {code} – Value per Acre ($)", value=5500, step=100, key=f"val_{code}")
        months = st.multiselect(f"Crop {code} – Growing Season (1–12)", options=list(range(1, 13)),
                                default=list(range(4, 10)), key=f"grow_{code}")
        crop_inputs[code] = {"Value": val, "GrowingSeason": months}

# ⚙️ Flood metadata inputs
if depth_files:
    depth_paths = []
    for i, f in enumerate(depth_files):
        temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".tif").name
        with open(temp_path, "wb") as out:
            out.write(f.read())
        depth_paths.append(temp_path)
        rp = st.number_input(f"Return Period for {f.name} (years)", min_value=1, value=100, key=f"rp_{i}")
        mo = st.number_input(f"Flood Month for {f.name} (1–12)", min_value=1, max_value=12, value=6, key=f"mo_{i}")
        flood_metadata[f.name] = {"return_period": rp, "flood_month": mo}
    st.session_state.depth_paths = depth_paths

# 🚀 Run main analysis
if st.button("🚀 Run Flood Damage Estimator"):
    if not crop_file or not depth_files:
        st.error("Please upload cropland and flood depth rasters.")
    elif not crop_inputs:
        st.error("Define crop values and growing seasons.")
    elif not flood_metadata:
        st.error("Set flood metadata for all rasters.")
    else:
        with st.spinner("⏳ Processing flood damage estimates..."):
            temp_dir = tempfile.mkdtemp()
            try:
                result_path, summaries, diagnostics = process_flood_damage(
                    st.session_state.crop_path,
                    st.session_state.depth_paths,
                    temp_dir,
                    period_years,
                    crop_inputs,
                    flood_metadata
                )
                st.session_state.result_path = result_path
                st.session_state.summaries = summaries
                st.session_state.diagnostics = diagnostics
                st.session_state.mc_results = None  # Reset MC state
                st.success("✅ Damage estimates complete!")
            except Exception as e:
                st.error(f"❌ Error: {e}")

# 🧾 Results output
if st.session_state.result_path and st.session_state.summaries:
    st.download_button("📥 Download Excel Summary", data=open(st.session_state.result_path, "rb"), file_name="ag_damage_summary.xlsx")

    st.markdown("## 🧪 Diagnostics")
    if st.session_state.diagnostics:
        st.dataframe(pd.DataFrame(st.session_state.diagnostics))
    else:
        st.success("No issues detected.")

    filename_map = {
        os.path.splitext(os.path.basename(p))[0]: os.path.basename(p)
        for p in st.session_state.depth_paths
    }

    st.markdown("## 📊 Damage Summaries")
    for flood, df in st.session_state.summaries.items():
        st.subheader(f"{flood}")
        filename = filename_map.get(flood)
        return_period = flood_metadata.get(filename, {}).get("return_period", 100)
        df["EAD"] = (df["DollarsLost"] / return_period).round(2)
        st.dataframe(df)

        if "CropCode" in df.columns and "EAD" in df.columns:
            fig, ax = plt.subplots()
            df.plot(kind="bar", x="CropCode", y="EAD", ax=ax, legend=False)
            ax.set_ylabel("EAD ($)")
            ax.set_title(f"Expected Annual Damage – {flood}")
            st.pyplot(fig)

# 🎲 Optional Monte Carlo analysis
if st.session_state.summaries and not st.session_state.mc_results:
    if st.button("🎲 Run Monte Carlo Analysis"):
        with st.spinner("Running uncertainty analysis..."):
            mc_results = run_monte_carlo_analysis(
                st.session_state.summaries,
                flood_metadata,
                samples
            )
            st.session_state.mc_results = mc_results
            st.success("✅ Monte Carlo complete!")

# 📈 Monte Carlo Results
if st.session_state.mc_results:
    st.markdown("## 🎲 Monte Carlo Summary")
    for flood, df in st.session_state.mc_results.items():
        st.subheader(f"📉 {flood} – Monte Carlo Summary")
        st.dataframe(df)

        fig, ax = plt.subplots()
        df.plot(kind="bar", x="CropCode", y="Mean_EAD", ax=ax, legend=False)
        ax.set_ylabel("Mean EAD ($)")
        ax.set_title(f"Monte Carlo EAD – {flood}")
        st.pyplot(fig)

    # Download MC results
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        mc_path = tmp.name
        with pd.ExcelWriter(mc_path, engine="openpyxl") as writer:
            for flood, df in st.session_state.mc_results.items():
                df.to_excel(writer, sheet_name=flood, index=False)
        st.download_button("📥 Download Monte Carlo Summary", data=open(mc_path, "rb"),
                           file_name="mc_damage_summary.xlsx")
