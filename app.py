import streamlit as st
import os
import tempfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from collections import Counter
from utils.processing import process_flood_damage, run_monte_carlo

st.set_page_config(layout="wide")
st.title("🌾 Agricultural Flood Damage Estimator")

# Session state setup
for key in ["result_path", "summaries", "diagnostics", "monte_carlo_results", "crop_path", "depth_paths", "damage_rasters"]:
    if key not in st.session_state:
        st.session_state[key] = None

# 🔁 Reset
if st.button("🔁 Reset App"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# 📁 Inputs
crop_file = st.file_uploader("🌾 Upload USDA Cropland Raster", type=["tif", "img"])
depth_files = st.file_uploader("🌊 Upload One or More Flood Depth Rasters", type="tif", accept_multiple_files=True)
period_years = st.number_input("📆 Analysis Period (Years)", min_value=1, value=50)
samples = st.number_input("🎲 Monte Carlo Samples", min_value=10, value=100)
depth_sd = st.number_input("± Depth Uncertainty (ft)", value=0.1)
value_sd = st.number_input("± Crop Value Uncertainty (%)", value=10)

crop_inputs, flood_metadata = {}, {}

# 🌱 Crops
if crop_file:
    crop_path = tempfile.NamedTemporaryFile(delete=False, suffix=".tif").name
    with open(crop_path, "wb") as f:
        f.write(crop_file.read())
    st.session_state["crop_path"] = crop_path

    with rasterio.open(crop_path) as src:
        crop_data = src.read(1)
    counts = Counter(crop_data.flatten())
    codes = [c for c, _ in counts.most_common(10) if c != 0]

    st.markdown("### 🌱 Crop Value and Growing Season")
    for code in codes:
        val = st.number_input(f"Crop {code} – $/Acre", value=5500, step=100, key=f"val_{code}")
        season = st.multiselect(f"Crop {code} – Growing Months", list(range(1, 13)), default=list(range(4, 10)), key=f"season_{code}")
        crop_inputs[code] = {"Value": val, "GrowingSeason": season}

# 🌊 Flood metadata
if depth_files:
    st.markdown("### ⚙️ Flood Metadata")
    depth_paths = []
    for i, f in enumerate(depth_files):
        path = tempfile.NamedTemporaryFile(delete=False, suffix=".tif").name
        with open(path, "wb") as out:
            out.write(f.read())
        depth_paths.append(path)
        rp = st.number_input(f"Return Period (yrs) – {f.name}", min_value=1, value=100, key=f"rp_{i}")
        mo = st.number_input(f"Flood Month (1–12) – {f.name}", min_value=1, max_value=12, value=6, key=f"mo_{i}")
        flood_metadata[f.name] = {"return_period": rp, "flood_month": mo}
    st.session_state["depth_paths"] = depth_paths

# 🚀 Run direct damage estimate
if st.button("🚀 Run Flood Damage Estimator"):
    if not crop_file or not depth_files:
        st.error("Please upload cropland and depth rasters.")
    elif not crop_inputs:
        st.error("Please define crop values and growing seasons.")
    else:
        with st.spinner("Running direct damage estimation..."):
            try:
                result_path, summaries, diagnostics, damage_rasters = process_flood_damage(
                    st.session_state.crop_path,
                    st.session_state.depth_paths,
                    tempfile.mkdtemp(),
                    period_years,
                    crop_inputs,
                    flood_metadata
                )
                st.session_state["result_path"] = result_path
                st.session_state["summaries"] = summaries
                st.session_state["diagnostics"] = diagnostics
                st.session_state["damage_rasters"] = damage_rasters
                st.success("✅ Direct damage results ready.")
            except Exception as e:
                st.error(f"Error during damage analysis: {e}")

# 📈 Monte Carlo
if st.session_state.summaries and st.button("📈 Run Monte Carlo Analysis"):
    with st.spinner("Running Monte Carlo..."):
        try:
            results = run_monte_carlo(
                st.session_state.summaries,
                flood_metadata,
                samples,
                value_sd,
                depth_sd
            )
            st.session_state["monte_carlo_results"] = results
            st.success("✅ Monte Carlo complete.")
        except Exception as e:
            st.error(f"Monte Carlo error: {e}")

# 📤 Download Excel
if st.session_state.result_path:
    st.download_button("📥 Download Excel Summary", data=open(st.session_state.result_path, "rb"), file_name="ag_damage_summary.xlsx")

# 🧪 Diagnostics
if st.session_state.diagnostics:
    st.markdown("### 🧪 Diagnostics Log")
    st.dataframe(pd.DataFrame(st.session_state.diagnostics))

# 📊 Results
if st.session_state.summaries:
    for flood, df in st.session_state.summaries.items():
        st.subheader(f"📊 {flood} – Direct Damage")
        st.dataframe(df)

        if "CropCode" in df.columns and "EAD" in df.columns:
            fig, ax = plt.subplots()
            df.plot(kind="bar", x="CropCode", y="EAD", ax=ax, legend=False)
            ax.set_ylabel("EAD ($/yr)")
            ax.set_title(f"{flood} – EAD by Crop")
            st.pyplot(fig)

        # Optional damage raster visual
        if flood in st.session_state.damage_rasters:
            arr = st.session_state.damage_rasters[flood]
            st.markdown(f"🖼️ Crop Damage Raster (Normalized): {flood}")
            fig, ax = plt.subplots(figsize=(6, 4))
            cax = ax.imshow(arr, cmap="Reds", vmin=0, vmax=1)
            fig.colorbar(cax, ax=ax, label="Damage %")
            st.pyplot(fig)

# 📈 Monte Carlo Results
if st.session_state.monte_carlo_results:
    for flood, df_mc in st.session_state.monte_carlo_results.items():
        st.subheader(f"🎲 {flood} – Monte Carlo Summary")
        st.dataframe(df_mc)

        fig, ax = plt.subplots()
        df_mc.plot(x="CropCode", y=["EAD_MC_Mean", "EAD_MC_5th", "EAD_MC_95th"], kind="bar", ax=ax)
        ax.set_ylabel("EAD ($/yr)")
        ax.set_title(f"{flood} – Monte Carlo EAD Range")
        st.pyplot(fig)
