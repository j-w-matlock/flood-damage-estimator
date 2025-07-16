import streamlit as st
import os
import tempfile
import pandas as pd
from utils.processing import process_flood_damage, run_monte_carlo_analysis
import rasterio
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(layout="wide")
st.title("🌾 Agricultural Flood Damage Estimator")

# Initialize session state
for key in ["result_path", "summaries", "diagnostics", "crop_path", "depth_paths", "mc_results", "mc_path"]:
    if key not in st.session_state:
        st.session_state[key] = None

# 🔁 Reset app
if st.button("🔁 Reset App"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# 📁 File upload
crop_file = st.file_uploader("🌾 Upload USDA Cropland Raster (GeoTIFF)", type=["tif", "img"])
depth_files = st.file_uploader("🌊 Upload One or More Flood Depth Grids (GeoTIFF)", type="tif", accept_multiple_files=True)
period_years = st.number_input("📆 Analysis Period (Years)", value=50, min_value=1)
samples = st.number_input("🎲 Monte Carlo Samples", value=1000, min_value=100)

crop_inputs = {}
flood_metadata = {}

# 🌱 Crop setup
if crop_file:
    crop_path = tempfile.NamedTemporaryFile(delete=False, suffix=".tif").name
    with open(crop_path, "wb") as f:
        f.write(crop_file.read())
    st.session_state["crop_path"] = crop_path

    with rasterio.open(crop_path) as src:
        arr = src.read(1)
    counts = Counter(arr.flatten())
    most_common = [c for c, _ in counts.most_common(10) if c != 0]

    st.markdown("### 🌱 Define Crop Values and Growing Seasons")
    for code in most_common:
        val = st.number_input(f"Crop {code} – Value per Acre ($)", value=5500, step=100, key=f"val_{code}")
        months = st.multiselect(f"Crop {code} – Growing Months", list(range(1, 13)), default=list(range(4, 10)), key=f"grow_{code}")
        if not months:
            st.warning(f"⚠️ No months selected for crop {code}.")
        crop_inputs[code] = {"Value": val, "GrowingSeason": months}

# ⚙️ Flood raster metadata
if depth_files:
    st.markdown("### ⚙️ Flood Raster Metadata")
    depth_paths = []
    for i, f in enumerate(depth_files):
        temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".tif").name
        with open(temp_path, "wb") as out:
            out.write(f.read())
        depth_paths.append(temp_path)
        rp = st.number_input(f"Return Period for {f.name}", min_value=1, value=100, key=f"rp_{i}")
        mo = st.number_input(f"Flood Month for {f.name}", min_value=1, max_value=12, value=6, key=f"mo_{i}")
        flood_metadata[f.name] = {"return_period": rp, "flood_month": mo}
    st.session_state["depth_paths"] = depth_paths

# 🚀 Run main analysis
if st.button("🚀 Run Flood Damage Estimator"):
    if not crop_file or not depth_files:
        st.error("Please upload both cropland and depth rasters.")
    elif not crop_inputs:
        st.error("Please set crop values and growing months.")
    elif not flood_metadata:
        st.error("Missing flood metadata.")
    else:
        with st.spinner("Processing..."):
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
                st.session_state.mc_results = None
                st.success("✅ Flood damage estimation complete.")
            except Exception as e:
                st.error(f"❌ Error: {e}")

# 📊 Results Tab
if st.session_state.summaries:
    tab1, tab2 = st.tabs(["📊 EAD Results", "🎲 Monte Carlo Results"])

    with tab1:
        st.download_button("📥 Download Excel Summary", data=open(st.session_state.result_path, "rb"), file_name="ag_damage_summary.xlsx")
        if st.session_state.diagnostics:
            st.markdown("### 🧪 Diagnostics")
            st.dataframe(pd.DataFrame(st.session_state.diagnostics))
        for flood, df in st.session_state.summaries.items():
            st.subheader(f"{flood} Results")
            st.dataframe(df)
            if "CropCode" in df.columns and "EAD" in df.columns:
                fig, ax = plt.subplots()
                df.plot(kind="bar", x="CropCode", y="EAD", ax=ax, legend=False)
                ax.set_ylabel("Expected Annual Damage ($)")
                ax.set_title(f"EAD by Crop – {flood}")
                st.pyplot(fig)

    with tab2:
        if st.session_state.mc_results:
            mc_path = st.session_state.mc_path
            st.download_button("📥 Download Monte Carlo Excel", data=open(mc_path, "rb"), file_name="monte_carlo_results.xlsx")
            for flood, df in st.session_state.mc_results.items():
                st.subheader(f"🎲 Monte Carlo – {flood}")
                st.dataframe(df)
        else:
            if st.button("🎲 Run Monte Carlo Analysis"):
                with st.spinner("Running Monte Carlo..."):
                    try:
                        mc_results = run_monte_carlo_analysis(
                            st.session_state.summaries,
                            flood_metadata,
                            samples=samples
                        )
                        st.session_state.mc_results = mc_results

                        # Save MC to Excel
                        mc_path = os.path.join(tempfile.mkdtemp(), "monte_carlo_results.xlsx")
                        with pd.ExcelWriter(mc_path, engine="openpyxl") as writer:
                            for flood, df in mc_results.items():
                                df.to_excel(writer, sheet_name=flood, index=False)
                        st.session_state.mc_path = mc_path

                        st.success("✅ Monte Carlo complete.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Monte Carlo failed: {e}")
