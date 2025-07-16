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
for key in ["result_path", "summaries", "diagnostics", "crop_path", "depth_paths"]:
    if key not in st.session_state:
        st.session_state[key] = None

# Reset
if st.button("🔁 Reset App"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# File upload
crop_file = st.file_uploader("🌾 USDA Cropland Raster", type=["tif", "img"])
depth_files = st.file_uploader("🌊 Flood Depth Grids", type=["tif"], accept_multiple_files=True)
period_years = st.number_input("📆 Analysis Period (Years)", min_value=1, value=50)
samples = st.number_input("🎲 Iterations", min_value=10, value=100)
depth_sd = st.number_input("± Depth Uncertainty (ft)", value=0.1)
value_sd = st.number_input("± Crop Value Uncertainty (%)", value=10)

crop_inputs, flood_metadata = {}, {}

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

    st.markdown("### 🌱 Crop Values and Seasons")
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
        flood_metadata[f.name] = {"return_period": rp, "flood_month": mo}
    st.session_state.depth_paths = depth_paths

# Run direct damage estimate
if st.button("🚀 Run Flood Damage Estimator"):
    if not (crop_file and depth_files):
        st.error("Both cropland and flood rasters are required.")
    else:
        with st.spinner("Processing flood damages..."):
            try:
                result_path, summaries, diagnostics, damage_rasters = process_flood_damage(
                    st.session_state.crop_path,
                    st.session_state.depth_paths,
                    tempfile.mkdtemp(),
                    period_years,
                    crop_inputs,
                    flood_metadata
                )
                st.session_state.result_path = result_path
                st.session_state.summaries = summaries
                st.session_state.diagnostics = diagnostics
                st.session_state.damage_rasters = damage_rasters
                st.success("✅ Direct damage analysis complete.")
            except Exception as e:
                st.error(f"❌ Error: {e}")
                # 📊 Visualize Results After Run
if st.session_state.result_path and "damage_rasters" in st.session_state:
    st.markdown("---")
    st.header("📈 Results & Visualizations")

    # Show summary tables
    for label, df in st.session_state.summaries.items():
        st.subheader(f"📋 Summary for {label}")
        st.dataframe(df)

    # Show diagnostics if any
    if st.session_state.diagnostics:
        st.subheader("🛠️ Diagnostics")
        st.dataframe(pd.DataFrame(st.session_state.diagnostics))

    # Show damage raster images
    st.subheader("🖼️ Damage Maps")
    for name, raster_path in st.session_state.damage_rasters.items():
        try:
            with rasterio.open(raster_path) as src:
                damage_arr = src.read(1)
                masked = np.ma.masked_where(damage_arr == 0, damage_arr)
                fig, ax = plt.subplots(figsize=(6, 4))
                cmap = plt.cm.get_cmap("Reds")
                im = ax.imshow(masked, cmap=cmap)
                ax.set_title(f"Crop Damage Raster – {name}")
                plt.colorbar(im, ax=ax, label="% Damage")
                st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not render {name}: {e}")

    # Export final Excel
    with open(st.session_state.result_path, "rb") as file:
        st.download_button(
            label="📥 Download Results Excel File",
            data=file,
            file_name=os.path.basename(st.session_state.result_path),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # Export PNG map previews
    st.subheader("🖼️ Export Raster Map Images")
    for name, raster_path in st.session_state.damage_rasters.items():
        try:
            with rasterio.open(raster_path) as src:
                damage_arr = src.read(1)
                masked = np.ma.masked_where(damage_arr == 0, damage_arr)
                fig, ax = plt.subplots(figsize=(6, 4))
                cmap = plt.cm.get_cmap("Reds")
                im = ax.imshow(masked, cmap=cmap)
                ax.set_title(f"Crop Damage – {name}")
                plt.colorbar(im, ax=ax, label="% Damage")

                # Save to PNG temp file
                img_tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                fig.savefig(img_tmp.name, bbox_inches="tight")
                plt.close(fig)

                with open(img_tmp.name, "rb") as img_file:
                    st.download_button(
                        label=f"🖼️ Download {name} PNG",
                        data=img_file,
                        file_name=f"{name}_damage.png",
                        mime="image/png"
                    )
        except Exception as e:
            st.warning(f"Image export failed for {name}: {e}")
