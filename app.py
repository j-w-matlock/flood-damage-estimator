import streamlit as st
import os
import tempfile
import pandas as pd
import rasterio
from utils.processing import (
    process_flood_damage,
    run_monte_carlo,
    polygon_mask_to_depth_array,
    constant_depth_array,
    drawn_features_to_depth_array,
)
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from openpyxl import load_workbook
from openpyxl.drawing.image import Image as XLImage

st.set_page_config(layout="wide")
st.title("🌾 Agricultural Flood Damage Estimator")

# Session state init
for key in [
    "result_path",
    "summaries",
    "diagnostics",
    "crop_path",
    "depth_inputs",
    "damage_rasters",
    "label_map",
    "label_metadata",
    "temp_files",
    "temp_dir",
]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "temp_files" else None

    files = st.session_state.get("temp_files") or []
    for path in files:
        if path and os.path.exists(path):
            os.remove(path)
    st.session_state["temp_files"] = []


def cleanup_temp_dir():
    tmp = st.session_state.get("temp_dir")
    if tmp:
        tmp.cleanup()
    st.session_state["temp_dir"] = None


def save_upload(uploaded, suffix):
    path = tempfile.NamedTemporaryFile(delete=False, suffix=suffix).name
    with open(path, "wb") as out:
        out.write(uploaded.read())
    st.session_state.temp_files.append(path)
    return path


# Reset
if st.sidebar.button("🔁 Reset App"):
    cleanup_temp_dir()
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# Sidebar Inputs
st.sidebar.header("🛠️ Settings")
mode = st.sidebar.radio(
    "Select Analysis Mode:",
    ["Direct Damages", "Monte Carlo Simulation"],
    help="Choose whether to run a straightforward flood loss calculation (Direct Damages) or include uncertainty using random simulations (Monte Carlo Simulation)",
)
crop_file = st.sidebar.file_uploader(
    "🌾 USDA CropScape Raster",
    type=["tif", "img"],
    help="Upload a CropScape raster that defines crop type per pixel",
)
depth_files = st.sidebar.file_uploader(
    "🌊 Flood Depth Grids",
    type=["tif"],
    accept_multiple_files=True,
    help="Upload one or more flood depth raster files (in feet)",
)
use_uniform_depth = st.sidebar.checkbox(
    "🏞️ Uniform Flood Depth",
    value=False,
    help="Duplicate the crop raster and fill with a user specified constant depth",
)
uniform_depth_ft = st.sidebar.number_input(
    "Uniform Depth (ft)",
    min_value=0.1,
    value=0.5,
    step=0.1,
    disabled=not use_uniform_depth,
    help="Flood depth applied everywhere when using the uniform option",
)
use_manual_mask = st.sidebar.checkbox(
    "🎨 Manual Depth Painting",
    value=False,
    help="Draw polygons on a map with selectable depth values",
)
period_years = st.sidebar.number_input(
    "📆 Analysis Period (Years)",
    min_value=1,
    value=50,
    help="Used for context in planning studies; not required for EAD computation",
)
samples = st.sidebar.number_input(
    "🎲 Monte Carlo Iterations",
    min_value=10,
    value=100,
    help="Number of random simulations per crop type to estimate uncertainty",
)
depth_sd = st.sidebar.number_input(
    "± Depth Uncertainty (ft)",
    value=0.1,
    help="Assumed standard deviation of flood depth error (used only in Monte Carlo)",
)
value_sd = st.sidebar.number_input(
    "± Crop Value Uncertainty (%)",
    value=10,
    help="Assumed variability in per-acre crop values (used only in Monte Carlo)",
)

crop_inputs, label_to_filename, label_to_metadata = {}, {}, {}

# Process cropland raster
if crop_file:
    crop_path = save_upload(crop_file, ".tif")
    st.session_state.crop_path = crop_path

    with rasterio.open(crop_path) as src:
        arr = src.read(1)
    counts = Counter(arr.flatten())
    codes = [c for c, _ in counts.most_common() if c != 0]

    st.markdown("### 🌱 Crop Values and Growing Seasons")
    for code in codes:
        val = st.number_input(
            f"Crop {code} – $/Acre",
            value=5500,
            step=100,
            key=f"val_{code}",
            help="Enter average crop value per acre for this code",
        )
        season = st.multiselect(
            f"Crop {code} – Growing Months",
            list(range(1, 13)),
            default=list(range(4, 10)),
            key=f"season_{code}",
            help="Choose the active growing months when this crop is vulnerable to flooding",
        )
        crop_inputs[code] = {"Value": val, "GrowingSeason": season}

# Process flood rasters or uniform depth or manual mask
if depth_files or use_uniform_depth or use_manual_mask:
    st.markdown("### ⚙️ Flood Raster Settings")
    depth_inputs = []

    if depth_files:
        for i, f in enumerate(depth_files):
            path = save_upload(f, ".tif")
            label = os.path.splitext(os.path.basename(f.name))[0]
            depth_inputs.append((label, path))
            rp = st.number_input(
                f"Return Period: {f.name}",
                min_value=1,
                value=100,
                key=f"rp_{i}",
                help="How often this flood event is expected to occur (e.g., 100 for 1-in-100 year flood)",
            )
            mo = st.number_input(
                f"Flood Month: {f.name}",
                min_value=1,
                max_value=12,
                value=6,
                key=f"mo_{i}",
                help="Month of flood to compare against crop growing season",
            )
            label_to_filename[label] = f.name
            label_to_metadata[label] = {"return_period": rp, "flood_month": mo}

    if use_uniform_depth and crop_file:
        rp = st.number_input(
            "Return Period: Uniform Depth",
            min_value=1,
            value=100,
            key="rp_uniform",
            help="How often this flood event is expected to occur",
        )
        mo = st.number_input(
            "Flood Month: Uniform Depth",
            min_value=1,
            max_value=12,
            value=6,
            key="mo_uniform",
            help="Month of flood to compare against crop growing season",
        )

        depth_arr = constant_depth_array(st.session_state.crop_path, uniform_depth_ft)
        label = f"uniform_{uniform_depth_ft}ft"
        depth_inputs.append((label, depth_arr))
        label_to_filename[label] = label
        label_to_metadata[label] = {"return_period": rp, "flood_month": mo}

    if use_manual_mask and crop_file:
        st.markdown("### 🎨 Manual Depth Painting")
        if "drawn_features" not in st.session_state:
            st.session_state.drawn_features = []
        if st.button("Reset Drawings"):
            st.session_state.drawn_features = []

        brush_depth = st.select_slider(
            "Current Brush Depth",
            options=[i / 2 for i in range(1, 13)],
            format_func=lambda x: f"{x} ft",
        )

        import folium
        from folium.plugins import Draw
        from streamlit_folium import st_folium

        m = folium.Map(tiles="cartodbpositron")
        Draw(export=False).add_to(m)
        map_data = st_folium(
            m,
            key="draw_map",
            height=400,
            width=700,
            returned_objects=["last_active_drawing"],
        )

        if map_data.get("last_active_drawing"):
            feat = map_data["last_active_drawing"]
            feat.setdefault("properties", {})["depth"] = float(brush_depth)
            st.session_state.drawn_features.append(feat)

        if st.session_state.drawn_features:
            st.info(f"Drawn polygons: {len(st.session_state.drawn_features)}")
            rp = st.number_input(
                "Return Period: Manual Mask",
                min_value=1,
                value=100,
                key="rp_manual",
                help="How often this flood event is expected to occur",
            )
            mo = st.number_input(
                "Flood Month: Manual Mask",
                min_value=1,
                max_value=12,
                value=6,
                key="mo_manual",
                help="Month of flood to compare against crop growing season",
            )

            depth_arr = drawn_features_to_depth_array(
                st.session_state.drawn_features, st.session_state.crop_path
            )
            label = "manual_mask"
            depth_inputs.append((label, depth_arr))
            label_to_filename[label] = label
            label_to_metadata[label] = {"return_period": rp, "flood_month": mo}

    st.session_state.depth_inputs = depth_inputs
    st.session_state.label_map = label_to_filename
    st.session_state.label_metadata = label_to_metadata

# Direct Damages Mode
if mode == "Direct Damages":
    if st.button("🚀 Run Flood Damage Estimator"):
        if not (crop_file and (depth_files or use_uniform_depth or use_manual_mask)):
            st.error(
                "❌ Please upload a cropland raster and at least one flood source."
            )
        else:
            cleanup_temp_dir()
            st.session_state.temp_dir = tempfile.TemporaryDirectory()
            with st.spinner("🔄 Processing flood damages..."):
                try:
                    result_path, summaries, diagnostics, damage_rasters = (
                        process_flood_damage(
                            st.session_state.crop_path,
                            st.session_state.depth_inputs,
                            st.session_state.temp_dir.name,
                            period_years,
                            crop_inputs,
                            st.session_state.label_metadata,
                        )
                    )
                    st.session_state.result_path = result_path
                    st.session_state.summaries = summaries
                    st.session_state.diagnostics = diagnostics
                    st.session_state.damage_rasters = damage_rasters
                    st.success("✅ Direct damage analysis complete.")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

    if st.session_state.result_path and "damage_rasters" in st.session_state:
        st.markdown("---")
        st.header("📈 Results & Visualizations")

        image_paths = {}

        for label, df in st.session_state.summaries.items():
            st.subheader(f"📋 Summary for {label}")
            with st.expander("ℹ️ Column Definitions"):
                st.markdown(
                    """
                - **CropCode**: CropScape code for crop type.
                - **FloodedAcres**: Area affected (1 pixel ≈ 0.222 acres).
                - **ValuePerAcre**: Input value per acre for the crop.
                - **DollarsLost**: Total crop damage.
                - **EAD**: Expected Annual Damage = DollarsLost ÷ ReturnPeriod.
                """
                )
            st.dataframe(df)
            chart_data = (
                df.sort_values("DollarsLost", ascending=False)
                .set_index("CropCode")["DollarsLost"]
            )
            fig_bar, ax_bar = plt.subplots()
            ax_bar.bar(chart_data.index.astype(str), chart_data.values)
            ax_bar.set_xlabel("Crop Code")
            ax_bar.set_ylabel("Dollars Lost")
            st.pyplot(fig_bar)
            bar_path = os.path.join(
                st.session_state.temp_dir.name, f"bar_{label}.png"
            )
            fig_bar.savefig(bar_path)
            plt.close(fig_bar)
            image_paths.setdefault(label, {})["bar"] = bar_path

        if st.session_state.diagnostics:
            st.subheader("🛠️ Diagnostics")
            st.dataframe(pd.DataFrame(st.session_state.diagnostics))

        for label, arrs in st.session_state.damage_rasters.items():
            st.subheader(f"🗺️ Damage Raster – {label}")
            crop_arr = arrs.get("crop") if isinstance(arrs, dict) else arrs
            masked = np.ma.masked_where(crop_arr == 0, crop_arr)
            fig, ax = plt.subplots()
            cax = ax.imshow(masked, cmap="tab20")
            fig.colorbar(cax, ax=ax, label="Crop Code")
            ax.set_title(f"Damaged Crops: {label}")
            st.pyplot(fig)
            raster_path = os.path.join(
                st.session_state.temp_dir.name, f"raster_{label}.png"
            )
            fig.savefig(raster_path)
            plt.close(fig)
            image_paths.setdefault(label, {})["raster"] = raster_path

        # Insert images into Excel
        if image_paths:
            wb = load_workbook(st.session_state.result_path)
            for label, imgs in image_paths.items():
                if label in wb.sheetnames:
                    ws = wb[label]
                    if "bar" in imgs:
                        ws.add_image(XLImage(imgs["bar"]), "H2")
                    if "raster" in imgs:
                        ws.add_image(XLImage(imgs["raster"]), "H20")
            wb.save(st.session_state.result_path)

        with open(st.session_state.result_path, "rb") as file:
            st.download_button(
                label="📥 Download Results Excel File",
                data=file,
                file_name=os.path.basename(st.session_state.result_path),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

# Monte Carlo Mode
elif mode == "Monte Carlo Simulation":
    if st.button("🧪 Run Monte Carlo Simulation"):
        if not (crop_file and (depth_files or use_uniform_depth or use_manual_mask)):
            st.error(
                "❌ Please upload a cropland raster and at least one flood source."
            )
        else:
            cleanup_temp_dir()
            st.session_state.temp_dir = tempfile.TemporaryDirectory()
            with st.spinner("🔬 Running Monte Carlo..."):
                try:
                    result_path, summaries, diagnostics, damage_rasters = (
                        process_flood_damage(
                            st.session_state.crop_path,
                            st.session_state.depth_inputs,
                            st.session_state.temp_dir.name,
                            period_years,
                            crop_inputs,
                            st.session_state.label_metadata,
                        )
                    )
                    mc_results = run_monte_carlo(
                        summaries,
                        st.session_state.label_metadata,
                        samples,
                        value_sd,
                        depth_sd,
                    )
                    st.session_state.result_path = result_path

                    st.markdown("---")
                    st.header("📉 Monte Carlo EAD Results")

                    for label, df in mc_results.items():
                        st.subheader(f"🧪 MC Summary for {label}")
                        with st.expander("ℹ️ Column Definitions"):
                            st.markdown(
                                """
                            - **CropCode**: CropScape code for crop type.
                            - **EAD_MC_Mean**: Mean simulated EAD from Monte Carlo.
                            - **EAD_MC_5th / 95th**: Uncertainty bounds (percentiles).
                            - **Original_EAD**: Deterministic EAD value for comparison.
                            """
                            )
                        st.dataframe(df)
                        chart_data = (
                            df.sort_values("EAD_MC_Mean", ascending=False)
                            .set_index("CropCode")["EAD_MC_Mean"]
                        )
                        st.bar_chart(chart_data)

                    with pd.ExcelWriter(
                        result_path, mode="a", engine="openpyxl"
                    ) as writer:
                        for label, df in mc_results.items():
                            df.to_excel(writer, sheet_name=f"MC_{label}", index=False)

                    with open(result_path, "rb") as file:
                        st.download_button(
                            label="📥 Download Monte Carlo Excel File",
                            data=file,
                            file_name=os.path.basename(result_path),
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )

                    st.success("✅ Monte Carlo analysis complete.")

                except Exception as e:
                    st.error(f"⚠️ Monte Carlo error: {e}")

