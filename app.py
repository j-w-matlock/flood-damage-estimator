import streamlit as st
import os
import tempfile
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import rasterio
from utils.processing import process_flood_damage

st.set_page_config(layout="wide")
st.title("🌾 Agricultural Flood Damage Estimator")

# Initialize session state
for key in ["result_path", "summaries", "diagnostics", "crop_path", "depth_paths", "monte_carlo_results"]:
    if key not in st.session_state:
        st.session_state[key] = None

# 🔁 Reset
if st.button("🔁 Reset App"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# File inputs
crop_file = st.file_uploader("🌾 Upload USDA CropScape Raster", type=["tif", "img"])
depth_files = st.file_uploader("🌊 Upload Flood Depth Raster(s)", type="tif", accept_multiple_files=True)

period_years = st.number_input("📆 Period of Analysis (Years)", value=50, min_value=1)
samples = st.number_input("🔁 Sampling Iterations (for Direct Damage Averaging)", value=50, min_value=10)

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
    top_crops = [c for c, _ in counts.most_common(10) if c != 0]

    st.markdown("### 🌱 Define Crop Values and Growing Seasons")
    for code in top_crops:
        val = st.number_input(f"Crop {code} – $/Acre", value=5500, step=100, key=f"val_{code}")
        months = st.multiselect(f"Crop {code} – Growing Months (1–12)", options=list(range(1, 13)),
                                default=list(range(4, 10)), key=f"grow_{code}")
        if months:
            crop_inputs[code] = {"Value": val, "GrowingSeason": months}
        else:
            st.warning(f"⚠️ Crop {code} has no growing months.")

# Flood metadata
if depth_files:
    st.markdown("### ⚙️ Flood Raster Settings")
    depth_paths = []
    for i, f in enumerate(depth_files):
        temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".tif").name
        with open(temp_path, "wb") as out:
            out.write(f.read())
        depth_paths.append(temp_path)

        rp = st.number_input(f"Return Period – {f.name}", min_value=1, value=100, key=f"rp_{i}")
        mo = st.number_input(f"Flood Month – {f.name}", min_value=1, max_value=12, value=6, key=f"mo_{i}")
        flood_metadata[f.name] = {"return_period": rp, "flood_month": mo}

    st.session_state["depth_paths"] = depth_paths

# Run model
if st.button("🚀 Run Flood Damage Estimator"):
    if not crop_file or not depth_files:
        st.error("Upload both a CropScape raster and one or more depth rasters.")
    elif not crop_inputs:
        st.error("Set at least one crop value and growing season.")
    else:
        with st.spinner("Running base damage calculation..."):
            temp_dir = tempfile.mkdtemp()
            try:
                result_path, summaries, diagnostics, _ = process_flood_damage(
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
                st.session_state.monte_carlo_results = None
                st.success("✅ Direct damage and EAD results ready!")
            except Exception as e:
                st.error(f"❌ Error: {e}")

# Results tabs
if st.session_state.summaries:
    tab1, tab2 = st.tabs(["📊 Direct EAD Results", "📈 Monte Carlo (Optional)"])

    with tab1:
        st.download_button("📥 Download Excel Summary", open(st.session_state.result_path, "rb").read(),
                           file_name="ag_damage_summary.xlsx")

        if st.session_state.diagnostics:
            st.markdown("### 🧪 Diagnostics")
            st.dataframe(pd.DataFrame(st.session_state.diagnostics))

        for flood, df in st.session_state.summaries.items():
            st.subheader(f"{flood}")
            df["EAD"] = (df["DollarsLost"] / flood_metadata[flood + '.tif']["return_period"]).round(2)
            st.dataframe(df)

            if "CropCode" in df.columns and "EAD" in df.columns:
                fig, ax = plt.subplots()
                df.plot(kind="bar", x="CropCode", y="EAD", ax=ax)
                ax.set_title("Expected Annual Damage (EAD)")
                ax.set_ylabel("$/year")
                st.pyplot(fig)

    with tab2:
        st.markdown("### 🧮 Optional Monte Carlo Simulation")
        if st.button("▶️ Run Monte Carlo Simulation"):
            mc_results = {}
            for flood, df in st.session_state.summaries.items():
                mc_rows = []
                for _, row in df.iterrows():
                    mean = row["DirectDamage_Mean"]
                    std = row["DirectDamage_Std"]
                    sims = np.random.normal(mean, std, size=1000)
                    eads = sims * (1 / flood_metadata[flood + '.tif']["return_period"])
                    mc_rows.append({
                        "CropCode": row["CropCode"],
                        "EAD_Mean": round(np.mean(eads), 2),
                        "EAD_5th": round(np.percentile(eads, 5), 2),
                        "EAD_95th": round(np.percentile(eads, 95), 2)
                    })
                mc_results[flood] = pd.DataFrame(mc_rows)
            st.session_state.monte_carlo_results = mc_results
            st.success("✅ Monte Carlo simulation complete!")

        if st.session_state.monte_carlo_results:
            for flood, df in st.session_state.monte_carlo_results.items():
                st.subheader(f"{flood} – Monte Carlo EAD")
                st.dataframe(df)

                fig, ax = plt.subplots()
                df.plot(kind="bar", x="CropCode", y="EAD_Mean", ax=ax)
                ax.set_title("Monte Carlo EAD (Mean)")
                ax.set_ylabel("$/year")
                st.pyplot(fig)

            # Export MC results
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
                with pd.ExcelWriter(tmp.name, engine="openpyxl") as writer:
                    for flood, df in st.session_state.monte_carlo_results.items():
                        df.to_excel(writer, sheet_name=flood, index=False)
                st.download_button("📤 Download Monte Carlo Results", open(tmp.name, "rb").read(),
                                   file_name="monte_carlo_EAD.xlsx")
