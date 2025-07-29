diff --git a//dev/null b/README.md
index 0000000000000000000000000000000000000000..c4fb1a8b6ead0e35fbf74a717db1cab404d8b5a2 100644
--- a//dev/null
+++ b/README.md
@@ -0,0 +1,23 @@
+# Flood Damage Estimator
+
+This Streamlit app estimates agricultural flood damages from cropland and flood depth rasters.
+
+## Usage
+
+1. Install requirements:
+   ```bash
+   pip install -r requirements.txt
+   ```
+2. Run the app:
+   ```bash
+   streamlit run app.py
+   ```
+
+## Providing Raster Data
+
+You may either upload `.tif` files directly in the sidebar or provide paths to existing files on disk:
+
+- **Crop Raster Path (optional)** – Enter the path to a local cropland raster.
+- **Flood Raster Paths (optional, one per line)** – Enter one or more flood depth raster paths.
+
+If paths are supplied, the application reads the files directly with `rasterio`, bypassing Streamlit's upload size limit.
