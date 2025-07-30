# Flood Damage Estimator

This Streamlit app estimates agricultural flood damages from cropland and flood depth rasters.

## Usage

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the app:
   ```bash
   streamlit run app.py
   ```

## Providing Raster Data

You may either upload `.tif` files directly in the sidebar or provide paths to existing files on disk:

- **Crop Raster Path (optional)** – Enter the path to a local cropland raster.
- **Flood Raster Paths (optional, one per line)** – Enter one or more flood depth raster paths.

If paths are supplied, the application reads the files directly with `rasterio`, bypassing Streamlit's upload size limit.

## Polygon Uploads

In place of a flood depth raster you may also upload a polygon layer (zipped Shapefile, GeoJSON, or KML). The polygon is rasterized to the crop raster's grid and assigned a uniform depth of **0.5&nbsp;ft** (6&nbsp;inches). The resulting depth array is processed just like any other flood raster when computing damages.

## Damage Assumption

Flood losses are scaled relative to a depth of **6&nbsp;ft** of water. Any pixel with 6&nbsp;feet or more of inundation is treated as a total crop loss.
