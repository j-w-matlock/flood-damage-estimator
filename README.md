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

You may also provide a polygon layer (zipped Shapefile, GeoJSON, or KML) instead of a flood depth raster. Polygons are rasterized onto the crop raster grid, and cells inside the polygon are assumed to receive **0.5&nbsp;ft** (6&nbsp;inches) of flooding. The original file names of uploaded rasters and polygons become the labels in the output Excel workbook.


## Running Tests

Unit tests are written with **pytest**. After installing the requirements simply run:

```bash
pytest
```

## Damage Assumption

Flood losses are scaled relative to a depth of **6&nbsp;ft** of water. Any pixel with 6&nbsp;feet or more of inundation is treated as a total crop loss.
