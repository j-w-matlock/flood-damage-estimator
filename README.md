# Flood Damage Estimator

This Streamlit app estimates agricultural flood damages from cropland and flood depth rasters.

Default crop values and typical growing seasons are built in, allowing the
model to run with only a CropScape raster and one or more flood depth grids.
The interface now reports **all** crop codes present in the uploaded raster and
displays bar chart visualizations of damages for quick comparison between crop
types.

## Usage

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the app:
   ```bash
   streamlit run app.py
   ```

The app automatically populates crop values and growing seasons using USDA
defaults. After uploading your data you can expand the **Crop Values and
Growing Seasons** section to review or adjust those assumptions.

## Providing Raster Data

You may either upload `.tif` files directly in the sidebar or provide paths to existing files on disk:

- **Crop Raster Path (optional)** – Enter the path to a local cropland raster.
- **Flood Raster Paths (optional, one per line)** – Enter one or more flood depth raster paths.

If paths are supplied, the application reads the files directly with `rasterio`, bypassing Streamlit's upload size limit.

## Uniform Depth Option

Instead of supplying a polygon to define flooded areas, you can enable the *Uniform Flood Depth* checkbox and enter a depth value in feet. The app will duplicate the crop raster and assign the specified depth to every pixel.

## Damage Map Visualization

Output maps color-code damaged areas by crop type and include a legend with crop names for easy interpretation.

## Running Tests

Unit tests are written with **pytest**. After installing the requirements simply run:

```bash
pytest
```

## Damage Assumption

Flood losses are scaled relative to a depth of **6&nbsp;ft** of water. Any pixel with 6&nbsp;feet or more of inundation is treated as a total crop loss.

## ArcGIS Pro Toolbox

A Python toolbox named `flood_damage_toolbox.pyt` is provided for running the model within ArcGIS Pro.

1. In ArcGIS Pro choose **Insert** → **Toolbox** and browse to `flood_damage_toolbox.pyt`.
2. Open the **Flood Damage Estimator** tool.
3. Provide a crop raster, one or more depth rasters, a CSV with crop codes, values and growing seasons, and an output folder.
4. Enter a return period and flood month for **each** depth raster (the lists must match in length).
5. Optionally enable Monte Carlo simulation and adjust the uncertainty parameters, including a setting to randomize the flood month so events can occur in any part of the year.

The tool writes an Excel summary in the chosen output folder, mirroring the results of the Streamlit application.

If the `rasterio` package is not available, the toolbox automatically falls back
to using `arcpy` for all raster operations. This makes it compatible with a
standard ArcGIS Pro Python installation.
