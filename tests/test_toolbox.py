import importlib.util
from importlib.machinery import SourceFileLoader
import sys
from types import SimpleNamespace
from pathlib import Path

import pandas as pd
import numpy as np
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.test_processing import create_raster


def load_toolbox(monkeypatch):
    fake_arcpy = SimpleNamespace(Parameter=lambda **kwargs: None)
    monkeypatch.setitem(sys.modules, "arcpy", fake_arcpy)
    path = Path(__file__).resolve().parents[1] / "flood_damage_toolbox.pyt"
    loader = SourceFileLoader("flood_damage_toolbox", str(path))
    spec = importlib.util.spec_from_loader("flood_damage_toolbox", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def test_execute_basic(tmp_path, monkeypatch):
    tb = load_toolbox(monkeypatch)

    crop_arr = np.array([[1, 2], [2, 1]], dtype=np.uint16)
    crop_path = tmp_path / "crop.tif"
    create_raster(crop_path, crop_arr, "EPSG:4326", from_origin(0, 2, 1, 1))

    depth_arr = np.full((2, 2), 6.0, dtype=float)
    depth_path = tmp_path / "depthA.tif"
    create_raster(depth_path, depth_arr, "EPSG:4326", from_origin(0, 2, 1, 1))

    crop_table = tmp_path / "crops.csv"
    pd.DataFrame({"CropCode": [1, 2], "Value": [10, 20], "GrowingSeason": [6, 6]}).to_csv(crop_table, index=False)

    out_dir = tmp_path / "out"

    params = [
        SimpleNamespace(value=None, valueAsText=str(crop_path)),
        SimpleNamespace(value=None, valueAsText=str(depth_path)),
        SimpleNamespace(value=None, valueAsText="10"),
        SimpleNamespace(value=None, valueAsText="6"),
        SimpleNamespace(value=None, valueAsText=str(crop_table)),
        SimpleNamespace(value=None, valueAsText=str(out_dir)),
        SimpleNamespace(value=True),
        SimpleNamespace(value=5),
        SimpleNamespace(value=20.0),
        SimpleNamespace(value=0.2),
    ]

    messages = SimpleNamespace(addMessage=lambda x: None)

    tb.FloodDamageTool().execute(params, messages)

    assert (out_dir / "ag_damage_summary.xlsx").exists()

