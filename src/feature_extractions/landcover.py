import ee
import geemap
import os
from src.feature_extractions.sentinel import load_aoi

def initialise_gee():
    try:
        ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize()

def load_corine_2018_rasterised(aoi):
    corine_2018 = ee.Image("COPERNICUS/CORINE/V20/100m/2018").select("landcover")
    return corine_2018.clip(aoi.geometry()).rename("corine_2018")

def export_corine_raster(image, region, path, scale=100, crs="EPSG:4326"):
    print(f"⬇️ Downloading {os.path.basename(path)}...")
    geemap.download_ee_image(
        image=image,
        region=region.geometry(),
        filename=path,
        scale=scale,
        crs=crs
    )

def run_landcover_extraction(aoi_path, output_dir):
    initialise_gee()
    aoi = load_aoi(aoi_path)
    lc_raster = load_corine_2018_rasterised(aoi)
    export_corine_raster(lc_raster, aoi, os.path.join(output_dir, "corine-2018.tif"))
