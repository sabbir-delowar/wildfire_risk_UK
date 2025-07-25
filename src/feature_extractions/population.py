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

def load_population_density(aoi):
    # 2020 is the latest available
    image = ee.Image("WorldPop/GP/100m/pop/GBR_2020").select("population")
    return image.clip(aoi.geometry()).rename("pop_density")

def export_population_raster(image, region, path, scale=100, crs="EPSG:4326"):
    print(f"⬇️ Downloading {os.path.basename(path)}...")
    geemap.download_ee_image(
        image=image,
        region=region.geometry(),
        filename=path,
        scale=scale,
        crs=crs
    )

def run_population_extraction(aoi_path, output_dir):
    initialise_gee()
    aoi = load_aoi(aoi_path)
    pop_image = load_population_density(aoi)

    export_population_raster(pop_image, aoi, os.path.join(output_dir, "population-2020.tif"))
