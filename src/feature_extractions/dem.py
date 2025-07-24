import ee
import geemap
import os
from src.feature_extractions.sentinel import load_aoi 

def initialise_gee():
    try:
        ee.Initialize()
    except:
        ee.Authenticate()
        ee.Initialize()

def load_dem_features(aoi, scale=30):
    """Returns elevation, slope, and aspect images clipped to AOI."""

    dem = (
        ee.ImageCollection("COPERNICUS/DEM/GLO30")
        .mosaic()
        .select("DEM")
        .rename("elevation")
    )

    elevation = dem.clip(aoi)

    return elevation

def export_to_local(image, region, filepath, scale=30, crs="EPSG:4326"):
    print(f"⬇️ Downloading {os.path.basename(filepath)}...")
    geemap.download_ee_image(
        image=image,
        region=region.geometry(),
        filename=filepath,
        scale=scale,
        crs=crs,
        max_tile_size=8
    )


def run_dem_extraction(aoi_path, output_dir): 
    initialise_gee()
    aoi = load_aoi(aoi_path)

    elevation = load_dem_features(aoi)
    
    export_to_local(elevation, aoi, os.path.join(output_dir, "elevation.tif"))
