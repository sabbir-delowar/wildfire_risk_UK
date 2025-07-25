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

def load_viirs_fire_count(aoi, start_year=2020, end_year=2024):
    print("📦 Loading VIIRS fire data...")
    collection = ee.ImageCollection("FIRMS") \
        .filterDate(f"{start_year}-01-01", f"{end_year}-12-31") \
        .filterBounds(aoi)

    fire_count = collection.select("T21").count().clip(aoi).rename("fire_count")

    # Convert to uint16 for export compatibility
    return fire_count.toUint16()

def export_fire_layer(image, aoi, out_path, scale=375, crs="EPSG:4326"):
    print(f"⬇️ Downloading {os.path.basename(out_path)}...")
    geemap.download_ee_image(
        image=image,
        region=aoi.geometry(),
        filename=out_path,
        scale=scale,
        crs=crs,
        dtype="uint16",
        max_tile_size=8
    )

def run_fire_extraction(aoi_path, output_dir):
    initialise_gee()

    aoi = load_aoi(aoi_path)
    fire_image = load_viirs_fire_count(aoi)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "historic_fire_count_2020_2024.tif")
    export_fire_layer(fire_image, aoi, out_path)

    print("✅ Historic fire raster saved to:", out_path, "\n")
