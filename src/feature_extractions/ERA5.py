import ee
import geemap
import os
import glob
import rasterio
import numpy as np
from src.feature_extractions.sentinel import load_aoi

def initialise_gee():
    try:
        ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize()

def kelvin_to_celsius(image):
    return image.subtract(273.15)

def compute_relative_humidity(temp_k, dew_k):
    """Approximate RH from temp and dew point in Kelvin"""
    return dew_k.subtract(273.15).subtract(temp_k.subtract(273.15)) \
        .multiply(-17.625).divide(dew_k.subtract(273.15).subtract(243.04)) \
        .exp().multiply(100).rename("rh")

def extract_monthly_images(aoi, year_range=(2020, 2024), months=[6, 7, 8]):
    era = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY").filterBounds(aoi)
    results = []

    for year in range(year_range[0], year_range[1] + 1):
        for month in months:
            filtered = era.filter(ee.Filter.calendarRange(year, year, 'year')) \
                          .filter(ee.Filter.calendarRange(month, month, 'month'))

            label = f"{year}_{month:02d}"

            temp = kelvin_to_celsius(filtered.select("temperature_2m").mean()).clip(aoi).rename(f"temp_{label}")
            precip = filtered.select("total_precipitation").sum().multiply(1000).clip(aoi).rename(f"precip_{label}")
            dew = filtered.select("dewpoint_temperature_2m").mean().clip(aoi)
            rh = compute_relative_humidity(filtered.select("temperature_2m").mean(), dew).clip(aoi).rename(f"rh_{label}")

            results.append((temp, precip, rh, label))
    return results

def export_climate_image(image, aoi, out_path, scale=5000, crs="EPSG:4326"):
    print(f"⬇️ Downloading {os.path.basename(out_path)}...")
    geemap.download_ee_image(
        image=image,
        region=aoi.geometry(),
        filename=out_path,
        scale=scale,
        crs=crs,
        max_tile_size=8
    )

def run_monthly_climate_extraction(aoi_path, output_dir, temp_dir="data/tempERA5"):
    print("📦 Starting monthly climate feature extraction...")
    initialise_gee()
    aoi = load_aoi(aoi_path)
    os.makedirs(output_dir, exist_ok=True)

    monthly_data = extract_monthly_images(aoi)

    for temp, precip, rh, label in monthly_data:
        export_climate_image(temp, aoi, os.path.join(temp_dir, f"temp_{label}.tif"))
        export_climate_image(precip, aoi, os.path.join(temp_dir, f"precip_{label}.tif"))
        export_climate_image(rh, aoi, os.path.join(temp_dir, f"rh_{label}.tif"))

    print("✅ Monthly climate extraction done (temp, precip, humidity).\n")
    
    


def create_mean_composites(output_dir, input_dir="data/temp/ERA5"):
    print("📦 Creating mean composites from ERA5 monthly rasters...")
    os.makedirs(output_dir, exist_ok=True)

    variables = ["temp", "precip", "rh"]

    for var in variables:
        files = sorted(glob.glob(os.path.join(input_dir, f"{var}_*.tif")))
        if not files:
            print(f"⚠️ No files found for {var}")
            continue

        print(f"📂 {var.upper()}: {len(files)} files found. Computing mean...")

        # Read first file to get metadata
        with rasterio.open(files[0]) as src0:
            meta = src0.meta.copy()
            data_stack = np.zeros((len(files), src0.height, src0.width), dtype=np.float32)

        # Read all files into the stack
        for i, f in enumerate(files):
            with rasterio.open(f) as src:
                data = src.read(1).astype(np.float32)
                data[data == src.nodata] = np.nan
                data_stack[i, :, :] = data

        mean_data = np.nanmean(data_stack, axis=0).astype(np.float32)

        out_path = os.path.join(output_dir, f"{var}_mean_2020_2024.tif")
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(mean_data, 1)

        print(f"✅ Saved mean composite: {out_path}")
