import os
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
import geopandas as gpd


def make_fire_risk_labels(aoi_path,  label_output_path, resolution=1000, fire_raster_path="data/processed/firms_fire_count_2020_2024.tif"):
    # Load AOI and get bounds
    gdf = gpd.read_file(aoi_path)
    gdf = gdf.to_crs("EPSG:4326")
    minx, miny, maxx, maxy = gdf.total_bounds

    width = int((maxx - minx) * 111320 / resolution)  # 111.32 km per degree approx
    height = int((maxy - miny) * 110574 / resolution)  # slightly different for lat

    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    # Create base label raster filled with 0s (no fire)
    label_arr = np.zeros((height, width), dtype=np.uint8)

    # Load fire raster and reproject/resample to match
    with rasterio.open(fire_raster_path) as src:
        fire_data = src.read(1)
        fire_arr = np.full((height, width), 0, dtype=np.float32)

        rasterio.warp.reproject(
            source=fire_data,
            destination=fire_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs="EPSG:4326",
            resampling=rasterio.warp.Resampling.nearest
        )

    # Assign fire labels
    label_arr = np.where(fire_arr > 0, 1, 0).astype(np.uint8)

    # Save final label raster
    out_meta = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': 1,
        'dtype': 'uint8',
        'crs': 'EPSG:4326',
        'transform': transform,
        'nodata': 255
    }

    os.makedirs(os.path.dirname(label_output_path), exist_ok=True)

    with rasterio.open(label_output_path, 'w', **out_meta) as dst:
        dst.write(label_arr, 1)

    print(f"✅ Fire risk labels saved to: {label_output_path}")
