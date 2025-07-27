import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

def stack_and_align_rasters(output_dir):
    """
    Stacks and aligns rasters to a reference raster. Saves the result as a NumPy array.

    Args:
        raster_paths (list of str): Paths to input rasters.
        reference_path (str): Path to the reference raster (e.g., label raster).
        output_path (str): Where to save the stacked NumPy array.

    Returns:
        shape (tuple): Shape of the stacked array.
        feature_names (list): Names of the raster layers stacked.
    """
    output_path = os.path.join(output_dir, "features_stack.npy")
    reference = "data/labels/fire_label.tif"
    raster_paths = [
        "data/processed/NDVI_mean_2020_2024.tif",
        "data/processed/NDWI_mean_2020_2024.tif",
        "data/processed/temperature_2m_merged.tif",
        "data/processed/elevation_merged.tif",
        "data/processed/corine_2018_merged.tif",
        "data/processed/total_precipitation_sum_merged.tif",
        "data/processed/dewpoint_temperature_2m_merged.tif",
        "data/processed/distance-to-roads.tif",
        "data/processed/distance-to-urban.tif",
        "data/processed/population_2020_merged.tif"
    ]
    
    with rasterio.open(reference) as ref:
        ref_array = ref.read(1)
        ref_meta = ref.meta
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_shape = ref_array.shape

    stacked = []

    for path in raster_paths:
        try:
            with rasterio.open(path) as src:
                
                arr = src.read(1).astype(np.float32)

                # Replace invalid values with nan
                if src.nodata is not None:
                    arr = np.where(arr == src.nodata, np.nan, arr)
                arr = np.where(np.isinf(arr), np.nan, arr)
                arr = np.where(arr > 1e5, np.nan, arr)  # optional: filter unreasonably large values

                # Resample if needed
                if src.shape != ref_shape or src.transform != ref_transform or src.crs != ref_crs:
                    resampled = np.empty(ref_shape, dtype=np.float32)
                    reproject(
                        source=arr,
                        destination=resampled,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=ref_transform,
                        dst_crs=ref_crs,
                        resampling=Resampling.bilinear
                    )
                    arr = resampled


                stacked.append(arr)
        except Exception as e:
            print(f"⚠️ Skipping {path}: {e}")

    stacked_array = np.stack(stacked, axis=-1)

    # Replace inf and extremely large values
    stacked_array[~np.isfinite(stacked_array)] = np.nan
    stacked_array[stacked_array > 1e5] = np.nan
    np.save(output_path, stacked_array)

    print(f"✅ Saved stacked raster array: {output_path}")
    return stacked_array.shape, [os.path.splitext(os.path.basename(p))[0] for p in raster_paths]
