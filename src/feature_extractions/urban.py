import os
import numpy as np
import geopandas as gpd
import rasterio
from shapely.geometry import Point
from rasterio.features import dataset_features
from rasterio.mask import mask
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt

def extract_urban_mask(corine_path, urban_classes=[111, 112, 121, 122, 123, 124]):
    print("📦 Extracting urban mask from CORINE...")
    with rasterio.open(corine_path) as src:
        corine = src.read(1)
        profile = src.profile

    # Create binary mask: 1 if pixel is urban class, else 0
    urban_mask = np.isin(corine, urban_classes).astype(np.uint8)
    return urban_mask, profile

def compute_distance_from_mask(mask_array, profile):
    print("📦 Computing distance transform...")
    # Invert mask: distance_transform_edt computes distance to non-zero
    inverted = (mask_array == 0).astype(np.uint8)
    distances = distance_transform_edt(inverted) * profile["transform"][0]  # scale to meters
    return distances

def clip_to_aoi(raster, profile, aoi_path):
    print("📦 Clipping distance raster to original AOI...")
    aoi = gpd.read_file(aoi_path).to_crs(profile["crs"])
    with rasterio.MemoryFile() as memfile:
        with memfile.open(**profile) as tmp:
            tmp.write(raster, 1)
            out_image, out_transform = mask(tmp, aoi.geometry, crop=True)
            out_meta = tmp.meta.copy()
            out_meta.update({
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })
    return out_image[0], out_meta

def run_distance_to_urban(original_aoi_path, output_dir,
                          corine_path="data/processed/corine-2018.tif",
                          buffered_aoi_path="data/temp/aoi_buffered_1000m.geojson"
                         ):
    urban_mask, profile = extract_urban_mask(corine_path)
    distances = compute_distance_from_mask(urban_mask, profile)
    clipped_distances, clipped_profile = clip_to_aoi(distances, profile, original_aoi_path)

    output_path = os.path.join(output_dir, "distance-to-urban.tif")
    with rasterio.open(output_path, "w", **clipped_profile) as dst:
        dst.write(clipped_distances.astype(np.float32), 1)
    
    print(f"✅ Distance to urban areas saved to: {output_path}\n")
