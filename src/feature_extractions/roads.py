import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.mask import mask
from shapely.geometry import box

def buffer_aoi(aoi_gdf, buffer_m=1000, temp_dir="data/temp"):
    print("📦 Buffering AOI ...")
    os.makedirs(temp_dir, exist_ok=True)
    buffer_path = os.path.join(temp_dir, f"aoi_buffered_{buffer_m}m.geojson")

    if os.path.exists(buffer_path):
        print(f"✅ Buffered AOI exists: {buffer_path}")
        return gpd.read_file(buffer_path).to_crs("EPSG:3857")

    aoi_proj = aoi_gdf.to_crs(epsg=3857)
    buffered = aoi_proj.buffer(buffer_m)
    buffered_gdf = gpd.GeoDataFrame(geometry=buffered, crs="EPSG:3857")
    buffered_gdf.to_file(buffer_path, driver="GeoJSON")
    print(f"✅ Buffered AOI saved: {buffer_path}")
    return buffered_gdf


def load_roads_clipped(roads_path, buffered_aoi, temp_dir="data/temp"):
    print("📦 Loading and clipping roads shapefile, this may take a while...")
    clipped_path = os.path.join(temp_dir, "roads_clipped.geojson")

    if os.path.exists(clipped_path):
        print(f"✅ Clipped roads exists: {clipped_path}")
        return gpd.read_file(clipped_path).to_crs("EPSG:3857")

    roads = gpd.read_file(roads_path).to_crs("EPSG:3857")
    clipped_roads = gpd.clip(roads, buffered_aoi)
    clipped_roads.to_file(clipped_path, driver="GeoJSON")
    print(f"✅ Clipped roads saved: {clipped_path}")
    return clipped_roads


def compute_distance_raster(buffered_aoi, roads_gdf, temp_dir="data/temp", pixel_size=10):
    print("📦 Computing rasterised distance map...")
    out_raster = os.path.join(temp_dir, "tmp_distance.tif")
    if os.path.exists(out_raster):
        print(f"✅ Distance raster exists: {out_raster}")
        return out_raster

    bounds = buffered_aoi.total_bounds
    width = int((bounds[2] - bounds[0]) / pixel_size)
    height = int((bounds[3] - bounds[1]) / pixel_size)

    transform = rasterio.transform.from_origin(bounds[0], bounds[3], pixel_size, pixel_size)

    # Rasterise roads as binary mask
    road_shapes = ((geom, 1) for geom in roads_gdf.geometry if geom.is_valid)
    road_raster = rasterize(
        road_shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype='uint8'
    )

    # Compute distance using scipy (optional: add fallback if not installed)
    from scipy.ndimage import distance_transform_edt
    distance = distance_transform_edt(road_raster == 0) * pixel_size  # in metres

    with rasterio.open(
        out_raster, "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:3857",
        transform=transform
    ) as dst:
        dst.write(distance.astype("float32"), 1)

    print(f"✅ Distance raster saved: {out_raster}")
    return out_raster


def clip_raster_to_aoi(raster_path, aoi_gdf, out_path):
    print("📦 Clipping raster to original AOI....")
    aoi = aoi_gdf.to_crs("EPSG:3857")

    with rasterio.open(raster_path) as src:
        out_image, out_transform = mask(
            src, 
            aoi.geometry, 
            crop=True,            # tightly crop
            all_touched=False,    # only include fully inside pixels
            nodata=np.nan         
        )
        out_meta = src.meta.copy()
        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "nodata": np.nan,
        })

    with rasterio.open(out_path, "w", **out_meta) as dest:
        dest.write(out_image)
        
        
def run_distance_to_roads(aoi_path, output_dir, roads_path, temp_dir="data/temp", buffer_m=1000):
    print("🏃 Running distance-to-roads pipeline...")
    os.makedirs(output_dir, exist_ok=True)
    aoi = gpd.read_file(aoi_path)
    buffered = buffer_aoi(aoi, buffer_m=buffer_m, temp_dir=temp_dir)
    roads = load_roads_clipped(roads_path, buffered, temp_dir=temp_dir)

    tmp_raster = compute_distance_raster(buffered, roads, temp_dir=temp_dir)
    final_raster = os.path.join(output_dir, "distance-to-roads.tif")

    clip_raster_to_aoi(tmp_raster, aoi, final_raster)
    os.remove(tmp_raster)

    print(f"✅ Final clipped distance raster saved to: {final_raster}\n")
