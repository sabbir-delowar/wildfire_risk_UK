import os
import ee
import geemap
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.vrt import WarpedVRT
from glob import glob
from shapely.geometry import box


def initialise_gee():
    try:
        ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize()

        
def load_aoi(aoi_path):
    gdf = gpd.read_file(aoi_path)

    # If CRS is missing, assign manually
    if gdf.crs is None:
        print("⚠️ AOI has no CRS. Forcing to EPSG:4326.")
        gdf.set_crs("EPSG:4326", inplace=True)

    # Reproject to EPSG:4326
    gdf = gdf.to_crs("EPSG:4326")

    # Use the reprojected GeoDataFrame directly — no manual geometry union!
    return geemap.geopandas_to_ee(gdf)


def split_aoi_into_tiles(gdf, tile_size_deg=1):
    """Split AOI into smaller tiles in degrees."""
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    minx, miny, maxx, maxy = bounds
    tiles = []
    y = miny
    while y < maxy:
        x = minx
        while x < maxx:
            tiles.append(box(x, y, min(x + tile_size_deg, maxx), min(y + tile_size_deg, maxy)))
            x += tile_size_deg
        y += tile_size_deg
    return [geemap.geopandas_to_ee(gpd.GeoDataFrame(geometry=[tile], crs="EPSG:4326")) for tile in tiles]


def get_index_image(index_type, year, aoi_tile):
    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
        .filterBounds(aoi_tile) \
        .filter(ee.Filter.calendarRange(6, 8, 'month')) \
        .filter(ee.Filter.calendarRange(year, year, 'year')) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))

    def add_indices(img):
        ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
        ndwi = img.normalizedDifference(['B3', 'B8']).rename('NDWI')
        return img.addBands([ndvi, ndwi])

    with_indices = s2.map(add_indices)
    image = with_indices.select(index_type).median().clip(aoi_tile)
    return image

def export_image(image, region, filepath, scale=100, crs="EPSG:4326"):
    print(f"⬇️ Downloading: {filepath}")
    geemap.download_ee_image(
        image=image,
        region=region.geometry(),
        filename=filepath,
        scale=scale,
        crs=crs,
        max_tile_size=4
    )

    
def run_sentinel_index_download(aoi_path, index_type, output_dir="data/temp/sentinel", years=[2020, 2021, 2022, 2023, 2024]):
    initialise_gee()
    print(f"📦 Starting Sentinel download for index: {index_type}")

    aoi_gdf = load_aoi(aoi_path)
    tiles = split_aoi_into_tiles(aoi_gdf, tile_size_deg=0.2)
    os.makedirs(output_dir, exist_ok=True)

    for year in years:
        os.makedirs(year_dir, exist_ok=True)

        for i, tile in enumerate(tiles):
            out_path = os.path.join(output_dir, f"{index_type.lower()}_tile{i}.tif")
            if os.path.exists(out_path):
                print(f"✅ Already exists: {out_path}")
                continue
            try:
                image = get_index_image(index_type.upper(), year, tile)
                export_image(image, tile, out_path)
            except Exception as e:
                print(f"❌ Failed for {year} tile {i}: {e}")

    print(f"✅ All Sentinel-{index_type} data downloaded.\n")


def compute_mean_composite(index_type, output_dir, index_dir="data/temp/sentinel"):
    """
    Computes mean composite from yearly rasters (e.g., NDVI, NDWI),
    aligning all rasters to the common union extent and resolution.
    """
    print(f"📦 Calculating mean for {index_type}...")

    search_path = os.path.join(index_dir, index_type.lower())
    file_pattern = os.path.join(search_path, "**", f"{index_type.lower()}_*.tif")
    index_files = sorted(glob(file_pattern, recursive=True))

    if not index_files:
        raise FileNotFoundError(f"No {index_type} files found in {search_path}/**")

    # Step 1: Use first file as reference
    with rasterio.open(index_files[0]) as ref:
        dst_crs = ref.crs
        dst_res = ref.res
        dtype = "float32"

    # Step 2: Compute union bounds
    bounds_list = []
    for fp in index_files:
        with rasterio.open(fp) as src:
            bounds = src.bounds
            bounds_list.append(bounds)
    minx = min(b[0] for b in bounds_list)
    miny = min(b[1] for b in bounds_list)
    maxx = max(b[2] for b in bounds_list)
    maxy = max(b[3] for b in bounds_list)

    dst_width = int((maxx - minx) / dst_res[0])
    dst_height = int((maxy - miny) / dst_res[1])
    dst_transform = rasterio.transform.from_origin(minx, maxy, *dst_res)

    arrays = []

    for fp in index_files:
        with rasterio.open(fp) as src:
            with WarpedVRT(
                src,
                crs=dst_crs,
                transform=dst_transform,
                width=dst_width,
                height=dst_height,
                resampling=Resampling.bilinear
            ) as vrt:
                data = vrt.read(1).astype(np.float32)
                if vrt.nodata is not None:
                    data = np.where(data == vrt.nodata, np.nan, data)
                arrays.append(data)

    # Step 3: Stack and compute mean
    stack = np.stack(arrays)
    mean = np.nanmean(stack, axis=0)

    # Step 4: Save output
    meta = {
        "driver": "GTiff",
        "height": dst_height,
        "width": dst_width,
        "count": 1,
        "dtype": dtype,
        "crs": dst_crs,
        "transform": dst_transform,
        "nodata": np.nan
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{index_type}_mean_2020_2024.tif")
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(mean.astype(dtype), 1)

    print(f"✅ Saved mean {index_type} to: {out_path}")
