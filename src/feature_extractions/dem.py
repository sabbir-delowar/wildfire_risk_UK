import os
import geopandas as gpd
import ee
import geemap
from shapely.geometry import box
import rasterio
from rasterio.merge import merge
from glob import glob
import numpy as np


def initialise_gee():
    try:
        ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize()

        
def load_aoi_with_bounds(aoi_path):
    gdf = gpd.read_file(aoi_path)
    if gdf.crs is None:
        print("⚠️ AOI has no CRS. Forcing to EPSG:4326.")
        gdf.set_crs("EPSG:4326", inplace=True)
    gdf = gdf.to_crs("EPSG:4326")
    return geemap.geopandas_to_ee(gdf), gdf.total_bounds


def split_aoi_into_tiles(bounds, tile_size_deg=1):
    minx, miny, maxx, maxy = bounds
    tiles = []
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            tile = box(x, y, min(x + tile_size_deg, maxx), min(y + tile_size_deg, maxy))
            tiles.append(tile)
            y += tile_size_deg
        x += tile_size_deg
    return tiles


def export_tile(image, tile_geom, filepath, scale=30):
    gdf = gpd.GeoDataFrame(geometry=[tile_geom], crs="EPSG:4326")
    ee_tile = geemap.geopandas_to_ee(gdf)

    try:
        geemap.download_ee_image(
            image=image,
            region=ee_tile.geometry(),
            filename=filepath,
            scale=scale,
            crs="EPSG:4326",
            max_tile_size=4
        )
    except Exception as e:
        print(f"❌ Failed to export tile: {e}")

        
def run_dem_extraction_tiled(aoi_path, output_dir="data/temp/dem", tile_size_deg=1):
    initialise_gee()
    aoi, bounds = load_aoi_with_bounds(aoi_path)
    os.makedirs(output_dir, exist_ok=True)

    dem = (
        ee.ImageCollection("COPERNICUS/DEM/GLO30")
        .mosaic()
        .select("DEM")
        .rename("elevation")
    )

    os.makedirs(output_dir, exist_ok=True)
    tiles = split_aoi_into_tiles(bounds, tile_size_deg=tile_size_deg)

    for i, tile_geom in enumerate(tiles):
        out_fp = os.path.join(output_dir, f"elevation_tile{i+1}.tif")
        if not os.path.exists(out_fp):
            print(f"⬇️ Exporting tile {i+1}/{len(tiles)}")
            export_tile(dem, tile_geom, out_fp)
        else:
            print(f"✅ Tile {i+1} already exists, skipping.")

    print("✅ All DEM tiles exported.")

    
def merge_dem_tiles(output_dir, input_dir="data/temp/dem"):
    print("🔄 Merging DEM tiles...")

    search_pattern = os.path.join(input_dir, "elevation_tile*.tif")
    dem_files = sorted(glob(search_pattern))

    if not dem_files:
        raise FileNotFoundError(f"No DEM tiles found in: {input_dir}")

    src_files_to_mosaic = [rasterio.open(fp) for fp in dem_files]
    mosaic, out_trans = merge(src_files_to_mosaic)

    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "count": 1,
        "dtype": mosaic.dtype
    })

    out_path = os.path.join(output_dir, "elevation_merged.tif")
    with rasterio.open(out_path, "w", **out_meta) as dest:
        dest.write(mosaic[0], 1)

    for src in src_files_to_mosaic:
        src.close()

    print(f"✅ Merged DEM saved to: {out_path}")
