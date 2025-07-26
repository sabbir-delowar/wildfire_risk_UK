import os
import ee
import geemap
import geopandas as gpd
from shapely.geometry import box
from glob import glob
import rasterio
from rasterio.merge import merge
from src.feature_extractions.dem import load_aoi_with_bounds  # same as before

def initialise_gee():
    try:
        ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize()

        
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


def export_tile(image, tile_geom, filepath, scale=100):
    gdf = gpd.GeoDataFrame(geometry=[tile_geom], crs="EPSG:4326")
    ee_tile = geemap.geopandas_to_ee(gdf)

    try:
        geemap.download_ee_image(
            image=image,
            region=ee_tile.bounds(),  # Use bounds, not full geometry
            filename=filepath,
            scale=scale,
            crs="EPSG:4326",
            max_tile_size=4
        )
    except Exception as e:
        print(f"❌ Failed to export tile {os.path.basename(filepath)}: {e}")

        
def run_population_extraction_tiled(aoi_path, output_dir="data/temp/population", tile_size_deg=1):
    initialise_gee()
    aoi, bounds = load_aoi_with_bounds(aoi_path)

    image = ee.Image("WorldPop/GP/100m/pop/GBR_2020").select("population").rename("pop_density")

    os.makedirs(output_dir, exist_ok=True)
    tiles = split_aoi_into_tiles(bounds, tile_size_deg)

    for i, tile_geom in enumerate(tiles):
        out_fp = os.path.join(output_dir, f"pop_tile{i+1}.tif")
        if not os.path.exists(out_fp):
            print(f"⬇️ Exporting tile {i+1}/{len(tiles)}")
            export_tile(image, tile_geom, out_fp)
        else:
            print(f"✅ Tile {i+1} already exists, skipping.")

    print("✅ All population tiles exported.")

    
def merge_population_tiles(output_dir, input_dir="data/temp/population"):
    print("🔄 Merging population tiles...")

    tile_files = sorted(glob(os.path.join(input_dir, "pop_tile*.tif")))

    if not tile_files:
        raise FileNotFoundError(f"No population tiles found in: {input_dir}")

    src_files_to_mosaic = [rasterio.open(fp) for fp in tile_files]
    mosaic, out_trans = merge(src_files_to_mosaic)

    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "count": 1,
        "dtype": mosaic.dtype,
        "compress": "deflate"
    })

    out_path = os.path.join(output_dir, "population_2020_merged.tif")
    with rasterio.open(out_path, "w", **out_meta) as dest:
        dest.write(mosaic[0], 1)

    for src in src_files_to_mosaic:
        src.close()

    print(f"✅ Merged population raster saved to: {out_path}")
