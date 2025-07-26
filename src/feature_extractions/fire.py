import os
import ee
import geemap
import geopandas as gpd


def initialise_gee():
    try:
        ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize()


def load_aoi(aoi_path):
    gdf = gpd.read_file(aoi_path)
    if gdf.crs is None:
        print("⚠️ AOI has no CRS. Assuming EPSG:4326.")
        gdf.set_crs("EPSG:4326", inplace=True)
    else:
        gdf = gdf.to_crs("EPSG:4326")
        
    # Simplify the geometry (adjust tolerance as needed)
    gdf["geometry"] = gdf["geometry"].simplify(tolerance=0.0005, preserve_topology=True)
    return geemap.geopandas_to_ee(gdf)


def load_firms_fire_count(aoi, start_date="2020-01-01", end_date="2024-12-31"):
    print("📦 Loading FIRMS VIIRS fire image collection...")
    collection = (
        ee.ImageCollection("FIRMS")
        .filterDate(start_date, end_date)
        .filterBounds(aoi)
        .select("T21")
    )
    fire_count = collection.count().clip(aoi).rename("fire_count")
    return fire_count.toUint16()


def export_fire_count_raster(image, aoi, output_tif, scale=375):
    print(f"⬇️ Downloading raster: {output_tif}")
    geemap.download_ee_image(
        image=image,
        region=aoi.geometry(),
        filename=output_tif,
        scale=scale,
        crs="EPSG:4326",
        dtype="uint16",
        max_tile_size=8
    )
    print(f"✅ Fire count raster saved to: {output_tif}")


def run_fire_extraction_firms(aoi_path, output_dir, start_date="2020-01-01", end_date="2024-12-31"):
    initialise_gee()
    aoi = load_aoi(aoi_path)
    fire_image = load_firms_fire_count(aoi, start_date, end_date)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"firms_fire_count_{start_date[:4]}_{end_date[:4]}.tif")

    export_fire_count_raster(fire_image, aoi, out_path)
    return out_path
