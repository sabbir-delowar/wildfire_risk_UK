import ee
import geemap
import geopandas as gpd
import os

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


def get_feature_composites(aoi, years=[2020, 2021, 2022, 2023, 2024]):
    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
        .filterBounds(aoi) \
        .filter(ee.Filter.calendarRange(6, 8, 'month')) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))

    def add_indices(img):
        ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
        ndwi = img.normalizedDifference(['B3', 'B8']).rename('NDWI')
        return img.addBands(ndvi).addBands(ndwi)

    s2_indexed = s2.map(add_indices)
    ndvi = s2_indexed.select('NDVI').median().clip(aoi)
    ndwi = s2_indexed.select('NDWI').median().clip(aoi)

    return ndvi, ndwi


def export_to_local(image, region, filepath, scale=10, crs="EPSG:4326"):
    print(f"⬇️ Downloading to: {filepath}")
    geemap.download_ee_image(
        image=image,
        region=region.geometry(),
        filename=filepath,
        scale=scale,
        crs=crs,
        max_tile_size=8  # LIMIT TILE SIZE TO REDUCE MEMORY BURST
    )

def run_feature_extraction(aoi_path, output_dir, years=[2020, 2021, 2022, 2023, 2024]):
    initialise_gee()
    aoi = load_aoi(aoi_path)
    ndvi, ndwi = get_feature_composites(aoi, years)

    export_to_local(ndvi, aoi, os.path.join(output_dir, "ndvi-2020-2024.tif"))
    export_to_local(ndwi, aoi, os.path.join(output_dir, "ndwi-2020-2024.tif"))
