import os

def main():
    
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    
    print("🔥 Static Wildfire Risk Feature Extractor")
    aoi_path = input("Enter AOI path [default: data/aoi/aoi.geojson]: ").strip()
    if not aoi_path:
        aoi_path = "data/aoi/aoi.geojson"

    if not os.path.exists(aoi_path):
        print(f"❌ AOI file not found: {aoi_path}")
        return
    
    osm_path = input("Enter OSM PBF path [default: data/raw/england-latest-free.shp/gis_osm_roads_free_1.shp]:").strip()
    if not osm_path:
        osm_path = "data/raw/england-latest-free.shp/gis_osm_roads_free_1.shp"

    if not os.path.exists(osm_path):
        print(f"❌ OSM file not found: {osm_path}")
        return
    
    
    '''Sentinel-2 feature extraction'''
    from src.feature_extractions.sentinel import run_sentinel_index_download, compute_mean_composite
    print("📦 Starting Sentinel-2 NDVI download...")
    # run_sentinel_index_download(aoi_path, "NDVI")
    print("📦 Starting Sentinel-2 NDWI download...")
    # run_sentinel_index_download(aoi_path, "NDWI")
    print("📦 Starting Sentinel-2 feature extractions...")
    # compute_mean_composite("NDVI", output_dir)
    # compute_mean_composite("NDWI", output_dir)
    print("✅ Sentinel-2 feature export completed.\n")
    
    
    '''DEM feature extraction'''
    from src.feature_extractions.dem import run_dem_extraction_tiled, merge_dem_tiles
    print("📦 Starting DEM extrdownloadaction...")
    # run_dem_extraction_tiled(aoi_path)
    print("📦 Starting DEM extraction...")
    merge_dem_tiles(output_dir)
    print("✅ DEM block done.\n")
    
    
    '''Landcover feature extraction'''
    from src.feature_extractions.landcover import run_landcover_extraction_tiled, merge_landcover_tiles
    print("📦 Starting Landcover downloading...")
    # run_landcover_extraction_tiled(aoi_path)
    print("📦 Starting Landcover extraction...")
    # merge_landcover_tiles(output_dir)
    print("✅ Landcover block done.\n")
    
    
    '''Population feature extraction'''
    from src.feature_extractions.population import run_population_extraction_tiled, merge_population_tiles
    print("📦 Starting Population downloding...")
    # run_population_extraction_tiled(aoi_path)
    print("📦 Starting Population extraction...")
    # merge_population_tiles(output_dir)
    print("✅ Population block done.\n")
    
    
    '''Road distance feature extraction'''
    from src.feature_extractions.roads import run_distance_to_roads
    print("📦 Road distance extraction...")
    # run_distance_to_roads(aoi_path, output_dir, osm_path)
    print("✅ Road distance block done.\n")
    
    
    '''ERA5 feature extraction'''
    from src.feature_extractions.ERA5 import extract_and_export_era5_monthly, compute_era5_mean_composites
    print("📦 Starting Temperature data downloading...")
    # extract_and_export_era5_monthly(aoi_path, "temperature_2m")
    print("📦 Starting Dew point data downloading...")
    # extract_and_export_era5_monthly(aoi_path, "dewpoint_temperature_2m")
    print("📦 Starting ERA5 Precipitation downloading...")
    # extract_and_export_era5_monthly(aoi_path, "total_precipitation_sum")
    print("📦 Starting ERA5 features extraction...")
    compute_era5_mean_composites(output_dir, "temperature_2m")
    compute_era5_mean_composites(output_dir, "dewpoint_temperature_2m")
    compute_era5_mean_composites(output_dir, "total_precipitation_sum")
    print("✅ ERA5 block done.\n")
    
    
    '''Urban distance feature extraction'''
    from src.feature_extractions.urban import run_distance_to_urban
    print("📦 Starting distance to urban feature extraction...")
    # run_distance_to_urban(aoi_path, output_dir)
    print("✅ Distance to urban feature extraction done.\n")
    
    
    '''FIRE feature extraction'''
    from src.feature_extractions.fire import run_fire_extraction_firms
    print("📦 Starting Historic Fire downloading...")
    # run_fire_extraction_firms(aoi_path, output_dir)
    print("✅ Historic Fire block done.\n")
    
    
if __name__ == "__main__":
    main()
