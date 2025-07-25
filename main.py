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
    
    
    '''Sentinel-2 feature extraction'''
    from src.feature_extractions.sentinel import run_feature_extraction
    print("📦 Starting Sentinel-2 feature extractions...")
    run_feature_extraction(aoi_path, output_dir=output_dir)
    print("✅ Sentinel-2 feature export completed.\n")
    
    
    '''DEM feature extraction'''
    from src.feature_extractions.dem import run_dem_extraction
    print("📦 Starting DEM extraction...")
    run_dem_extraction(aoi_path, output_dir)
    print("✅ DEM block done.\n")
    
    '''Landcover feature extraction'''
    from src.feature_extractions.landcover import run_landcover_extraction
    print("📦 Starting Landcover extraction...")
    run_landcover_extraction(aoi_path, output_dir)
    print("✅ Landcover block done.\n")
    
    
    '''Population feature extraction'''
    from src.feature_extractions.population import run_population_extraction
    print("📦 Starting Population extraction...")
    run_population_extraction(aoi_path, output_dir)
    print("✅ Population block done.\n")
    
    
    '''Road distance feature extraction'''
    from src.feature_extractions.roads import run_distance_to_roads
    osm_path = input("Enter OSM PBF path [default: data/raw/england-latest-free.shp/gis_osm_roads_free_1.shp]:").strip()
    if not osm_path:
        osm_path = "data/raw/england-latest-free.shp/gis_osm_roads_free_1.shp"

    if not os.path.exists(osm_path):
        print(f"❌ AOI file not found: {osm_path}")
        return
    
    print("📦 Road distance extraction...")
    run_distance_to_roads(aoi_path, output_dir, osm_path)
    print("✅ Road distance block done.\n")
    
    
    '''ERA5 feature extraction'''
    from src.feature_extractions.ERA5 import run_monthly_climate_extraction, create_mean_composites
    print("📦 Starting ERA5 data downloading...")
    run_monthly_climate_extraction(aoi_path, output_dir)
    print("📦 Starting ERA5 features extraction...")
    create_mean_composites(output_dir=output_dir)
    print("✅ ERA5 block done.\n")
    
    
    '''Urban distance feature extraction'''
    from src.feature_extractions.urban import run_distance_to_urban

    print("📦 Starting distance to urban feature extraction...")
    run_distance_to_urban(aoi_path, output_dir)
    print("✅ Distance to urban feature extraction done.\n")
    
    
    '''FIRE feature extraction'''
    from src.feature_extractions.fire import run_fire_extraction

    print("📦 Starting Historic Fire extraction...")
    run_fire_extraction(aoi_path, output_dir)
    print("✅ Historic Fire block done.\n")
if __name__ == "__main__":
    main()
