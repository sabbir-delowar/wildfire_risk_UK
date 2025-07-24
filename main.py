import os

def main():
    
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    
    
    # Sentinel-2 feature extraction
    from src.feature_extractions.sentinel import run_feature_extraction
    
    print("🔥 Static Wildfire Risk Feature Extractor")
    aoi_path = input("Enter AOI path [default: data/aoi/aoi.geojson]: ").strip()
    if not aoi_path:
        aoi_path = "data/aoi/aoi.geojson"

    if not os.path.exists(aoi_path):
        print(f"❌ AOI file not found: {aoi_path}")
        return

    print("📦 Starting Sentinel-2 feature extractions...")
    # run_feature_extraction(aoi_path, output_dir=output_dir)
    print("✅ Sentinel-2 feature export completed.\n")
    
    
    # DEM feature extraction
    from src.feature_extractions.dem import run_dem_extraction
    
    print("📦 Starting DEM extraction...")
    # run_dem_extraction(aoi_path, output_dir)
    print("✅ DEM block done.\n")
    
    # Landcover feature extraction
    from src.feature_extractions.landcover_corine import run_corine_extraction

    print("📦 Starting Landcover extraction...")
    run_corine_extraction(aoi_path, output_dir)
    print("✅ Landcover block done.\n")

if __name__ == "__main__":
    main()
