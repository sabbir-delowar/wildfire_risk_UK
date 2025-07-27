# Wildfire Risk UK

A pipeline to generate wildfire risk maps for regions in the United Kingdom using open geospatial datasets and a Random Forest model.

## Pipeline Overview
1. Download Sentinel-2 imagery and compute NDVI/NDWI composites.
2. Extract Digital Elevation Model (DEM) tiles.
3. Retrieve land cover and population data.
4. Measure distance to roads and urban areas.
5. Download ERA5 climate variables (temperature, dew point and precipitation).
6. Collect historic fire information from FIRMS.
7. Build training labels from historic fires.
8. Stack all raster features and create balanced training samples.
9. Train a Random Forest classifier and evaluate it.
10. Deploy the model to classify wildfire risk.
11. Post-process predictions and generate visualisations.

## Usage
Run the complete workflow:
```bash
python main.py
```
The script prompts for an AOI GeoJSON file and a roads dataset and writes processed data to `data/processed` and predictions to `outputs`.

## Repository Structure
- `src/feature_extractions/` – feature extraction modules for satellite, climate and vector data.
- `src/ml/` – scripts for labelling, stacking, training, deployment and visualisation.
- `data/` – sample AOI and label data.
- `models/` – trained models.
- `outputs/` – example risk maps.

## Requirements
Install the necessary Python packages:
```bash
pip install requirement.txt
```

## License
MIT © Sabbir Delowar
