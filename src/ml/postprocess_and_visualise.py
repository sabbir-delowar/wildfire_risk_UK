import os
import numpy as np
import rasterio
from rasterio.enums import Resampling
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

def postprocess_and_visualise_prediction(
    input_path="outputs/fire_risk_prediction_rf.tif",
    smoothed_output_path="outputs/fire_risk_prediction_rf_smoothed.tif",
    jpg_output_path="outputs/fire_risk_map.jpg",
    smoothing_sigma=1.0
):
    """
    Smooths prediction raster using Gaussian filter, exports GeoTIFF and visual JPG map.

    Args:
        input_path (str): Path to raw model prediction (GeoTIFF).
        smoothed_output_path (str): Path to save smoothed GeoTIFF.
        jpg_output_path (str): Path to save the JPG visualisation.
        smoothing_sigma (float): Sigma for Gaussian smoothing.
    """

    print("📦 Loading prediction raster...")
    with rasterio.open(input_path) as src:
        prediction = src.read(1).astype(np.float32)
        meta = src.meta.copy()
        original_nodata = src.nodata if src.nodata is not None else 255

    # Mask out no-data values
    mask = (prediction == original_nodata)
    prediction[mask] = np.nan

    print("🔧 Applying Gaussian smoothing...")
    smoothed = gaussian_filter(np.nan_to_num(prediction, nan=0.0), sigma=smoothing_sigma)

    # Restore NaN to no-data areas
    smoothed[mask] = np.nan
    final_nodata = -9999.0
    smoothed_to_write = np.where(np.isnan(smoothed), final_nodata, smoothed)

    print(f"💾 Saving smoothed raster to: {smoothed_output_path}")
    meta.update(dtype="float32", count=1, nodata=final_nodata)
    os.makedirs(os.path.dirname(smoothed_output_path), exist_ok=True)
    with rasterio.open(smoothed_output_path, 'w', **meta) as dst:
        dst.write(smoothed_to_write, 1)

    print(f"🖼️ Creating visualisation JPG: {jpg_output_path}")
    plt.figure(figsize=(12, 8))
    masked_smoothed = np.ma.masked_where(np.isnan(smoothed), smoothed)
    im = plt.imshow(masked_smoothed, cmap='hot', vmin=0, vmax=1)
    plt.title("Wildfire Risk Prediction (Smoothed)", fontsize=16)
    plt.axis('off')
    cbar = plt.colorbar(im, label="Fire Probability (0–1)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(jpg_output_path), exist_ok=True)
    plt.savefig(jpg_output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("✅ Postprocessing and visualisation complete.")
