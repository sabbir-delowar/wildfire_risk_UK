import os
import numpy as np
import rasterio
import joblib

def classify_raster(model_path, features_path, label_template_path, output_path):
    # Load model
    print("📦 Loading model...")
    model = joblib.load(model_path)

    # Load features
    print("📦 Loading feature stack...")
    X = np.load(features_path)  # Shape: (H, W, bands)
    height, width, bands = X.shape
    X_reshaped = X.reshape(-1, bands)

    # Mask invalid pixels (any NaN in features)
    valid_mask = ~np.isnan(X_reshaped).any(axis=1)

    # Create empty prediction array
    predictions = np.full(X_reshaped.shape[0], 255, dtype=np.uint8)  # 255 = nodata

    # Predict only on valid pixels
    print("🤖 Running prediction on valid pixels...")
    predictions[valid_mask] = model.predict(X_reshaped[valid_mask])

    # Reshape back to 2D
    prediction_raster = predictions.reshape((height, width))

    # Use metadata from label raster
    with rasterio.open(label_template_path) as src:
        meta = src.meta.copy()
        meta.update(dtype="uint8", count=1, nodata=255)

    # Save prediction raster
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(prediction_raster, 1)

    print(f"✅ Prediction raster saved to: {output_path}")
