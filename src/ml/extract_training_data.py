from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import rasterio
import os
from sklearn.utils import resample

def extract_balanced_training_data(test_size=0.3):
    label_path = "data/labels/fire_label.tif"
    stack_path = "data/processed/features_stack.npy"

    # Load label and feature stack
    X = np.load(stack_path)  # (H, W, bands)
    with rasterio.open(label_path) as src:
        y = src.read(1)  # (H, W)

    X_flat = X.reshape(-1, X.shape[2])
    y_flat = y.flatten()

    # Only valid (no NaNs) and binary labels
    mask_fire = y_flat == 1
    mask_no_fire = (y_flat == 0) & (~np.isnan(X_flat).any(axis=1))
    mask_fire = mask_fire & (~np.isnan(X_flat).any(axis=1))

    # Extract pixels
    X_fire = X_flat[mask_fire]
    y_fire = y_flat[mask_fire]
    X_no_fire = X_flat[mask_no_fire]
    y_no_fire = y_flat[mask_no_fire]

    # Balance the classes
    n_fire = len(y_fire)
    X_no_fire_downsampled, y_no_fire_downsampled = resample(
        X_no_fire, y_no_fire, replace=False, n_samples=n_fire, random_state=42
    )

    X_final = np.vstack((X_fire, X_no_fire_downsampled))
    y_final = np.concatenate((y_fire, y_no_fire_downsampled))

    print(f"✅ Balanced data: {Counter(y_final)}")

    return train_test_split(X_final, y_final, test_size=test_size, random_state=42, stratify=y_final)
