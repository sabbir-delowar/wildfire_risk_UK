import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def train_and_evaluate_model(X_train, y_train, X_test, y_test, save_model=True, model_dir="models"):
    """
    Trains a Random Forest classifier and evaluates it.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test labels.
        save_model (bool): If True, save model to disk.
        model_dir (str): Directory to save model.

    Returns:
        model (RandomForestClassifier): Trained model.
    """
    print("🚀 Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    print("🧪 Evaluating model...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc:.4f}")
    print("📊 Classification Report:")
    print(classification_report(y_test, y_pred))

    if save_model:
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "rf_model.pkl")
        joblib.dump(model, model_path)
        print(f"💾 Model saved to: {model_path}")

    return model
