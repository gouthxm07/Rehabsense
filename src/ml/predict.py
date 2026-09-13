"""
Loads the trained model/scaler and exposes a single predict() call
used by main.py during the live camera loop.
"""

from pathlib import Path
import joblib
import numpy as np

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
FEATURES = ["knee_angle", "hip_angle", "elbow_angle", "shoulder_angle", "velocity", "rom"]


class ExerciseClassifier:
    def __init__(self):
        self.model = joblib.load(MODELS_DIR / "exercise_classifier.pkl")
        self.scaler = joblib.load(MODELS_DIR / "scaler.pkl")

    def predict(self, feature_dict):
        """feature_dict must contain all keys in FEATURES."""
        x = np.array([[feature_dict.get(f, 0.0) for f in FEATURES]])
        x_scaled = self.scaler.transform(x)
        prediction = self.model.predict(x_scaled)[0]
        confidence = float(np.max(self.model.predict_proba(x_scaled)))
        return prediction, confidence
