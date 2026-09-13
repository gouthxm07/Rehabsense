"""
Trains the exercise-recognition classifier (blueprint section 12).

For a hackathon, you have two options:
  1. Record real sessions with main.py (it logs features to data/raw/features_log.csv
     if you set LOG_FEATURES=True), label them, then train on that file.
  2. Use the synthetic generator below to get a working model in minutes so the
     rest of the pipeline (scoring, dashboard, storage) can be built/demoed
     while real data collection happens in parallel.

Run:  python -m src.ml.train
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
FEATURES = ["knee_angle", "hip_angle", "elbow_angle", "shoulder_angle", "velocity", "rom"]

# Rough angle envelopes per exercise used only to generate believable synthetic
# examples. Replace this file's synthetic data with real logged sessions ASAP.
EXERCISE_PROFILES = {
    "knee_flexion":         {"knee_angle": (90, 170), "hip_angle": (150, 180), "elbow_angle": (150, 180), "shoulder_angle": (10, 30), "velocity": (10, 120), "rom": (40, 90)},
    "straight_leg_raise":   {"knee_angle": (160, 180), "hip_angle": (120, 175), "elbow_angle": (150, 180), "shoulder_angle": (10, 30), "velocity": (10, 90),  "rom": (20, 50)},
    "shoulder_raise":       {"knee_angle": (160, 180), "hip_angle": (160, 180), "elbow_angle": (150, 180), "shoulder_angle": (15, 100), "velocity": (10, 100), "rom": (50, 90)},
    "elbow_flexion":        {"knee_angle": (160, 180), "hip_angle": (160, 180), "elbow_angle": (50, 170), "shoulder_angle": (10, 40), "velocity": (10, 130), "rom": (60, 110)},
}


def generate_synthetic_dataset(n_per_class=300, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for exercise, ranges in EXERCISE_PROFILES.items():
        for _ in range(n_per_class):
            row = {feat: rng.uniform(*bounds) for feat, bounds in ranges.items()}
            row["exercise"] = exercise
            rows.append(row)
    df = pd.DataFrame(rows)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_DIR / "synthetic_training_data.csv", index=False)
    return df


def train(df=None):
    if df is None:
        df = generate_synthetic_dataset()

    X = df[FEATURES]
    y = df["exercise"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)
    print(classification_report(y_test, predictions))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "exercise_classifier.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    print(f"\nSaved model + scaler to {MODELS_DIR}")


if __name__ == "__main__":
    train()
