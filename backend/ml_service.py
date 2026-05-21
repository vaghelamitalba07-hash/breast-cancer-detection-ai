"""ML model load and prediction (same logic as original prototype)."""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from config import (
    DEFAULT_FEATURE_VALUES,
    FEATURE_NAMES,
    MODEL_PATH,
    SCALER_PATH,
)

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "breast-cancer-wisconsin/wdbc.data"
)

_model = None
_scaler = None


def _train_and_save():
    metric_names = [
        "radius", "texture", "perimeter", "area", "smoothness",
        "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension",
    ]
    stat_names = ["mean", "se", "worst"]
    columns = ["id", "diagnosis"]
    for metric in metric_names:
        for stat in stat_names:
            columns.append(f"{metric}_{stat}")

    df = pd.read_csv(UCI_URL, header=None, names=columns)
    df.drop("id", axis=1, inplace=True)
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

    x_data = df.drop("diagnosis", axis=1)
    y_data = df["diagnosis"]

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_data)

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    model.fit(x_scaled, y_data)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    return model, scaler


def get_model():
    global _model, _scaler
    if _model is None or _scaler is None:
        if not MODEL_PATH.exists() or not SCALER_PATH.exists():
            _model, _scaler = _train_and_save()
        else:
            _model = joblib.load(MODEL_PATH)
            _scaler = joblib.load(SCALER_PATH)
    return _model, _scaler


def predict(features: dict) -> dict:
    """Run prediction; missing features use dataset means."""
    model, scaler = get_model()
    row = {name: DEFAULT_FEATURE_VALUES.get(name, 0.0) for name in FEATURE_NAMES}
    row.update(features)
    df = pd.DataFrame([row])[FEATURE_NAMES]
    scaled = scaler.transform(df)
    probability = float(model.predict_proba(scaled)[0, 1])
    pct = round(probability * 100, 1)

    if probability >= 0.3:
        level = "elevated"
        message = "Model assigns higher probability to malignancy. NOT a diagnosis."
    elif probability >= 0.1:
        level = "intermediate"
        message = "Intermediate probability. NOT a diagnosis."
    else:
        level = "lower"
        message = "Lower probability to malignancy. NOT a diagnosis."

    return {
        "probability": probability,
        "probability_percent": pct,
        "level": level,
        "message": message,
    }


def get_metrics() -> dict:
    """Static research metrics (from original app)."""
    return {
        "roc_auc": 0.990,
        "test_accuracy": 0.971,
        "thresholds": [
            {"threshold": "0.5 (default)", "sensitivity": "93.8%", "specificity": "99.1%", "fn": 4, "fp": 1},
            {"threshold": "0.3 (balanced)", "sensitivity": "96.9%", "specificity": "97.2%", "fn": 2, "fp": 3},
            {"threshold": "0.2 (sensitive)", "sensitivity": "98.4%", "specificity": "91.6%", "fn": 1, "fp": 10},
        ],
    }
