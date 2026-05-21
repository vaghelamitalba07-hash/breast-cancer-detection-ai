from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "breast_cancer_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
USERS_PATH = BASE_DIR / "data" / "users.json"
FRONTEND_DIR = BASE_DIR.parent / "frontend"

FEATURE_NAMES = [
    "radius_mean", "radius_se", "radius_worst",
    "texture_mean", "texture_se", "texture_worst",
    "perimeter_mean", "perimeter_se", "perimeter_worst",
    "area_mean", "area_se", "area_worst",
    "smoothness_mean", "smoothness_se", "smoothness_worst",
    "compactness_mean", "compactness_se", "compactness_worst",
    "concavity_mean", "concavity_se", "concavity_worst",
    "concave_points_mean", "concave_points_se", "concave_points_worst",
    "symmetry_mean", "symmetry_se", "symmetry_worst",
    "fractal_dimension_mean", "fractal_dimension_se", "fractal_dimension_worst",
]

DEFAULT_FEATURE_VALUES = {
    "radius_mean": 12.0, "radius_se": 0.25,
    "texture_mean": 18.0, "texture_se": 0.5,
    "perimeter_mean": 78.0, "perimeter_se": 1.5,
    "area_mean": 450.0, "area_se": 20.0,
    "smoothness_mean": 0.095, "smoothness_se": 0.007,
    "compactness_mean": 0.1, "compactness_se": 0.015,
    "compactness_worst": 0.15, "concavity_se": 0.02,
    "concave_points_se": 0.01, "symmetry_mean": 0.18,
    "symmetry_se": 0.02, "fractal_dimension_mean": 0.062,
    "fractal_dimension_se": 0.003, "fractal_dimension_worst": 0.08,
}
