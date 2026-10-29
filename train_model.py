
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "synthetic_flights.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "flight_delay_model.joblib"
METRICS_PATH = MODEL_DIR / "model_metrics.json"

# Features available before departure.
# Intentionally exclude delay_minutes, is_delayed, delay_reason,
# and other fields that reveal the actual outcome.
CATEGORICAL_FEATURES = [
    "airline",
    "airline_code",
    "origin",
    "destination",
    "weather_condition",
]

NUMERIC_FEATURES = [
    "departure_hour",
    "scheduled_duration_min",
    "distance_km",
    "visibility_km",
    "wind_speed_kmh",
]

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "is_delayed"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Check that data/synthetic_flights.csv exists."
        )

    df = pd.read_csv(DATA_PATH)

    required = FEATURES + [TARGET]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in the dataset: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    # Convert target values safely to 0 and 1.
    if df[TARGET].dtype == object:
        target_values = (
            df[TARGET].astype(str).str.strip().str.lower()
        )
        target_map = {
            "true": 1, "false": 0,
            "1": 1, "0": 0,
            "yes": 1, "no": 0,
        }
        y = target_values.map(target_map)
    else:
        y = pd.to_numeric(df[TARGET], errors="coerce")

    # Remove rows with missing target values.
    valid = y.isin([0, 1])
    X = df.loc[valid, FEATURES].copy()
    y = y.loc[valid].astype(int)

    if len(X) < 20:
        raise ValueError("Not enough valid rows to train the model.")

    if y.nunique() < 2:
        raise ValueError(
            "The target must contain both delayed and non-delayed flights."
        )

    # Convert feature types.
    for col in NUMERIC_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].fillna("Unknown").astype(str)

    # Stratified split preserves the target class proportions.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )),
    ])

    print("Training Random Forest model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "dataset": "synthetic_flights.csv",
        "dataset_is_synthetic": True,
        "total_valid_rows": int(len(X)),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "target": TARGET,
        "features": FEATURES,
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(
            y_test, predictions, zero_division=0
        )), 4),
        "recall": round(float(recall_score(
            y_test, predictions, zero_division=0
        )), 4),
        "f1_score": round(float(f1_score(
            y_test, predictions, zero_division=0
        )), 4),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist(),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\nModel training completed!")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print("\nEvaluation results:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Not Delayed", "Delayed"],
        zero_division=0,
    ))
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(metrics["confusion_matrix"])
    print("\nNote: Results use synthetic data and are not real-world validation.")


if __name__ == "__main__":
    main()
