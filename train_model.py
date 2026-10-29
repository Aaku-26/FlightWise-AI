
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
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


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "synthetic_flights.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "flight_delay_model.joblib"
METRICS_PATH = MODEL_DIR / "model_metrics.json"


# --------------------------------------------------
# 2. Features and target
# --------------------------------------------------

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


# --------------------------------------------------
# 3. Build preprocessing pipeline
# --------------------------------------------------

def create_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


# --------------------------------------------------
# 4. Evaluate a model
# --------------------------------------------------

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": round(
            float(accuracy_score(y_test, predictions)), 4
        ),
        "precision": round(
            float(precision_score(
                y_test, predictions, zero_division=0
            )), 4
        ),
        "recall": round(
            float(recall_score(
                y_test, predictions, zero_division=0
            )), 4
        ),
        "f1_score": round(
            float(f1_score(
                y_test, predictions, zero_division=0
            )), 4
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist(),
    }

    print(f"\n{name} results:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Not Delayed", "Delayed"],
        zero_division=0,
    ))
    print("Confusion matrix:")
    print(metrics["confusion_matrix"])

    return {
        "model": model,
        "metrics": metrics,
    }


# --------------------------------------------------
# 5. Main training process
# --------------------------------------------------

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
            f"Missing columns in dataset: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    # Convert target values to 0 and 1.
    if df[TARGET].dtype == object:
        target_values = (
            df[TARGET].astype(str).str.strip().str.lower()
        )

        target_map = {
            "true": 1,
            "false": 0,
            "1": 1,
            "0": 0,
            "yes": 1,
            "no": 0,
        }

        y = target_values.map(target_map)
    else:
        y = pd.to_numeric(df[TARGET], errors="coerce")

    # Remove rows with invalid target values.
    valid = y.isin([0, 1])
    X = df.loc[valid, FEATURES].copy()
    y = y.loc[valid].astype(int)

    if len(X) < 20:
        raise ValueError("Not enough valid rows to train the models.")

    if y.nunique() < 2:
        raise ValueError(
            "The target must contain delayed and non-delayed flights."
        )

    # Convert numeric features safely.
    for col in NUMERIC_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Handle missing categorical values in the pipeline.
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].replace(
            r"^\s*$", pd.NA, regex=True
        )

    # Check whether stratified splitting is possible.
    class_counts = y.value_counts()

    if class_counts.min() < 2:
        raise ValueError(
            "Each target class needs at least two rows "
            "for a stratified train/test split."
        )

    # Split once so both models use identical test data.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("=" * 60)
    print("FLIGHTWISE AI - MODEL COMPARISON")
    print("=" * 60)
    print(f"Dataset: {DATA_PATH.name}")
    print("Dataset type: Synthetic demonstration data")
    print(f"Valid rows: {len(X)}")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print(f"Features: {len(FEATURES)}")

    # --------------------------------------------------
    # 6. Define both models
    # --------------------------------------------------

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
        ),
    }

    results = {}

    for name, classifier in models.items():
        pipeline = Pipeline([
            ("preprocessing", create_preprocessor()),
            ("classifier", classifier),
        ])

        results[name] = evaluate_model(
            name,
            pipeline,
            X_train,
            X_test,
            y_train,
            y_test,
        )

    # --------------------------------------------------
    # 7. Select the best model
    # F1-score first, recall second.
    # --------------------------------------------------

    best_name = max(
        results,
        key=lambda name: (
            results[name]["metrics"]["f1_score"],
            results[name]["metrics"]["recall"],
        ),
    )

    best_model = results[best_name]["model"]

    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)

    for name, result in results.items():
        m = result["metrics"]

        print(
            f"{name:20s} | "
            f"Accuracy: {m['accuracy']:.4f} | "
            f"Precision: {m['precision']:.4f} | "
            f"Recall: {m['recall']:.4f} | "
            f"F1-score: {m['f1_score']:.4f}"
        )

    print(f"\nSelected model: {best_name}")

    # --------------------------------------------------
    # 8. Save selected model and evaluation metrics
    # --------------------------------------------------

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, MODEL_PATH)

    metrics_output = {
        "dataset": DATA_PATH.name,
        "dataset_is_synthetic": True,
        "total_valid_rows": int(len(X)),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "target": TARGET,
        "features": FEATURES,
        "selection_metric": "F1-score, then recall",
        "selected_model": best_name,
        "models": {
            name: result["metrics"]
            for name, result in results.items()
        },
        # Keep these top-level metrics for compatibility
        # with code that expects the original JSON format.
        **results[best_name]["metrics"],
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics_output, file, indent=4)

    print("\nTraining completed successfully!")
    print(f"Selected model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")

    print(
        "\nIMPORTANT: Evaluation uses synthetic data. "
        "These results do not establish real-world aviation accuracy."
    )


if __name__ == "__main__":
    main()
