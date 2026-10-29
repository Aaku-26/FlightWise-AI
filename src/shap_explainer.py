
import numpy as np
import pandas as pd
import shap


def explain_flight_prediction(model, flight_features):
    """
    Explain a flight-delay prediction using SHAP.

    model: Trained scikit-learn Pipeline
    flight_features: DataFrame containing the 10 original input features
    """

    try:
        # Get preprocessing and trained classifier
        preprocessing = model.named_steps["preprocessing"]
        classifier = model.named_steps["classifier"]

        # Transform the original flight features
        transformed_features = preprocessing.transform(flight_features)

        # Get transformed feature names
        feature_names = preprocessing.get_feature_names_out()

        # Convert sparse output to a dense array if necessary
        if hasattr(transformed_features, "toarray"):
            transformed_features = transformed_features.toarray()

        transformed_features = np.asarray(transformed_features)

        # Create SHAP explainer for the trained classifier
        explainer = shap.TreeExplainer(classifier)

        # Calculate SHAP values for this flight
        shap_values = explainer.shap_values(transformed_features)

        # Handle binary classification output
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        shap_values = np.asarray(shap_values)

        # Extract contributions for this flight
        values = shap_values[0]

        # Sort the strongest contributions by absolute magnitude
        top_indices = np.argsort(np.abs(values))[::-1][:10]

        results = []

        for index in top_indices:
            results.append({
                "feature": str(feature_names[index]),
                "shap_value": float(values[index]),
                "impact": (
                    "Increases delay risk"
                    if values[index] > 0
                    else "Decreases delay risk"
                    if values[index] < 0
                    else "No impact"
                )
            })

        return results

    except Exception as error:
        print(f"SHAP explanation error: {error}")
        return []
