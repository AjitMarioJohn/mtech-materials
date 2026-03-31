from __future__ import annotations

import json
import time
from typing import Any

import requests

from src.data_pipeline import load_adult_income_data
from src.logistic_regression import LogisticRegressionScratch
from src.metrics import binary_classification_metrics
from src.mlp import MLPScratch


def load_assignment_results_from_url(url: str) -> dict[str, Any]:
    """Load assignment results from a remote URL (e.g., hosted JSON file).
    
    Args:
        url: Full URL to the JSON file (e.g., https://example.com/data/results.json)
        
    Returns:
        Dictionary with assignment results
        
    Raises:
        requests.RequestException: If URL fetch fails
        json.JSONDecodeError: If response is not valid JSON
    """
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def get_assignment_results() -> dict[str, Any]:
    """Train baseline + MLP models and return assignment-ready result fields."""
    data = load_adult_income_data(test_size=0.2, random_state=42)

    baseline_model = LogisticRegressionScratch(learning_rate=0.1, epochs=800)
    baseline_start = time.perf_counter()
    baseline_model.fit(data.X_train, data.y_train)
    baseline_train_time = time.perf_counter() - baseline_start

    baseline_train_pred = baseline_model.predict(data.X_train)
    baseline_test_pred = baseline_model.predict(data.X_test)
    baseline_train_metrics = binary_classification_metrics(data.y_train, baseline_train_pred)
    baseline_test_metrics = binary_classification_metrics(data.y_test, baseline_test_pred)

    mlp_architecture = [data.X_train.shape[1], 128, 64, 1]
    mlp_model = MLPScratch(
        architecture=mlp_architecture,
        learning_rate=0.01,
        epochs=700,
        random_state=42,
    )
    mlp_start = time.perf_counter()
    mlp_model.fit(data.X_train, data.y_train)
    mlp_train_time = time.perf_counter() - mlp_start

    mlp_train_pred = mlp_model.predict(data.X_train)
    mlp_test_pred = mlp_model.predict(data.X_test)
    mlp_train_metrics = binary_classification_metrics(data.y_train, mlp_train_pred)
    mlp_test_metrics = binary_classification_metrics(data.y_test, mlp_test_pred)

    return {
        "dataset_name": data.dataset_name,
        "n_samples": data.n_samples,
        "n_features": data.n_features,
        "problem_type": "binary_classification",
        "primary_metric": "f1",
        "baseline_model": {
            "name": "LogisticRegressionScratch",
            "training_time_seconds": baseline_train_time,
            "train_metrics": baseline_train_metrics,
            "test_metrics": baseline_test_metrics,
            "loss_history": baseline_model.loss_history,
        },
        "mlp_model": {
            "name": "MLPScratch",
            "architecture": mlp_architecture,
            "training_time_seconds": mlp_train_time,
            "train_metrics": mlp_train_metrics,
            "test_metrics": mlp_test_metrics,
            "loss_history": mlp_model.loss_history,
        },
    }

