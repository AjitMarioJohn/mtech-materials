from __future__ import annotations

import numpy as np


class LogisticRegressionScratch:
    """Binary logistic regression trained with batch gradient descent."""

    def __init__(self, learning_rate: float = 0.1, epochs: int = 1000) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.loss_history: list[float] = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        # Clip to avoid overflow in exp for very large magnitude values.
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def _binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        eps = 1e-12
        y_prob = np.clip(y_prob, eps, 1.0 - eps)
        return float(-np.mean(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionScratch":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)

        n_samples, n_features = X.shape
        self.w = np.zeros((n_features, 1), dtype=np.float64)
        self.b = 0.0
        self.loss_history = []

        for _ in range(self.epochs):
            z = X @ self.w + self.b
            y_prob = self._sigmoid(z)

            loss = self._binary_cross_entropy(y, y_prob)
            self.loss_history.append(loss)

            dz = y_prob - y
            dw = (X.T @ dz) / n_samples
            db = float(np.sum(dz) / n_samples)

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise ValueError("Model is not trained. Call fit() before predict_proba().")

        X = np.asarray(X, dtype=np.float64)
        z = X @ self.w + self.b
        return self._sigmoid(z).reshape(-1)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        y_prob = self.predict_proba(X)
        return (y_prob >= threshold).astype(np.int64)

