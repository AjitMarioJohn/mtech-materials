from __future__ import annotations

import numpy as np


class MLPScratch:
    """Fully-connected MLP for binary classification with ReLU hidden layers."""

    def __init__(
        self,
        architecture: list[int],
        learning_rate: float = 0.01,
        epochs: int = 300,
        random_state: int = 42,
    ) -> None:
        if len(architecture) < 3:
            raise ValueError("Architecture must include input, at least one hidden, and output layer.")
        if architecture[-1] != 1:
            raise ValueError("For binary classification, output layer size must be 1.")

        self.architecture = architecture
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state

        self.parameters: dict[str, np.ndarray] = {}
        self.loss_history: list[float] = []

        # Cached activations/linear outputs from the latest forward pass.
        self._activations: list[np.ndarray] = []
        self._linear_outputs: list[np.ndarray] = []

        self.initialize_parameters()

    def initialize_parameters(self) -> None:
        """Initialize W and b for each layer using He initialization."""
        rng = np.random.default_rng(self.random_state)
        self.parameters = {}

        for layer_idx in range(1, len(self.architecture)):
            fan_in = self.architecture[layer_idx - 1]
            fan_out = self.architecture[layer_idx]

            self.parameters[f"W{layer_idx}"] = (
                rng.normal(size=(fan_in, fan_out)) * np.sqrt(2.0 / fan_in)
            ).astype(np.float64)
            self.parameters[f"b{layer_idx}"] = np.zeros((1, fan_out), dtype=np.float64)

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    @staticmethod
    def _relu_derivative(x: np.ndarray) -> np.ndarray:
        return (x > 0.0).astype(np.float64)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        eps = 1e-12
        y_prob = np.clip(y_prob, eps, 1.0 - eps)
        return float(-np.mean(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob)))

    def forward_propagation(self, X: np.ndarray) -> np.ndarray:
        """Compute activations through all layers and return output probabilities."""
        A = np.asarray(X, dtype=np.float64)
        self._activations = [A]
        self._linear_outputs = []

        num_layers = len(self.architecture) - 1
        for layer_idx in range(1, num_layers + 1):
            W = self.parameters[f"W{layer_idx}"]
            b = self.parameters[f"b{layer_idx}"]

            Z = A @ W + b
            self._linear_outputs.append(Z)

            if layer_idx < num_layers:
                A = self._relu(Z)
            else:
                A = self._sigmoid(Z)

            self._activations.append(A)

        return A

    def backward_propagation(self, y_true: np.ndarray) -> dict[str, np.ndarray]:
        """Compute gradients with chain rule for all parameters."""
        y = np.asarray(y_true, dtype=np.float64).reshape(-1, 1)
        m = y.shape[0]

        num_layers = len(self.architecture) - 1
        gradients: dict[str, np.ndarray] = {}

        A_last = self._activations[-1]
        dZ = A_last - y

        for layer_idx in range(num_layers, 0, -1):
            A_prev = self._activations[layer_idx - 1]
            W = self.parameters[f"W{layer_idx}"]

            dW = (A_prev.T @ dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m

            gradients[f"dW{layer_idx}"] = dW
            gradients[f"db{layer_idx}"] = db

            if layer_idx > 1:
                Z_prev = self._linear_outputs[layer_idx - 2]
                dA_prev = dZ @ W.T
                dZ = dA_prev * self._relu_derivative(Z_prev)

        return gradients

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPScratch":
        X_train = np.asarray(X, dtype=np.float64)
        y_train = np.asarray(y, dtype=np.float64).reshape(-1, 1)

        if X_train.shape[1] != self.architecture[0]:
            raise ValueError(
                f"Input feature count mismatch. Expected {self.architecture[0]}, got {X_train.shape[1]}."
            )

        self.loss_history = []

        for _ in range(self.epochs):
            y_prob = self.forward_propagation(X_train)
            loss = self._binary_cross_entropy(y_train, y_prob)
            self.loss_history.append(loss)

            gradients = self.backward_propagation(y_train)

            for layer_idx in range(1, len(self.architecture)):
                self.parameters[f"W{layer_idx}"] -= self.learning_rate * gradients[f"dW{layer_idx}"]
                self.parameters[f"b{layer_idx}"] -= self.learning_rate * gradients[f"db{layer_idx}"]

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        y_prob = self.forward_propagation(np.asarray(X, dtype=np.float64))
        return y_prob.reshape(-1)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(np.int64)

