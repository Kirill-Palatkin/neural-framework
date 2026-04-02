from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .data import Dataset
from .layers import Layer
from .metrics import accuracy_score, binary_accuracy, mean_absolute_error, mean_squared_error


MetricFn = Callable[[np.ndarray, np.ndarray], float]


METRIC_REGISTRY: dict[str, MetricFn] = {
    "accuracy": accuracy_score,
    "binary_accuracy": binary_accuracy,
    "mae": mean_absolute_error,
    "mse": mean_squared_error,
}


@dataclass
class History:
    history: dict[str, list[float]] = field(default_factory=dict)

    def log(self, **values: float) -> None:
        for name, value in values.items():
            self.history.setdefault(name, []).append(float(value))


class Sequential:
    def __init__(self, layers: list[Layer]) -> None:
        if not layers:
            raise ValueError("Sequential model requires at least one layer.")
        self.layers = layers
        self._built = False

    def build(self, input_dim: int) -> None:
        current_dim = input_dim
        for layer in self.layers:
            current_dim = layer.build(current_dim)
        self._built = True

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        output = inputs
        for layer in self.layers:
            output = layer.forward(output, training=training)
        return output

    def backward(self, grad_output: np.ndarray) -> None:
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def _trainable_layers(self) -> list[Layer]:
        return [layer for layer in self.layers if layer.trainable]

    def _ensure_built(self, features: np.ndarray) -> None:
        if self._built:
            return
        if features.ndim == 1:
            input_dim = 1
        else:
            input_dim = features.shape[1]
        self.build(input_dim=input_dim)

    def fit(
        self,
        features: Dataset | np.ndarray,
        targets: np.ndarray | None = None,
        *,
        epochs: int = 100,
        batch_size: int = 32,
        optimizer,
        loss,
        metrics: list[str | MetricFn] | None = None,
        validation_data: Dataset | tuple[np.ndarray, np.ndarray] | None = None,
        shuffle: bool = True,
        seed: int | None = 42,
        verbose: bool = True,
        restore_best_weights: bool = False,
        monitor: str = "val_accuracy",
        early_stopping_patience: int | None = None,
        min_delta: float = 0.0,
    ) -> History:
        if isinstance(features, Dataset):
            train_dataset = features
        else:
            if targets is None:
                raise ValueError("Targets are required when features are passed as numpy arrays.")
            train_dataset = Dataset.from_arrays(features, targets)

        self._ensure_built(train_dataset.features)
        history = History()
        metric_functions = self._resolve_metrics(metrics or [])
        best_value: float | None = None
        best_weights: dict[str, np.ndarray] | None = None
        epochs_without_improvement = 0
        if "loss" in monitor:
            monitor_mode = "min"
        else:
            monitor_mode = "max"

        for epoch in range(1, epochs + 1):
            epoch_dataset = train_dataset.shuffle(seed + epoch if shuffle and seed is not None else None) if shuffle else train_dataset

            losses: list[float] = []
            predictions: list[np.ndarray] = []
            references: list[np.ndarray] = []

            for batch_x, batch_y in epoch_dataset.batch(batch_size=batch_size):
                outputs = self.forward(batch_x, training=True)
                batch_loss = loss.forward(outputs, batch_y)
                grad_loss = loss.backward()
                self.backward(grad_loss)
                optimizer.step(self._trainable_layers())

                losses.append(batch_loss)
                predictions.append(outputs)
                references.append(batch_y)

            train_pred = np.vstack(predictions)
            train_true = np.vstack(references) if references[0].ndim > 1 else np.concatenate(references)
            epoch_logs = {"loss": float(np.mean(losses))}
            epoch_logs.update(self._compute_metrics(metric_functions, train_true, train_pred))

            if validation_data is not None:
                val_logs = self._evaluate_internal(validation_data, loss, metric_functions)
                epoch_logs.update({f"val_{key}": value for key, value in val_logs.items()})

            history.log(**epoch_logs)

            improved = False
            if monitor in epoch_logs:
                current_value = epoch_logs[monitor]
                if best_value is None:
                    improved = True
                elif monitor_mode == "max":
                    improved = current_value > best_value + min_delta
                else:
                    improved = current_value < best_value - min_delta

                if improved:
                    best_value = current_value
                    epochs_without_improvement = 0
                    if restore_best_weights:
                        best_weights = self._snapshot_weights()
                else:
                    epochs_without_improvement += 1

            if verbose:
                print(self._format_epoch(epoch, epochs, epoch_logs))

            if early_stopping_patience is not None and monitor in epoch_logs:
                if epochs_without_improvement >= early_stopping_patience:
                    if verbose:
                        print(
                            f"Early stopping: no improvement in '{monitor}' for {early_stopping_patience} epochs."
                        )
                    break

        if restore_best_weights and best_weights is not None:
            self._restore_weights(best_weights)

        return history

    def evaluate(
        self,
        features: Dataset | np.ndarray,
        targets: np.ndarray | None = None,
        *,
        loss,
        metrics: list[str | MetricFn] | None = None,
    ) -> dict[str, float]:
        metric_functions = self._resolve_metrics(metrics or [])
        return self._evaluate_internal((features, targets) if not isinstance(features, Dataset) else features, loss, metric_functions)

    def predict(self, features: np.ndarray) -> np.ndarray:
        self._ensure_built(features)
        return self.forward(np.asarray(features, dtype=np.float32), training=False)

    def predict_classes(self, features: np.ndarray) -> np.ndarray:
        predictions = self.predict(features)
        if predictions.ndim == 1 or predictions.shape[1] == 1:
            return (predictions.reshape(-1) >= 0.5).astype(int)
        return np.argmax(predictions, axis=1)

    def summary(self) -> str:
        lines = ["Model summary:"]
        for index, layer in enumerate(self.layers, start=1):
            lines.append(f"{index:>2}. {layer.__class__.__name__}")
        return "\n".join(lines)

    def save_weights(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        arrays: dict[str, np.ndarray] = {}
        dense_index = 0
        for layer in self._trainable_layers():
            parameters = layer.parameters()
            for name, parameter in parameters.items():
                arrays[f"layer_{dense_index}_{name}"] = parameter.value
            dense_index += 1

        np.savez(path, **arrays)

    def load_weights(self, path: str | Path) -> None:
        path = Path(path)
        with np.load(path) as stored:
            dense_index = 0
            for layer in self._trainable_layers():
                parameters = layer.parameters()
                if not parameters:
                    continue
                for name, parameter in parameters.items():
                    key = f"layer_{dense_index}_{name}"
                    if key not in stored:
                        raise KeyError(f"Weight '{key}' was not found in {path}.")
                    if stored[key].shape != parameter.value.shape:
                        raise ValueError(
                            f"Shape mismatch for '{key}': expected {parameter.value.shape}, got {stored[key].shape}."
                        )
                    parameter.value[...] = stored[key]
                dense_index += 1

    def _snapshot_weights(self) -> dict[str, np.ndarray]:
        snapshot: dict[str, np.ndarray] = {}
        dense_index = 0
        for layer in self._trainable_layers():
            for name, parameter in layer.parameters().items():
                snapshot[f"layer_{dense_index}_{name}"] = parameter.value.copy()
            dense_index += 1
        return snapshot

    def _restore_weights(self, snapshot: dict[str, np.ndarray]) -> None:
        dense_index = 0
        for layer in self._trainable_layers():
            for name, parameter in layer.parameters().items():
                key = f"layer_{dense_index}_{name}"
                if key in snapshot:
                    parameter.value[...] = snapshot[key]
            dense_index += 1

    def _evaluate_internal(
        self,
        data: Dataset | tuple[np.ndarray, np.ndarray | None],
        loss,
        metric_functions: dict[str, MetricFn],
    ) -> dict[str, float]:
        if isinstance(data, Dataset):
            dataset = data
        else:
            features, targets = data
            if targets is None:
                raise ValueError("Targets are required for evaluation.")
            dataset = Dataset.from_arrays(features, targets)

        predictions = self.forward(dataset.features, training=False)
        logs = {"loss": loss.forward(predictions, dataset.targets)}
        logs.update(self._compute_metrics(metric_functions, dataset.targets, predictions))
        return logs

    def _resolve_metrics(self, metrics: list[str | MetricFn]) -> dict[str, MetricFn]:
        resolved: dict[str, MetricFn] = {}
        for metric in metrics:
            if isinstance(metric, str):
                if metric not in METRIC_REGISTRY:
                    raise KeyError(f"Unknown metric: {metric}")
                resolved[metric] = METRIC_REGISTRY[metric]
            else:
                resolved[metric.__name__] = metric
        return resolved

    def _compute_metrics(
        self,
        metrics: dict[str, MetricFn],
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, float]:
        return {name: fn(y_true, y_pred) for name, fn in metrics.items()}

    @staticmethod
    def _format_epoch(epoch: int, total_epochs: int, logs: dict[str, float]) -> str:
        metrics_line = " | ".join(f"{name}={value:.4f}" for name, value in logs.items())
        return f"Epoch {epoch:03d}/{total_epochs:03d} | {metrics_line}"
