from __future__ import annotations

import numpy as np


def _ensure_2d(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    return y_true, y_pred


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        predicted = np.argmax(y_pred, axis=1)
        target = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true
    else:
        predicted = (y_pred.reshape(-1) >= 0.5).astype(int)
        target = y_true.reshape(-1).astype(int)
    return float(np.mean(predicted == target))


def binary_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return accuracy_score(y_true, y_pred)


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _ensure_2d(y_true, y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _ensure_2d(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))
