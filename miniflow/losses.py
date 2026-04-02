from __future__ import annotations

import numpy as np


class Loss:
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        raise NotImplementedError

    def backward(self) -> np.ndarray:
        raise NotImplementedError


class MSELoss(Loss):
    def __init__(self) -> None:
        self._y_pred: np.ndarray | None = None
        self._y_true: np.ndarray | None = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        aligned_true = y_true.reshape(y_pred.shape) if y_true.shape != y_pred.shape else y_true
        self._y_pred = y_pred
        self._y_true = aligned_true
        return float(np.mean((y_pred - aligned_true) ** 2))

    def backward(self) -> np.ndarray:
        if self._y_pred is None or self._y_true is None:
            raise RuntimeError("MSELoss has no cached tensors.")
        return 2.0 * (self._y_pred - self._y_true) / np.prod(self._y_true.shape)


class BinaryCrossEntropy(Loss):
    def __init__(self) -> None:
        self._y_pred: np.ndarray | None = None
        self._y_true: np.ndarray | None = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        clipped = np.clip(y_pred, 1e-7, 1.0 - 1e-7)
        self._y_pred = clipped
        self._y_true = y_true.reshape(clipped.shape)
        loss = -np.mean(self._y_true * np.log(clipped) + (1.0 - self._y_true) * np.log(1.0 - clipped))
        return float(loss)

    def backward(self) -> np.ndarray:
        if self._y_pred is None or self._y_true is None:
            raise RuntimeError("BinaryCrossEntropy has no cached tensors.")
        numerator = self._y_pred - self._y_true
        denominator = self._y_pred * (1.0 - self._y_pred)
        return numerator / (denominator * self._y_true.shape[0])


class CategoricalCrossEntropy(Loss):
    def __init__(self) -> None:
        self._y_pred: np.ndarray | None = None
        self._y_true: np.ndarray | None = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        clipped = np.clip(y_pred, 1e-7, 1.0 - 1e-7)
        if y_true.ndim == 1:
            encoded = np.eye(clipped.shape[1], dtype=np.float32)[y_true.astype(int)]
        else:
            encoded = y_true
        self._y_pred = clipped
        self._y_true = encoded
        loss = -np.sum(encoded * np.log(clipped), axis=1)
        return float(np.mean(loss))

    def backward(self) -> np.ndarray:
        if self._y_pred is None or self._y_true is None:
            raise RuntimeError("CategoricalCrossEntropy has no cached tensors.")
        return -(self._y_true / self._y_pred) / self._y_true.shape[0]
