from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def _he_uniform(shape: tuple[int, int]) -> np.ndarray:
    fan_in = shape[0]
    limit = np.sqrt(6.0 / fan_in)
    return np.random.uniform(-limit, limit, size=shape).astype(np.float32)


def _xavier_uniform(shape: tuple[int, int]) -> np.ndarray:
    fan_in, fan_out = shape
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=shape).astype(np.float32)


@dataclass
class Parameter:
    value: np.ndarray
    grad: np.ndarray


class Layer:
    trainable = False

    def build(self, input_dim: int) -> int:
        return input_dim

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def parameters(self) -> dict[str, Parameter]:
        return {}


class Dense(Layer):
    trainable = True

    def __init__(
        self,
        units: int,
        input_dim: Optional[int] = None,
        initializer: str = "xavier",
    ) -> None:
        self.units = units
        self.input_dim = input_dim
        self.initializer = initializer
        self.weights: Optional[np.ndarray] = None
        self.biases: Optional[np.ndarray] = None
        self.weight_grad: Optional[np.ndarray] = None
        self.bias_grad: Optional[np.ndarray] = None
        self._inputs: Optional[np.ndarray] = None

    def build(self, input_dim: int) -> int:
        actual_input_dim = self.input_dim or input_dim
        shape = (actual_input_dim, self.units)

        if self.initializer == "he":
            self.weights = _he_uniform(shape)
        else:
            self.weights = _xavier_uniform(shape)

        self.biases = np.zeros((1, self.units), dtype=np.float32)
        self.weight_grad = np.zeros_like(self.weights)
        self.bias_grad = np.zeros_like(self.biases)
        self.input_dim = actual_input_dim
        return self.units

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        if self.weights is None or self.biases is None:
            raise RuntimeError("Dense layer must be built before forward pass.")
        self._inputs = inputs
        return inputs @ self.weights + self.biases

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._inputs is None:
            raise RuntimeError("Dense layer has no cached inputs for backpropagation.")
        if self.weight_grad is None or self.bias_grad is None or self.weights is None:
            raise RuntimeError("Dense layer gradients are not initialized.")

        self.weight_grad[...] = self._inputs.T @ grad_output
        self.bias_grad[...] = np.sum(grad_output, axis=0, keepdims=True)
        return grad_output @ self.weights.T

    def parameters(self) -> dict[str, Parameter]:
        if self.weights is None or self.biases is None:
            return {}
        if self.weight_grad is None or self.bias_grad is None:
            raise RuntimeError("Dense layer gradients are not initialized.")
        return {
            "weights": Parameter(self.weights, self.weight_grad),
            "biases": Parameter(self.biases, self.bias_grad),
        }
