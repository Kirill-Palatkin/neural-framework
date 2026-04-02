from __future__ import annotations

import numpy as np

from .layers import Layer


class Activation(Layer):
    def __init__(self) -> None:
        self._inputs: np.ndarray | None = None
        self._outputs: np.ndarray | None = None
        self._mask: np.ndarray | None = None


class Identity(Activation):
    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        self._inputs = inputs
        self._outputs = inputs
        return inputs

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output


class ReLU(Activation):
    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        self._inputs = inputs
        self._outputs = np.maximum(0.0, inputs)
        return self._outputs

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._inputs is None:
            raise RuntimeError("ReLU has no cached inputs for backpropagation.")
        return grad_output * (self._inputs > 0.0)


class LeakyReLU(Activation):
    def __init__(self, negative_slope: float = 0.01) -> None:
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        self._inputs = inputs
        self._outputs = np.where(inputs > 0.0, inputs, self.negative_slope * inputs)
        return self._outputs

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._inputs is None:
            raise RuntimeError("LeakyReLU has no cached inputs for backpropagation.")
        local_grad = np.where(self._inputs > 0.0, 1.0, self.negative_slope)
        return grad_output * local_grad


class Sigmoid(Activation):
    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        self._inputs = inputs
        self._outputs = 1.0 / (1.0 + np.exp(-inputs))
        return self._outputs

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._outputs is None:
            raise RuntimeError("Sigmoid has no cached outputs for backpropagation.")
        return grad_output * self._outputs * (1.0 - self._outputs)


class Tanh(Activation):
    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        self._inputs = inputs
        self._outputs = np.tanh(inputs)
        return self._outputs

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._outputs is None:
            raise RuntimeError("Tanh has no cached outputs for backpropagation.")
        return grad_output * (1.0 - self._outputs**2)


class Softmax(Activation):
    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        shifted = inputs - np.max(inputs, axis=1, keepdims=True)
        exps = np.exp(shifted)
        self._outputs = exps / np.sum(exps, axis=1, keepdims=True)
        return self._outputs

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._outputs is None:
            raise RuntimeError("Softmax has no cached outputs for backpropagation.")

        grad_inputs = np.empty_like(grad_output)
        for index, (sample_output, sample_grad) in enumerate(zip(self._outputs, grad_output)):
            column = sample_output.reshape(-1, 1)
            jacobian = np.diagflat(column) - column @ column.T
            grad_inputs[index] = jacobian @ sample_grad
        return grad_inputs


class Dropout(Activation):
    def __init__(self, rate: float = 0.2) -> None:
        super().__init__()
        if not 0.0 <= rate < 1.0:
            raise ValueError("Dropout rate must be in the range [0, 1).")
        self.rate = rate

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        if not training or self.rate == 0.0:
            self._mask = None
            self._outputs = inputs
            return inputs

        keep_probability = 1.0 - self.rate
        self._mask = (np.random.rand(*inputs.shape) < keep_probability).astype(np.float32) / keep_probability
        self._outputs = inputs * self._mask
        return self._outputs

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._mask is None:
            return grad_output
        return grad_output * self._mask
