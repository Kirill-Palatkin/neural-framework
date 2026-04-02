from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .layers import Layer


@dataclass
class GradientClipping:
    max_norm: float

    def apply(self, layers: list[Layer]) -> float:
        squared_norm = 0.0
        for layer in layers:
            for parameter in layer.parameters().values():
                squared_norm += float(np.sum(parameter.grad**2))

        norm = squared_norm**0.5
        if norm == 0.0 or norm <= self.max_norm:
            return norm

        scale = self.max_norm / (norm + 1e-12)
        for layer in layers:
            for parameter in layer.parameters().values():
                parameter.grad[...] *= scale
        return norm


class Optimizer:
    def __init__(self, learning_rate: float = 0.01, gradient_clipping: GradientClipping | None = None) -> None:
        self.learning_rate = learning_rate
        self.gradient_clipping = gradient_clipping

    def step(self, layers: list[Layer]) -> None:
        raise NotImplementedError

    def _prepare(self, layers: list[Layer]) -> None:
        if self.gradient_clipping is not None:
            self.gradient_clipping.apply(layers)


class SGD(Optimizer):
    def step(self, layers: list[Layer]) -> None:
        self._prepare(layers)
        for layer in layers:
            for parameter in layer.parameters().values():
                parameter.value[...] -= self.learning_rate * parameter.grad


class MomentumSGD(Optimizer):
    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        gradient_clipping: GradientClipping | None = None,
    ) -> None:
        super().__init__(learning_rate=learning_rate, gradient_clipping=gradient_clipping)
        self.momentum = momentum
        self.velocity: dict[tuple[int, str], np.ndarray] = {}

    def step(self, layers: list[Layer]) -> None:
        self._prepare(layers)
        for layer in layers:
            for name, parameter in layer.parameters().items():
                key = (id(layer), name)
                velocity = self.velocity.setdefault(key, np.zeros_like(parameter.value))
                velocity[...] = self.momentum * velocity - self.learning_rate * parameter.grad
                parameter.value[...] += velocity


class Adam(Optimizer):
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        gradient_clipping: GradientClipping | None = None,
    ) -> None:
        super().__init__(learning_rate=learning_rate, gradient_clipping=gradient_clipping)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.moments: dict[tuple[int, str], np.ndarray] = {}
        self.velocities: dict[tuple[int, str], np.ndarray] = {}
        self.timestep = 0

    def step(self, layers: list[Layer]) -> None:
        self._prepare(layers)
        self.timestep += 1

        for layer in layers:
            for name, parameter in layer.parameters().items():
                key = (id(layer), name)
                first_moment = self.moments.setdefault(key, np.zeros_like(parameter.value))
                second_moment = self.velocities.setdefault(key, np.zeros_like(parameter.value))

                first_moment[...] = self.beta1 * first_moment + (1.0 - self.beta1) * parameter.grad
                second_moment[...] = self.beta2 * second_moment + (1.0 - self.beta2) * (parameter.grad**2)

                first_hat = first_moment / (1.0 - self.beta1**self.timestep)
                second_hat = second_moment / (1.0 - self.beta2**self.timestep)
                parameter.value[...] -= self.learning_rate * first_hat / (np.sqrt(second_hat) + self.epsilon)
