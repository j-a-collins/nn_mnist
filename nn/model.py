"""
for creating a sequential container for nn layers
"""

# third
import numpy as np


class Sequential:
    __slots__ = ("layers", "_reversed", "_params")

    def __init__(self, layers: list) -> None:
        self.layers = layers
        self._reversed = layers[::-1]
        self._params = [p for layer in layers for p in layer.parameters()]

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        for layer in self._reversed:
            grad_out = layer.backward(grad_out)
        return grad_out

    def parameters(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return self._params
