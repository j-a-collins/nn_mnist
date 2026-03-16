"""
creates the rectified linear activation function
"""

# third
import numpy as np


class ReLU:
    __slots__ = ("mask",)
    _empty_params: list = []

    def __init__(self) -> None:
        self.mask: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = mask = np.greater(x, 0)
        return np.multiply(x, mask)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        return np.multiply(grad_out, self.mask)

    def parameters(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return ReLU._empty_params
