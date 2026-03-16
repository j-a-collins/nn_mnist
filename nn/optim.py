"""
an optimiser for stochastic gradient descent
"""

# third
import numpy as np


class SGD:
    __slots__ = ("lr",)

    def __init__(self, lr: float = 1e-2) -> None:
        self.lr = np.float32(lr)

    def step(self, parameters: list[tuple[np.ndarray, np.ndarray]]) -> None:
        lr = self.lr
        for param, grad in parameters:
            np.subtract(param, np.multiply(lr, grad, out=grad), out=param)
