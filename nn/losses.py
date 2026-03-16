"""
module for softmax and mean cross-entropy loss
"""

# third
import numpy as np


class SoftmaxCrossEntropy:
    __slots__ = ("probs", "y", "_n")

    def __init__(self) -> None:
        self.probs: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self._n: int = 0

    def forward(self, logits: np.ndarray, y: np.ndarray) -> float:
        n = logits.shape[0]
        self._n = n
        self.y = y
        shifted = logits - logits.max(axis=1, keepdims=True)
        np.exp(shifted, out=shifted)
        shifted /= shifted.sum(axis=1, keepdims=True)
        self.probs = shifted
        arange_n = np.arange(n)
        return float(-np.log(shifted[arange_n, y] + 1e-12).sum() / n)

    def backward(self) -> np.ndarray:
        grad = self.probs.copy()
        grad[np.arange(self._n), self.y] -= 1.0
        grad *= 1.0 / self._n
        return grad
