"""
creates a fully connected affine layer
"""

# third
import numpy as np


class Dense:
    __slots__ = ("W", "b", "dW", "db", "x", "_params")

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rng: np.random.Generator | None = None,
    ) -> None:
        rng = rng or np.random.default_rng()
        self.W = (
            rng.standard_normal((in_features, out_features))
            * np.float32(np.sqrt(2.0 / in_features))
        ).astype(np.float32)
        self.b = np.zeros(out_features, dtype=np.float32)
        self.dW = np.empty_like(self.W)
        self.db = np.empty_like(self.b)
        self.x: np.ndarray | None = None
        self._params = [(self.W, self.dW), (self.b, self.db)]

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        out = np.empty((x.shape[0], self.W.shape[1]), dtype=np.float32)
        np.dot(x, self.W, out=out)
        np.add(out, self.b, out=out)
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        np.dot(self.x.T, grad_out, out=self.dW)
        np.sum(grad_out, axis=0, out=self.db)
        return np.dot(grad_out, self.W.T)

    def parameters(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return self._params
