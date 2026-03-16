from __future__ import annotations

# lib
from pathlib import Path

# third
import numpy as np

# loc
from nn.layers import Dense
from nn.model import Sequential


def save_checkpoint(model: Sequential, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    idx = 0
    for layer in model.layers:
        if isinstance(layer, Dense):
            arrays[f"W{idx}"] = layer.W
            arrays[f"b{idx}"] = layer.b
            idx += 1
    np.savez(path, **arrays)


def load_checkpoint(model: Sequential, path: str | Path) -> None:
    with np.load(path) as checkpoint:
        idx = 0
        for layer in model.layers:
            if isinstance(layer, Dense):
                np.copyto(layer.W, checkpoint[f"W{idx}"])
                np.copyto(layer.b, checkpoint[f"b{idx}"])
                idx += 1
