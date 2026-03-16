"""
downloader module for the mnist images
"""

from __future__ import annotations

# lib
import gzip
import struct
import urllib.request
from pathlib import Path

# third
import numpy as np

BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
_NORM_SCALE = np.float32(1.0 / 255.0)


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dst)


def download_mnist(
    data_dir: str | Path = "data/raw", overwrite: bool = False
) -> dict[str, Path]:
    data_dir = Path(data_dir)
    paths: dict[str, Path] = {}
    for key, filename in FILES.items():
        path = data_dir / filename
        if overwrite or not path.exists():
            _download(BASE_URL + filename, path)
        paths[key] = path
    return paths


def _read_idx_images(path: str | Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        buf = f.read()
    magic, n, rows, cols = struct.unpack_from(">IIII", buf, 0)
    if magic != 2051:
        raise ValueError(f"unexpected image magic number: {magic}")
    return np.frombuffer(buf, dtype=np.uint8, offset=16).reshape(n, rows, cols)


def _read_idx_labels(path: str | Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        buf = f.read()
    magic, n = struct.unpack_from(">II", buf, 0)
    if magic != 2049:
        raise ValueError(f"unexpected label magic number: {magic}")
    return np.frombuffer(buf, dtype=np.uint8, offset=8).reshape(n)


def _process_images(imgs: np.ndarray, normalise: bool, flatten: bool) -> np.ndarray:
    if flatten:
        imgs = imgs.reshape(imgs.shape[0], -1)
    if normalise:
        return np.multiply(imgs, _NORM_SCALE, dtype=np.float32)
    return imgs


def load_mnist(
    data_dir: str | Path = "data/raw",
    normalise: bool = True,
    flatten: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    paths = download_mnist(data_dir=data_dir, overwrite=False)
    x_train = _process_images(
        _read_idx_images(paths["train_images"]), normalise, flatten
    )
    x_test = _process_images(_read_idx_images(paths["test_images"]), normalise, flatten)
    y_train = _read_idx_labels(paths["train_labels"]).astype(np.int64)
    y_test = _read_idx_labels(paths["test_labels"]).astype(np.int64)
    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_mnist()

    print("x_train:", x_train.shape, x_train.dtype)
    print("y_train:", y_train.shape, y_train.dtype)
    print("x_test:", x_test.shape, x_test.dtype)
    print("y_test:", y_test.shape, y_test.dtype)
    print("x_train range:", x_train.min(), x_train.max())
