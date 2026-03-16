"""
main train entry point
"""

# third
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# loc
from data.mnist import load_mnist
from nn.activations import ReLU
from nn.checkpoint import save_checkpoint
from nn.layers import Dense
from nn.losses import SoftmaxCrossEntropy
from nn.model import Sequential
from nn.optim import SGD


def iterate_minibatches(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    shuffle: bool = True,
):
    """Yield mini-batches.

    Parameters
    ----------
    x : np.ndarray
        Input array of shape ``(n_samples, n_features)``.
    y : np.ndarray
        Label array of shape ``(n_samples,)``.
    batch_size : int
        Batch size.
    rng : np.random.Generator
        Random number generator.
    shuffle : bool, default=True
        Whether to shuffle samples before batching.

    Yields
    ------
    tuple[np.ndarray, np.ndarray]
        Mini-batch inputs and labels.
    """
    n = x.shape[0]
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)

    for start in range(0, n, batch_size):
        batch_idx = idx[start : start + batch_size]
        yield x[batch_idx], y[batch_idx]


def accuracy(logits: np.ndarray, y: np.ndarray) -> float:
    """Compute classification accuracy.

    Parameters
    ----------
    logits : np.ndarray
        Logits of shape ``(n_samples, n_classes)``.
    y : np.ndarray
        Integer labels of shape ``(n_samples,)``.

    Returns
    -------
    float
        Mean accuracy.
    """
    return float(np.mean(np.argmax(logits, axis=1) == y))


def plot_metric(
    values_a: list[float],
    values_b: list[float],
    label_a: str,
    label_b: str,
    ylabel: str,
    path: str | Path,
) -> None:
    """Plot a train/test metric curve.

    Parameters
    ----------
    values_a : list[float]
        First metric series.
    values_b : list[float]
        Second metric series.
    label_a : str
        Legend label for the first series.
    label_b : str
        Legend label for the second series.
    ylabel : str
        Y-axis label.
    path : str | Path
        Output image path.
    """
    epochs = np.arange(1, len(values_a) + 1)

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, values_a, label=label_a)
    plt.plot(epochs, values_b, label=label_b)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    """Train an MLP on MNIST and save metrics and weights."""
    rng = np.random.default_rng(0)

    x_train, y_train, x_test, y_test = load_mnist()

    model = Sequential(
        [
            Dense(784, 256, rng=rng),
            ReLU(),
            Dense(256, 10, rng=rng),
        ]
    )
    loss_fn = SoftmaxCrossEntropy()
    optim = SGD(lr=0.1)

    batch_size = 128
    epochs = 10

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    for epoch in range(epochs):
        train_loss_sum = 0.0
        train_correct = 0
        train_count = 0

        for xb, yb in iterate_minibatches(x_train, y_train, batch_size, rng):
            logits = model.forward(xb)
            loss = loss_fn.forward(logits, yb)

            grad = loss_fn.backward()
            model.backward(grad)
            optim.step(model.parameters())

            train_loss_sum += loss * xb.shape[0]
            train_correct += np.sum(np.argmax(logits, axis=1) == yb)
            train_count += xb.shape[0]

        train_loss = train_loss_sum / train_count
        train_acc = train_correct / train_count

        test_logits = model.forward(x_test)
        test_loss = loss_fn.forward(test_logits, y_test)
        test_acc = accuracy(test_logits, y_test)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(float(test_loss))
        history["test_acc"].append(test_acc)

        print(
            f"epoch={epoch + 1:02d} "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.4f} "
            f"test_acc={test_acc:.4f}"
        )

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    plot_metric(
        history["train_loss"],
        history["test_loss"],
        "train_loss",
        "test_loss",
        "Loss",
        artifacts_dir / "loss.png",
    )
    plot_metric(
        history["train_acc"],
        history["test_acc"],
        "train_acc",
        "test_acc",
        "Accuracy",
        artifacts_dir / "accuracy.png",
    )
    save_checkpoint(model, artifacts_dir / "mlp_mnist.npz")


if __name__ == "__main__":
    main()
