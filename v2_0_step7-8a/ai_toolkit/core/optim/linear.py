from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class OptimResult:
    w: np.ndarray
    loss: float
    steps: int


def mse_loss_and_grad(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> Tuple[float, np.ndarray]:
    r = X @ w - y
    loss = float(np.mean(r * r))
    grad = (2.0 / X.shape[0]) * (X.T @ r)
    return loss, grad


def batch_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    *,
    eta: float = 0.1,
    steps: int = 500,
    w0: Optional[np.ndarray] = None,
) -> OptimResult:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    d = X.shape[1]
    w = np.zeros(d, dtype=float) if w0 is None else np.asarray(w0, dtype=float).reshape(d)

    for _ in range(steps):
        _loss, grad = mse_loss_and_grad(X, y, w)
        w = w - eta * grad
    loss, _ = mse_loss_and_grad(X, y, w)
    return OptimResult(w=w, loss=loss, steps=steps)


def stochastic_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int = 10,
    batch_size: int = 64,
    eta0: float = 1.0,
    schedule: str = "inv_sqrt",
    shuffle: bool = True,
    seed: int = 0,
    w0: Optional[np.ndarray] = None,
) -> OptimResult:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    n, d = X.shape
    w = np.zeros(d, dtype=float) if w0 is None else np.asarray(w0, dtype=float).reshape(d)

    rng = np.random.default_rng(seed)
    steps = 0
    for _epoch in range(epochs):
        idx = np.arange(n)
        if shuffle:
            rng.shuffle(idx)
        for i in range(0, n, batch_size):
            steps += 1
            if schedule == "inv":
                eta = eta0 / steps
            elif schedule == "inv_sqrt":
                eta = eta0 / np.sqrt(steps)
            elif schedule == "constant":
                eta = eta0
            else:
                raise ValueError(f"Unknown schedule: {schedule}")

            batch = idx[i:i + batch_size]
            _loss, grad = mse_loss_and_grad(X[batch], y[batch], w)
            w = w - eta * grad

    loss, _ = mse_loss_and_grad(X, y, w)
    return OptimResult(w=w, loss=loss, steps=steps)


def make_synthetic_linear_regression(
    *,
    n: int = 50_000,
    d: int = 5,
    noise_std: float = 1.0,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    w_true = np.arange(1, d + 1, dtype=float)
    X = rng.standard_normal((n, d))
    y = X @ w_true + rng.normal(0.0, noise_std, size=n)
    return X, y, w_true
