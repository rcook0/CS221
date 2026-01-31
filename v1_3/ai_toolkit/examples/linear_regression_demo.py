from __future__ import annotations

import numpy as np

from ai_toolkit.optim import (
    batch_gradient_descent,
    make_synthetic_linear_regression,
    stochastic_gradient_descent,
)


def main() -> None:
    X, y, w_true = make_synthetic_linear_regression(n=50_000, d=5, noise_std=1.0, seed=0)

    b = batch_gradient_descent(X, y, eta=0.1, steps=500)
    print("batch_gd  w:", np.round(b.w, 3), "loss:", round(b.loss, 4))

    s = stochastic_gradient_descent(X, y, epochs=5, batch_size=256, eta0=0.5, schedule="inv_sqrt", seed=0)
    print("sgd       w:", np.round(s.w, 3), "loss:", round(s.loss, 4))

    print("w_true    w:", w_true)


if __name__ == "__main__":
    main()
