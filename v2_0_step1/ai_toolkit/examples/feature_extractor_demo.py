from __future__ import annotations

from collections import defaultdict
from typing import Dict

from ai_toolkit.ml import train_perceptron


def feature_extractor(x: str) -> Dict[str, float]:
    # very small demo: bag-of-words + 2-gram prefix/suffix for middle token(s)
    phi = defaultdict(float)
    tokens = x.split()
    for tok in tokens:
        phi[f"tok={tok}"] += 1.0
        phi[f"pref={tok[:3]}"] += 1.0
        phi[f"suf={tok[-3:]}"] += 1.0
    return phi


def main() -> None:
    # Toy dataset: label +1 if sentence contains "Mauritius" else -1
    train = [
        (+1, "took Mauritius into"),
        (+1, "visit Mauritius soon"),
        (-1, "took Brazil into"),
        (-1, "visit Denmark soon"),
    ]
    dev = [
        (+1, "Mauritius is nice"),
        (-1, "Brazil is big"),
    ]

    res = train_perceptron(train, dev, feature_extractor, iters=10, seed=0)
    print("train_error:", res.train_error)
    print("dev_error:", res.dev_error)
    print("num_features:", len(res.weights))


if __name__ == "__main__":
    main()
