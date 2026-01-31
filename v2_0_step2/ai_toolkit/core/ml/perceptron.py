from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, DefaultDict, Dict, Iterable, List, Sequence, Tuple

SparseVec = Dict[str, float]


def dot(a: SparseVec, b: SparseVec) -> float:
    # iterate over smaller dict
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())


def add_scaled_inplace(w: SparseVec, phi: SparseVec, scale: float) -> None:
    for k, v in phi.items():
        w[k] = w.get(k, 0.0) + scale * v


def read_labeled_text(path: str) -> List[Tuple[int, str]]:
    """Read lines like:  <label>\t<text>  (also accepts space-separated)."""
    out: List[Tuple[int, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "\t" in line:
                y_str, x = line.split("\t", 1)
            else:
                y_str, x = line.split(" ", 1)
            out.append((int(y_str), x))
    return out


def evaluate(examples: Sequence[Tuple[int, str]], predictor: Callable[[str], int]) -> float:
    wrong = 0
    for y, x in examples:
        if predictor(x) != y:
            wrong += 1
    return wrong / max(1, len(examples))


@dataclass(frozen=True)
class PerceptronResult:
    weights: SparseVec
    train_error: float
    dev_error: float
    iters: int


def train_perceptron(
    train: Sequence[Tuple[int, str]],
    dev: Sequence[Tuple[int, str]],
    feature_extractor: Callable[[str], SparseVec],
    *,
    iters: int = 20,
    seed: int = 0,
) -> PerceptronResult:
    """Binary perceptron on sparse feature dicts. Labels are expected in {+1, -1}."""
    rng = __import__("random").Random(seed)
    w: SparseVec = {}

    def predict(x: str) -> int:
        return 1 if dot(feature_extractor(x), w) > 0 else -1

    best_w = dict(w)
    best_dev = float("inf")
    last_train_err = 1.0
    last_dev_err = 1.0

    idx = list(range(len(train)))
    for t in range(1, iters + 1):
        rng.shuffle(idx)
        for i in idx:
            y, x = train[i]
            phi = feature_extractor(x)
            score = dot(phi, w)
            if float(y) * score <= 0.0:
                add_scaled_inplace(w, phi, float(y))
        last_train_err = evaluate(train, predict)
        last_dev_err = evaluate(dev, predict)
        if last_dev_err < best_dev:
            best_dev = last_dev_err
            best_w = dict(w)

    return PerceptronResult(weights=best_w, train_error=last_train_err, dev_error=last_dev_err, iters=iters)
