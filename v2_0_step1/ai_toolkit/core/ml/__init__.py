"""Basic ML utilities (feature extraction + perceptron).

Note: this is intentionally minimal and mirrors the original CS221-style utilities.
"""

from .perceptron import SparseVec, PerceptronResult, add_scaled_inplace, dot, evaluate, read_labeled_text, train_perceptron

__all__ = [
    "SparseVec",
    "PerceptronResult",
    "dot",
    "add_scaled_inplace",
    "read_labeled_text",
    "evaluate",
    "train_perceptron",
]
