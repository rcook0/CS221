from __future__ import annotations

import random
from typing import Optional

try:
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover
    _np = None


def make_rng(seed: Optional[int]) -> random.Random:
    """Create an isolated RNG for deterministic runs.

    Use this instead of the global random module state. If seed is None, uses
    system entropy (non-deterministic).
    """
    return random.Random(seed)


def seed_numpy(seed: Optional[int]) -> None:
    """Seed NumPy if available (no-op if NumPy is unavailable)."""
    if _np is None:
        return
    if seed is None:
        return
    _np.random.seed(int(seed))
