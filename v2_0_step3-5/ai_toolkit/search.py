"""Compatibility re-export for ai_toolkit v2.0 layout.

This module remains for backwards compatibility with v0.x/v1.x. Prefer importing from
ai_toolkit.core.* (or ai_toolkit.viz / ai_toolkit.domains) in new code.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "ai_toolkit.search is deprecated; use ai_toolkit.core.search.",
    DeprecationWarning,
    stacklevel=2,
)

from .core.search.algorithms import *  # noqa: F401,F403
