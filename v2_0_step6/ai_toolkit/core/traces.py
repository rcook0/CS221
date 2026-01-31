from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Generic, Hashable, List, Optional, Tuple, TypeVar

S = TypeVar("S", bound=Hashable)
A = TypeVar("A")

# Frozen schema version for serialized traces.
TRACE_SCHEMA_VERSION = "2.0"


@dataclass
class SearchTrace(Generic[S, A]):
    """Optional trace data captured during search.

    Notes
    -----
    - parent and g_score represent the best-known predecessor tree at termination.
    - expanded_order is the order states were expanded (popped from frontier).
    - generated_edges can be large; collect only when explicitly requested.
    """

    parent: Dict[S, Tuple[Optional[S], Optional[A]]]
    g_score: Dict[S, float]
    expanded_order: List[S]
    generated_edges: Optional[List[Tuple[S, S, A, float]]] = None
