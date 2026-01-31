from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Generic, Hashable, List, Optional, TypeVar, Literal

from .traces import SearchTrace

S = TypeVar("S", bound=Hashable)
A = TypeVar("A")


Status = Literal["success", "failure", "timeout"]

# Frozen schema version for serialized results.
RESULT_SCHEMA_VERSION = "2.0"


@dataclass(frozen=True)
class SearchResult(Generic[S, A]):
    """Result of a deterministic search run.

    This keeps v1.x-compatible fields (cost/actions/states) while adding a minimal
    v2.0-compatible envelope (status/meta/schema_version).
    """

    cost: Optional[float]
    actions: List[A]
    states: List[S]
    expanded: int
    generated: int = 0
    reopens: int = 0
    max_frontier: int = 0
    runtime_sec: float = 0.0
    trace: Optional[SearchTrace[S, A]] = None

    status: Status = "success"
    meta: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = RESULT_SCHEMA_VERSION


@dataclass(frozen=True)
class MDPResult(Generic[S, A]):
    """Result of an MDP planning algorithm (VI/PI)."""

    V: Dict[S, float]
    policy: Dict[S, Optional[A]]
    iterations: int
    delta: float

    status: Status = "success"
    runtime_sec: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = RESULT_SCHEMA_VERSION


@dataclass(frozen=True)
class GameResult(Generic[A]):
    """Result of a game search run (minimax/alpha-beta)."""

    value: float
    action: Optional[A]
    nodes: int

    status: Status = "success"
    runtime_sec: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = RESULT_SCHEMA_VERSION
