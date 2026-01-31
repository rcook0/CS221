from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, Generic, Hashable, List, Optional, Tuple, TypeVar

S = TypeVar("S", bound=Hashable)


@dataclass
class PriorityQueue(Generic[S]):
    """A tiny priority queue with decrease-key semantics.

    - update(state, priority) inserts or decreases.
    - pop_min() returns the currently-best (state, priority), skipping stale entries.

    This implementation allows a state to be re-opened later with a better priority,
    which is important for A* when the heuristic is admissible but not consistent.
    """

    _heap: List[Tuple[float, S]] = field(default_factory=list, init=False, repr=False)
    _best: Dict[S, float] = field(default_factory=dict, init=False, repr=False)

    def update(self, state: S, priority: float) -> bool:
        old = self._best.get(state)
        if old is None or priority < old:
            self._best[state] = priority
            heapq.heappush(self._heap, (priority, state))
            return True
        return False

    def pop_min(self) -> Tuple[Optional[S], Optional[float]]:
        while self._heap:
            pri, state = heapq.heappop(self._heap)
            if self._best.get(state) == pri:
                return state, pri
        return None, None

    def __len__(self) -> int:
        return len(self._heap)
