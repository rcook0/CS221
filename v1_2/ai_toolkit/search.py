from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time
from typing import Callable, Deque, Dict, Generic, Hashable, Iterable, List, Optional, Protocol, Tuple, TypeVar

from .structures import PriorityQueue

S = TypeVar("S", bound=Hashable)
A = TypeVar("A")


class SearchProblem(Protocol[S, A]):
    def start_state(self) -> S: ...
    def is_goal(self, state: S) -> bool: ...
    def successors(self, state: S) -> Iterable[Tuple[A, S, float]]: ...


@dataclass(frozen=True)
class SearchResult(Generic[S, A]):
    cost: float
    actions: List[A]
    states: List[S]
    expanded: int
    generated: int = 0
    reopens: int = 0
    max_frontier: int = 0
    runtime_sec: float = 0.0


def _reconstruct(goal: S, parent: Dict[S, Tuple[Optional[S], Optional[A]]]) -> Tuple[List[S], List[A]]:
    states: List[S] = []
    actions: List[A] = []
    cur: Optional[S] = goal
    while cur is not None:
        states.append(cur)
        p, a = parent[cur]
        if a is not None:
            actions.append(a)
        cur = p
    states.reverse()
    actions.reverse()
    return states, actions


def bfs(problem: SearchProblem[S, A], *, max_expansions: int = 10_000_000) -> SearchResult[S, A]:
    t0 = time.perf_counter()
    start = problem.start_state()
    if problem.is_goal(start):
        return SearchResult(cost=0.0, actions=[], states=[start], expanded=0, max_frontier=1, runtime_sec=0.0)

    q: Deque[S] = deque([start])
    parent: Dict[S, Tuple[Optional[S], Optional[A]]] = {start: (None, None)}
    expanded = 0
    generated = 0
    max_frontier = len(q)

    while q:
        if expanded >= max_expansions:
            raise RuntimeError("BFS exceeded max_expansions")
        s = q.popleft()
        expanded += 1
        for a, s2, _c in problem.successors(s):
            generated += 1
            if s2 in parent:
                continue
            parent[s2] = (s, a)
            if problem.is_goal(s2):
                states, actions = _reconstruct(s2, parent)
                return SearchResult(
                    cost=float(len(actions)),
                    actions=actions,
                    states=states,
                    expanded=expanded,
                    generated=generated,
                    max_frontier=max_frontier,
                    runtime_sec=time.perf_counter() - t0,
                )
            q.append(s2)
            if len(q) > max_frontier:
                max_frontier = len(q)

    raise ValueError("No solution found (BFS)")


def dfs(problem: SearchProblem[S, A], *, max_expansions: int = 10_000_000) -> SearchResult[S, A]:
    t0 = time.perf_counter()
    start = problem.start_state()
    stack: List[S] = [start]
    parent: Dict[S, Tuple[Optional[S], Optional[A]]] = {start: (None, None)}
    expanded = 0
    generated = 0
    max_frontier = len(stack)

    while stack:
        if expanded >= max_expansions:
            raise RuntimeError("DFS exceeded max_expansions")
        s = stack.pop()
        expanded += 1
        if problem.is_goal(s):
            states, actions = _reconstruct(s, parent)
            return SearchResult(
                cost=float(len(actions)),
                actions=actions,
                states=states,
                expanded=expanded,
                generated=generated,
                max_frontier=max_frontier,
                runtime_sec=time.perf_counter() - t0,
            )
        for a, s2, _c in problem.successors(s):
            generated += 1
            if s2 in parent:
                continue
            parent[s2] = (s, a)
            stack.append(s2)
            if len(stack) > max_frontier:
                max_frontier = len(stack)

    raise ValueError("No solution found (DFS)")


def ucs(problem: SearchProblem[S, A], *, max_expansions: int = 10_000_000) -> SearchResult[S, A]:
    t0 = time.perf_counter()
    start = problem.start_state()
    frontier: PriorityQueue[S] = PriorityQueue()
    frontier.update(start, 0.0)

    parent: Dict[S, Tuple[Optional[S], Optional[A]]] = {start: (None, None)}
    best_g: Dict[S, float] = {start: 0.0}
    expanded = 0
    generated = 0
    reopens = 0
    max_frontier = len(frontier)

    while True:
        if expanded >= max_expansions:
            raise RuntimeError("UCS exceeded max_expansions")
        s, g = frontier.pop_min()
        if s is None or g is None:
            break
        # stale check (heap may contain older entries)
        if g != best_g.get(s):
            continue

        expanded += 1
        if problem.is_goal(s):
            states, actions = _reconstruct(s, parent)
            return SearchResult(
                cost=g,
                actions=actions,
                states=states,
                expanded=expanded,
                generated=generated,
                reopens=reopens,
                max_frontier=max_frontier,
                runtime_sec=time.perf_counter() - t0,
            )

        for a, s2, c in problem.successors(s):
            generated += 1
            g2 = g + float(c)
            if s2 in best_g and g2 < best_g[s2]:
                reopens += 1
            if s2 not in best_g or g2 < best_g[s2]:
                best_g[s2] = g2
                parent[s2] = (s, a)
                frontier.update(s2, g2)
                if len(frontier) > max_frontier:
                    max_frontier = len(frontier)

    raise ValueError("No solution found (UCS)")


def astar(
    problem: SearchProblem[S, A],
    heuristic: Callable[[S], float],
    *,
    max_expansions: int = 10_000_000,
) -> SearchResult[S, A]:
    t0 = time.perf_counter()
    start = problem.start_state()
    frontier: PriorityQueue[S] = PriorityQueue()
    best_g: Dict[S, float] = {start: 0.0}
    best_f: Dict[S, float] = {start: float(heuristic(start))}
    frontier.update(start, best_f[start])

    parent: Dict[S, Tuple[Optional[S], Optional[A]]] = {start: (None, None)}
    expanded = 0
    generated = 0
    reopens = 0
    max_frontier = len(frontier)

    while True:
        if expanded >= max_expansions:
            raise RuntimeError("A* exceeded max_expansions")
        s, f = frontier.pop_min()
        if s is None or f is None:
            break
        # stale entry?
        if f != best_f.get(s):
            continue

        g = best_g[s]
        expanded += 1
        if problem.is_goal(s):
            states, actions = _reconstruct(s, parent)
            return SearchResult(
                cost=g,
                actions=actions,
                states=states,
                expanded=expanded,
                generated=generated,
                reopens=reopens,
                max_frontier=max_frontier,
                runtime_sec=time.perf_counter() - t0,
            )

        for a, s2, c in problem.successors(s):
            generated += 1
            g2 = g + float(c)
            if s2 in best_g and g2 < best_g[s2]:
                reopens += 1
            if s2 not in best_g or g2 < best_g[s2]:
                best_g[s2] = g2
                parent[s2] = (s, a)
                f2 = g2 + float(heuristic(s2))
                best_f[s2] = f2
                frontier.update(s2, f2)
                if len(frontier) > max_frontier:
                    max_frontier = len(frontier)

    raise ValueError("No solution found (A*)")
