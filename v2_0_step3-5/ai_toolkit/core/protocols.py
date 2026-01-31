from __future__ import annotations

from typing import Hashable, Iterable, Protocol, Tuple, TypeVar

S = TypeVar("S", bound=Hashable)
A = TypeVar("A")


class SearchProblem(Protocol[S, A]):
    """Deterministic search problem.

    successors(s) yields (action, next_state, step_cost).
    For UCS/A*, step_cost must be nonnegative.
    """

    def start_state(self) -> S: ...

    def is_goal(self, state: S) -> bool: ...

    def successors(self, state: S) -> Iterable[Tuple[A, S, float]]: ...


class MDP(Protocol[S, A]):
    """Tabular Markov Decision Process interface used by the toolkit.

    succ_prob_reward(s,a) yields (s_next, prob, reward). Probabilities should sum to 1
    across successors for any fixed (s,a).
    """

    def states(self) -> Iterable[S]: ...

    def actions(self, state: S) -> Iterable[A]: ...

    def succ_prob_reward(self, state: S, action: A) -> Iterable[Tuple[S, float, float]]: ...

    def is_terminal(self, state: S) -> bool: ...

    def discount(self) -> float: ...


class ZeroSumGame(Protocol[S, A]):
    """Deterministic turn-taking zero-sum game under perfect information."""

    def start_state(self) -> S: ...

    def current_player(self, state: S) -> int: ...

    def actions(self, state: S) -> Iterable[A]: ...

    def succ(self, state: S, action: A) -> S: ...

    def is_terminal(self, state: S) -> bool: ...

    def utility(self, state: S) -> float: ...
