from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Hashable, Iterable, Optional, Protocol, Tuple, TypeVar

S = TypeVar("S", bound=Hashable)
A = TypeVar("A")


class MDP(Protocol[S, A]):
    def states(self) -> Iterable[S]: ...
    def actions(self, state: S) -> Iterable[A]: ...
    def succ_prob_reward(self, state: S, action: A) -> Iterable[Tuple[S, float, float]]: ...
    def is_terminal(self, state: S) -> bool: ...
    def discount(self) -> float: ...


@dataclass(frozen=True)
class MDPResult:
    V: Dict[S, float]
    policy: Dict[S, Optional[A]]
    iterations: int
    delta: float


def value_iteration(
    mdp: MDP[S, A],
    *,
    epsilon: float = 1e-10,
    max_iters: int = 100_000,
) -> MDPResult:
    states = list(mdp.states())
    gamma = float(mdp.discount())

    V: Dict[S, float] = {s: 0.0 for s in states}
    it = 0
    delta = float("inf")

    def Q(s: S, a: A, Vref: Dict[S, float]) -> float:
        return sum(prob * (reward + gamma * Vref[s2]) for s2, prob, reward in mdp.succ_prob_reward(s, a))

    for it in range(1, max_iters + 1):
        newV: Dict[S, float] = {}
        for s in states:
            if mdp.is_terminal(s):
                newV[s] = 0.0
            else:
                newV[s] = max(Q(s, a, V) for a in mdp.actions(s))
        delta = max(abs(newV[s] - V[s]) for s in states)
        V = newV
        if delta < epsilon:
            break

    policy: Dict[S, Optional[A]] = {}
    for s in states:
        if mdp.is_terminal(s):
            policy[s] = None
        else:
            best = max(((Q(s, a, V), a) for a in mdp.actions(s)), key=lambda x: x[0])
            policy[s] = best[1]

    return MDPResult(V=V, policy=policy, iterations=it, delta=delta)
