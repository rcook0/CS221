from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Hashable, Iterable, Optional, Protocol, Tuple, TypeVar

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


MDPRenderFn = Callable[[int, Dict[S, float], Dict[S, Optional[A]], float], None]


def _Q(mdp: MDP[S, A], s: S, a: A, Vref: Dict[S, float], gamma: float) -> float:
    return sum(prob * (reward + gamma * Vref[s2]) for s2, prob, reward in mdp.succ_prob_reward(s, a))


def greedy_policy(mdp: MDP[S, A], V: Dict[S, float]) -> Dict[S, Optional[A]]:
    """Greedy policy with respect to V (ties broken by first max)."""
    states = list(mdp.states())
    gamma = float(mdp.discount())
    policy: Dict[S, Optional[A]] = {}
    for s in states:
        if mdp.is_terminal(s):
            policy[s] = None
            continue
        best_a: Optional[A] = None
        best_q = float("-inf")
        for a in mdp.actions(s):
            q = _Q(mdp, s, a, V, gamma)
            if q > best_q:
                best_q, best_a = q, a
        policy[s] = best_a
    return policy


def value_iteration(
    mdp: MDP[S, A],
    *,
    epsilon: float = 1e-10,
    max_iters: int = 100_000,
    render_every: int = 0,
    render_fn: Optional[MDPRenderFn] = None,
) -> MDPResult:
    """Value Iteration.

    If render_every > 0 and render_fn is provided, render_fn(iter, V, greedy_pi, delta)
    is called every k iterations using the *current* value function.
    """
    states = list(mdp.states())
    gamma = float(mdp.discount())

    V: Dict[S, float] = {s: 0.0 for s in states}
    it = 0
    delta = float("inf")

    for it in range(1, max_iters + 1):
        newV: Dict[S, float] = {}
        for s in states:
            if mdp.is_terminal(s):
                newV[s] = 0.0
            else:
                newV[s] = max(_Q(mdp, s, a, V, gamma) for a in mdp.actions(s))
        delta = max(abs(newV[s] - V[s]) for s in states)
        V = newV

        if render_fn is not None and render_every > 0 and (it % render_every == 0):
            render_fn(it, V, greedy_policy(mdp, V), delta)

        if delta < epsilon:
            break

    policy = greedy_policy(mdp, V)
    return MDPResult(V=V, policy=policy, iterations=it, delta=delta)


def policy_iteration(
    mdp: MDP[S, A],
    *,
    max_iters: int = 10_000,
    eval_epsilon: float = 1e-10,
    eval_max_iters: int = 100_000,
    render_every: int = 0,
    render_fn: Optional[MDPRenderFn] = None,
) -> MDPResult:
    """Policy Iteration with iterative policy evaluation.

    - policy evaluation: iterative sweeps until eval_epsilon or eval_max_iters
    - policy improvement: greedy w.r.t. evaluated V

    If render_every > 0 and render_fn is provided, render_fn(iter, V, pi, last_eval_delta)
    is called every k *outer* iterations.
    """
    states = list(mdp.states())
    gamma = float(mdp.discount())

    # init V=0, pi arbitrary
    V: Dict[S, float] = {s: 0.0 for s in states}
    policy: Dict[S, Optional[A]] = {}
    for s in states:
        if mdp.is_terminal(s):
            policy[s] = None
        else:
            acts = list(mdp.actions(s))
            policy[s] = acts[0] if acts else None

    outer_delta = float("inf")

    for it in range(1, max_iters + 1):
        # --- Policy evaluation
        for _k in range(eval_max_iters):
            newV: Dict[S, float] = {}
            for s in states:
                if mdp.is_terminal(s):
                    newV[s] = 0.0
                else:
                    a = policy[s]
                    if a is None:
                        newV[s] = 0.0
                    else:
                        newV[s] = _Q(mdp, s, a, V, gamma)
            outer_delta = max(abs(newV[s] - V[s]) for s in states)
            V = newV
            if outer_delta < eval_epsilon:
                break

        # --- Policy improvement
        stable = True
        for s in states:
            if mdp.is_terminal(s):
                continue
            old = policy[s]
            best_a: Optional[A] = None
            best_q = float("-inf")
            for a in mdp.actions(s):
                q = _Q(mdp, s, a, V, gamma)
                if q > best_q:
                    best_q, best_a = q, a
            policy[s] = best_a
            if best_a != old:
                stable = False

        if render_fn is not None and render_every > 0 and (it % render_every == 0):
            render_fn(it, V, policy, outer_delta)

        if stable:
            return MDPResult(V=V, policy=policy, iterations=it, delta=outer_delta)

    return MDPResult(V=V, policy=policy, iterations=max_iters, delta=outer_delta)
