from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ...mdp import MDP
from ...search import SearchProblem


@dataclass(frozen=True)
class TramCosts:
    walk: float = 1.0
    tram: float = 2.0


class TransportationProblem(SearchProblem[int, str]):
    def __init__(self, N: int, costs: TramCosts = TramCosts()):
        if N < 1:
            raise ValueError("N must be >= 1")
        self.N = int(N)
        self.costs = costs

    def start_state(self) -> int:
        return 1

    def is_goal(self, state: int) -> bool:
        return state == self.N

    def successors(self, state: int) -> Iterable[Tuple[str, int, float]]:
        if state + 1 <= self.N:
            yield "walk", state + 1, float(self.costs.walk)
        if state * 2 <= self.N:
            yield "tram", state * 2, float(self.costs.tram)

    def admissible_heuristic(self, state: int) -> float:
        """Admissible heuristic: (min remaining action count) * (cheapest action cost).

        We compute the exact *minimum number of actions* to reach N in a unit-cost relaxation
        (each action has cost 1). This is a lower bound on the number of actions in any plan.
        Multiplying by the cheapest possible action cost yields a valid lower bound on cost.
        """
        if state == self.N:
            return 0.0

        cache: Dict[int, int] = {}

        def min_steps(s: int) -> int:
            if s == self.N:
                return 0
            if s in cache:
                return cache[s]
            best = 10**18
            if s + 1 <= self.N:
                best = min(best, 1 + min_steps(s + 1))
            if s * 2 <= self.N:
                best = min(best, 1 + min_steps(s * 2))
            cache[s] = best
            return best

        cheapest = float(min(self.costs.walk, self.costs.tram))
        return float(min_steps(state)) * cheapest


def shortest_cost_dp(problem: TransportationProblem) -> Tuple[float, List[Tuple[str, int, float]]]:
    cache: Dict[int, Tuple[float, Optional[Tuple[str, int, float]]]] = {}

    def best_from(s: int) -> float:
        if problem.is_goal(s):
            return 0.0
        if s in cache:
            return cache[s][0]
        best_cost = float("inf")
        best_step: Optional[Tuple[str, int, float]] = None
        for a, s2, c in problem.successors(s):
            cost = float(c) + best_from(s2)
            if cost < best_cost:
                best_cost = cost
                best_step = (a, s2, float(c))
        cache[s] = (best_cost, best_step)
        return best_cost

    total = best_from(problem.start_state())
    hist: List[Tuple[str, int, float]] = []
    s = problem.start_state()
    while not problem.is_goal(s):
        step = cache[s][1]
        assert step is not None
        hist.append(step)
        s = step[1]
    return total, hist


class TransportationMDP(MDP[int, str]):
    def __init__(self, N: int, *, fail_prob: float = 0.9, costs: TramCosts = TramCosts()):
        self.N = int(N)
        self.fail_prob = float(fail_prob)
        self.costs = costs

    def states(self) -> Iterable[int]:
        return range(1, self.N + 1)

    def actions(self, state: int) -> Iterable[str]:
        if self.is_terminal(state):
            return []
        acts: List[str] = []
        if state + 1 <= self.N:
            acts.append("walk")
        if state * 2 <= self.N:
            acts.append("tram")
        return acts

    def succ_prob_reward(self, state: int, action: str):
        if action == "walk":
            yield (state + 1, 1.0, -float(self.costs.walk))
        elif action == "tram":
            yield (state * 2, 1.0 - self.fail_prob, -float(self.costs.tram))
            yield (state, self.fail_prob, -float(self.costs.tram))
        else:
            raise ValueError(action)

    def is_terminal(self, state: int) -> bool:
        return state == self.N

    def discount(self) -> float:
        return 1.0


def structured_perceptron_action_costs(
    examples: Sequence[Tuple[int, Sequence[str]]],
    *,
    iters: int = 50,
) -> Dict[str, float]:
    weights: Dict[str, float] = {"walk": 0.0, "tram": 0.0}

    def predict_actions(N: int) -> List[str]:
        p = TransportationProblem(N, costs=TramCosts(walk=weights["walk"], tram=weights["tram"]))
        _cost, hist = shortest_cost_dp(p)
        return [a for a, _s2, _c in hist]

    for _t in range(iters):
        mistakes = 0
        for N, y_true in examples:
            y_pred = predict_actions(N)
            if list(y_true) != list(y_pred):
                mistakes += 1
            # weights are *costs* => true should become cheaper, pred should become pricier
            for a in y_true:
                weights[a] -= 1.0
            for a in y_pred:
                weights[a] += 1.0
        if mistakes == 0:
            break
    return weights


def render_tram_grid(
    V: Dict[int, float],
    policy: Dict[int, Optional[str]],
    *,
    N: int,
    value_width: int = 9,
    value_precision: int = 2,
) -> str:
    """Stable, fixed-width table for the tram MDP.

    Example (N=10):
      s :    1    2    3 ...   10
      pi:    w    t    w ...    .
      V : -5.3 -4.7 -...     0.0

    Where pi uses:
      - 'w' for walk
      - 't' for tram
      - '.' for terminal/None
    """
    states = list(range(1, int(N) + 1))

    def pi_char(s: int) -> str:
        a = policy.get(s)
        if a is None:
            return "."
        if a == "walk":
            return "w"
        if a == "tram":
            return "t"
        return str(a)[:1]

    s_line = "s :" + "".join(f"{s:>{value_width}d}" for s in states)
    pi_line = "pi:" + "".join(f"{pi_char(s):>{value_width}}" for s in states)

    fmt = f"{{:>{value_width}.{value_precision}f}}"
    v_line = "V :" + "".join(fmt.format(float(V.get(s, 0.0))) for s in states)

    return "\n".join([s_line, pi_line, v_line])
