from __future__ import annotations

from functools import lru_cache
from typing import Generic, Hashable, Iterable, Optional, Tuple, TypeVar

S = TypeVar("S", bound=Hashable)
A = TypeVar("A")

from ..protocols import ZeroSumGame
from ..results import GameResult

def minimax(game: ZeroSumGame[S, A], state: S, *, memoize: bool = True) -> GameResult[A]:
    nodes = 0

    def rec(s: S) -> Tuple[float, Optional[A]]:
        nonlocal nodes
        nodes += 1
        if game.is_terminal(s):
            return game.utility(s), None
        p = game.current_player(s)
        best_val = float("-inf") if p == +1 else float("inf")
        best_act: Optional[A] = None
        for a in game.actions(s):
            v, _ = rec(game.succ(s, a))
            if p == +1:
                if v > best_val:
                    best_val, best_act = v, a
            else:
                if v < best_val:
                    best_val, best_act = v, a
        return best_val, best_act

    if memoize:
        rec = lru_cache(maxsize=None)(rec)  # type: ignore[assignment]
    v, a = rec(state)
    return GameResult(value=v, action=a, nodes=nodes)


def alphabeta(game: ZeroSumGame[S, A], state: S) -> GameResult[A]:
    nodes = 0

    def rec(s: S, alpha: float, beta: float) -> Tuple[float, Optional[A]]:
        nonlocal nodes
        nodes += 1
        if game.is_terminal(s):
            return game.utility(s), None
        p = game.current_player(s)
        if p == +1:
            best_val = float("-inf")
            best_act: Optional[A] = None
            for a in game.actions(s):
                v, _ = rec(game.succ(s, a), alpha, beta)
                if v > best_val:
                    best_val, best_act = v, a
                alpha = max(alpha, best_val)
                if alpha >= beta:
                    break
            return best_val, best_act
        else:
            best_val = float("inf")
            best_act = None
            for a in game.actions(s):
                v, _ = rec(game.succ(s, a), alpha, beta)
                if v < best_val:
                    best_val, best_act = v, a
                beta = min(beta, best_val)
                if alpha >= beta:
                    break
            return best_val, best_act

    v, a = rec(state, float("-inf"), float("inf"))
    return GameResult(value=v, action=a, nodes=nodes)