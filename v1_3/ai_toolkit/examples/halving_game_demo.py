from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

from ai_toolkit.games import minimax, alphabeta, ZeroSumGame


@dataclass(frozen=True)
class HalvingGame(ZeroSumGame[Tuple[int, int], str]):
    N: int

    def start_state(self) -> Tuple[int, int]:
        return (+1, self.N)

    def current_player(self, state: Tuple[int, int]) -> int:
        p, _n = state
        return p

    def actions(self, state: Tuple[int, int]) -> Iterable[str]:
        _p, n = state
        return [] if n == 0 else ["-", "/"]

    def succ(self, state: Tuple[int, int], action: str) -> Tuple[int, int]:
        p, n = state
        if action == "-":
            return (-p, n - 1)
        if action == "/":
            return (-p, n // 2)
        raise ValueError(action)

    def is_terminal(self, state: Tuple[int, int]) -> bool:
        _p, n = state
        return n == 0

    def utility(self, state: Tuple[int, int]) -> float:
        p_to_move, _n = state
        loser = p_to_move
        winner = -loser
        return float("inf") if winner == +1 else float("-inf")


def main() -> None:
    game = HalvingGame(15)
    s = game.start_state()
    r1 = minimax(game, s, memoize=True)
    r2 = alphabeta(game, s)
    print(f"minimax:   value={r1.value} action={r1.action} nodes={r1.nodes}")
    print(f"alphabeta: value={r2.value} action={r2.action} nodes={r2.nodes}")


if __name__ == "__main__":
    main()
