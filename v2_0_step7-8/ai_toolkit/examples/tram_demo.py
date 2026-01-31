from __future__ import annotations

from ai_toolkit.domains.tram import TransportationProblem, shortest_cost_dp
from ai_toolkit.search import ucs, astar


def main() -> None:
    N = 10_000
    problem = TransportationProblem(N)

    dp_cost, dp_hist = shortest_cost_dp(problem)
    print(f"[DP]   cost={dp_cost:.0f} steps={len(dp_hist)}")

    u = ucs(problem)
    print(f"[UCS]  cost={u.cost:.0f} steps={len(u.actions)} expanded={u.expanded}")

    a = astar(problem, heuristic=problem.admissible_heuristic)
    print(f"[A*]   cost={a.cost:.0f} steps={len(a.actions)} expanded={a.expanded}")


if __name__ == "__main__":
    main()
