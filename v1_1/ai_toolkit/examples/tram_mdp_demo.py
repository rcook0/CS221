from __future__ import annotations

from ai_toolkit.domains.tram import TransportationMDP
from ai_toolkit.mdp import value_iteration


def main() -> None:
    mdp = TransportationMDP(N=10, fail_prob=0.9)
    res = value_iteration(mdp, epsilon=1e-12)

    print(f"iterations={res.iterations} delta={res.delta:.3e}")
    print(f"{'s':>3} {'V(s)':>12} {'pi(s)':>8}")
    for s in mdp.states():
        pi = res.policy[s] if res.policy[s] is not None else "none"
        print(f"{s:>3} {res.V[s]:>12.6f} {pi:>8}")


if __name__ == "__main__":
    main()
