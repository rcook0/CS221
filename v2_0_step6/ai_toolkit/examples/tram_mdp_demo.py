from __future__ import annotations

from ai_toolkit.domains.tram import TransportationMDP, TramCosts, render_tram_grid
from ai_toolkit.mdp import policy_iteration, value_iteration


def main() -> None:
    mdp = TransportationMDP(N=10, fail_prob=0.9, costs=TramCosts(walk=1.0, tram=2.0))

    print("=== Value Iteration (render every 2 iters) ===")

    def render(it, V, pi, delta):
        print(f"\niter={it} delta={delta:.3e}")
        print(render_tram_grid(V, pi, N=10))

    res_vi = value_iteration(mdp, epsilon=1e-12, render_every=2, render_fn=render)
    print(f"\nconverged: iters={res_vi.iterations} delta={res_vi.delta:.3e}")
    print(render_tram_grid(res_vi.V, res_vi.policy, N=10))

    print("\n=== Policy Iteration ===")
    res_pi = policy_iteration(mdp, eval_epsilon=1e-12)
    print(f"iters={res_pi.iterations} delta={res_pi.delta:.3e}")
    print(render_tram_grid(res_pi.V, res_pi.policy, N=10))


if __name__ == "__main__":
    main()
