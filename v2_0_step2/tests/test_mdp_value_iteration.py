import unittest

from ai_toolkit.domains.tram import TransportationMDP
from ai_toolkit.mdp import value_iteration


class TestMDPValueIteration(unittest.TestCase):
    def test_value_iteration_runs(self):
        mdp = TransportationMDP(N=10, fail_prob=0.9)
        res = value_iteration(mdp, epsilon=1e-12, max_iters=100_000)
        self.assertAlmostEqual(res.V[10], 0.0, places=12)
        for s in range(1, 10):
            self.assertIn(res.policy[s], ("walk", "tram"))
        self.assertIsNone(res.policy[10])


if __name__ == "__main__":
    unittest.main()
