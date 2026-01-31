import unittest

from ai_toolkit.domains.tram import TransportationMDP
from ai_toolkit.mdp import policy_iteration, value_iteration


class TestMDPPolicyIteration(unittest.TestCase):
    def test_policy_iteration_matches_value_iteration(self):
        mdp = TransportationMDP(N=12, fail_prob=0.5)
        vi = value_iteration(mdp, epsilon=1e-12)
        pi = policy_iteration(mdp, eval_epsilon=1e-12)

        # Policies should match on this small MDP (ties are unlikely here).
        for s in range(1, 12):
            self.assertEqual(vi.policy[s], pi.policy[s])

        # Values should be very close.
        for s in range(1, 13):
            self.assertAlmostEqual(vi.V[s], pi.V[s], places=7)


if __name__ == "__main__":
    unittest.main()
