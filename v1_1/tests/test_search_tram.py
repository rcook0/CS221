import unittest

from ai_toolkit.domains.tram import TransportationProblem, TramCosts, shortest_cost_dp
from ai_toolkit.search import ucs, astar


class TestSearchTram(unittest.TestCase):
    def test_tram_optimal_cost(self):
        problem = TransportationProblem(10, costs=TramCosts(walk=1.0, tram=2.0))
        dp_cost, _ = shortest_cost_dp(problem)
        u = ucs(problem)
        a = astar(problem, heuristic=problem.admissible_heuristic)

        self.assertEqual(dp_cost, 6.0)
        self.assertEqual(u.cost, 6.0)
        self.assertEqual(a.cost, 6.0)

    def test_custom_costs(self):
        problem = TransportationProblem(20, costs=TramCosts(walk=1.0, tram=0.1))
        dp_cost, _ = shortest_cost_dp(problem)
        u = ucs(problem)
        self.assertAlmostEqual(dp_cost, u.cost, places=9)
        self.assertLess(u.cost, 19.0)


if __name__ == "__main__":
    unittest.main()
