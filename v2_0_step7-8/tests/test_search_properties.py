import unittest


class RandomGraphProblem:
    """Tiny SearchProblem for property testing.

    Nodes are ints [0..n-1]. Edges are directed with nonnegative costs.
    """

    def __init__(self, adj, start, goal):
        self._adj = adj
        self._start = start
        self._goal = goal

    def start_state(self):
        return self._start

    def is_goal(self, s):
        return s == self._goal

    def successors(self, s):
        # (action, next_state, step_cost) ; action is a tuple for traceability
        for (t, c) in self._adj[s]:
            yield ("go", t, c)


def dijkstra_costs(adj, goal):
    """Compute exact shortest-path distance-to-goal for admissible heuristics."""
    import heapq

    n = len(adj)
    rev = [[] for _ in range(n)]
    for u in range(n):
        for v, c in adj[u]:
            rev[v].append((u, c))

    dist = [float("inf")] * n
    dist[goal] = 0.0
    pq = [(0.0, goal)]
    while pq:
        d, x = heapq.heappop(pq)
        if d != dist[x]:
            continue
        for p, c in rev[x]:
            nd = d + c
            if nd < dist[p]:
                dist[p] = nd
                heapq.heappush(pq, (nd, p))
    return dist


class TestSearchProperties(unittest.TestCase):
    def test_astar_matches_ucs_on_admissible(self):
        import random
        from ai_toolkit.search import ucs, astar

        rng = random.Random(0)
        for _ in range(25):
            n = rng.randint(6, 18)
            goal = n - 1

            # Generate a connected-ish directed graph.
            adj = [[] for _ in range(n)]
            # Ensure a backbone path 0->1->...->goal
            for i in range(n - 1):
                adj[i].append((i + 1, float(rng.randint(1, 9))))
            # Add random extra edges
            m = rng.randint(n, 4 * n)
            for _e in range(m):
                u = rng.randrange(n)
                v = rng.randrange(n)
                if u == v:
                    continue
                c = float(rng.randint(1, 9))
                adj[u].append((v, c))

            prob = RandomGraphProblem(adj, start=0, goal=goal)
            d2g = dijkstra_costs(adj, goal)

            def h(s):
                return d2g[s]

            u = ucs(prob)
            a = astar(prob, heuristic=h)
            self.assertIsNotNone(u.cost)
            self.assertIsNotNone(a.cost)
            self.assertAlmostEqual(u.cost, a.cost, places=9)

    def test_astar_reopen_handles_inconsistent_admissible(self):
        import random
        from ai_toolkit.search import ucs, astar

        rng = random.Random(1)
        for _ in range(15):
            n = rng.randint(6, 16)
            goal = n - 1
            adj = [[] for _ in range(n)]
            for i in range(n - 1):
                adj[i].append((i + 1, float(rng.randint(1, 9))))
            for _e in range(rng.randint(n, 3 * n)):
                u = rng.randrange(n)
                v = rng.randrange(n)
                if u == v:
                    continue
                adj[u].append((v, float(rng.randint(1, 9))))

            prob = RandomGraphProblem(adj, start=0, goal=goal)
            exact = dijkstra_costs(adj, goal)

            # Admissible but potentially inconsistent: sharply lower heuristic on a few nodes.
            h_tab = exact[:]
            for _k in range(rng.randint(1, max(1, n // 3))):
                v = rng.randrange(n)
                if v == goal:
                    continue
                # Drop heuristic toward 0 (still <= exact => admissible)
                h_tab[v] = max(0.0, h_tab[v] - float(rng.randint(1, 20)))

            def h(s):
                return h_tab[s]

            u = ucs(prob)
            a = astar(prob, heuristic=h)
            self.assertAlmostEqual(u.cost, a.cost, places=9)
            # Inconsistency often forces reopens; not guaranteed, but at least ensure metric is sane.
            self.assertGreaterEqual(a.reopens, 0)


if __name__ == "__main__":
    unittest.main()
