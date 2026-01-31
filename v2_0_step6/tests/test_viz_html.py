import tempfile
import unittest

from ai_toolkit.domains.tram import TransportationProblem, TramCosts
from ai_toolkit.search import astar
from ai_toolkit.viz import write_search_trace_html


class TestHtmlVisualizer(unittest.TestCase):
    def test_html_written(self):
        problem = TransportationProblem(15, costs=TramCosts(walk=1.0, tram=2.0))
        res = astar(problem, heuristic=problem.admissible_heuristic, trace=True, trace_edges=True)

        with tempfile.TemporaryDirectory() as td:
            out = write_search_trace_html(res, f"{td}/viz.html", title="test")
            txt = out.read_text(encoding="utf-8")

        # Smoke-test: ensure the HTML contains the embedded data and key UI elements.
        self.assertIn("const DATA =", txt)
        self.assertIn("Legend:", txt)
        self.assertIn("btnPlay", txt)


if __name__ == "__main__":
    unittest.main()
