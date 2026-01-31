import io
import json
import tempfile
import unittest


class TestSchemas(unittest.TestCase):
    def test_trace_jsonl_header_has_versions(self):
        from ai_toolkit.domains.tram import TransportationProblem, TramCosts
        from ai_toolkit.search import astar
        from ai_toolkit.viz import write_search_trace_jsonl

        problem = TransportationProblem(10, costs=TramCosts(walk=1.0, tram=2.0))
        res = astar(problem, heuristic=problem.admissible_heuristic, trace=True)
        self.assertIsNotNone(res.trace)

        with tempfile.TemporaryDirectory() as td:
            p = f"{td}/trace.jsonl"
            write_search_trace_jsonl(
                res.trace,
                p,
                result=res,
                context={"domain": "tram", "algo": "astar", "N": 10},
            )
            with open(p, "r", encoding="utf-8") as f:
                first = json.loads(f.readline())
                self.assertEqual(first["type"], "header")
                self.assertIn("trace_schema", first)
                self.assertIn("result_schema", first)
                self.assertIn("toolkit_version", first)

    def test_bench_rows_have_schema_envelope(self):
        from ai_toolkit.cli.bench import benchmark_tram_search, to_jsonl

        rows = benchmark_tram_search([8], repeats=1, seed=123)
        buf = io.StringIO()
        to_jsonl(rows, buf)
        line = buf.getvalue().splitlines()[0]
        obj = json.loads(line)

        for k in ("schema_version", "toolkit_version", "timestamp_utc", "seed", "trial", "domain", "algo"):
            self.assertIn(k, obj)


if __name__ == "__main__":
    unittest.main()
