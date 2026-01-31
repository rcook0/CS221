import os
import subprocess
import sys
import tempfile
import unittest


class TestCliSmoke(unittest.TestCase):
    def test_run_tram_writes_trace(self):
        with tempfile.TemporaryDirectory() as td:
            trace_path = os.path.join(td, "trace.jsonl")
            cmd = [
                sys.executable,
                "-m",
                "ai_toolkit",
                "run",
                "tram",
                "--algo",
                "ucs",
                "--N",
                "12",
                "--trace-out",
                trace_path,
            ]
            out = subprocess.check_output(cmd, text=True)
            self.assertIn("cost:", out)
            self.assertTrue(os.path.exists(trace_path))

    def test_bench_tram_writes_jsonl(self):
        with tempfile.TemporaryDirectory() as td:
            out_path = os.path.join(td, "bench.jsonl")
            cmd = [
                sys.executable,
                "-m",
                "ai_toolkit",
                "bench",
                "tram",
                "--Ns",
                "8,10",
                "--algo",
                "ucs",
                "--repeats",
                "1",
                "--seed",
                "7",
                "--out",
                out_path,
                "--format",
                "jsonl",
            ]
            out = subprocess.check_output(cmd, text=True)
            # CLI prints nothing special on success, but should create file.
            self.assertTrue(os.path.exists(out_path))
            with open(out_path, "r", encoding="utf-8") as f:
                first = f.readline()
            self.assertTrue(first.strip())


if __name__ == "__main__":
    unittest.main()
