"""Self-test for summarize.py. Run: python3 -m unittest test_summarize"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "summarize.py")
CONFIG = {"warmup": 1}


def rust_report():
    return {"schema": "inference-engine.dispatch-cost.v1", "config": CONFIG,
            "setup_us": 5.0,
            "results": [{"layers": 2, "calls_per_chain": 4,
                         "per_call_us_from_chain_median": 2.5,
                         "chain": {"median_us": 10.0, "p95_us": 12.0}}]}


def py_report():
    rows = [{"mode": m, "layers": 2, "calls_per_chain": 4,
             "chain": {"median_us": v, "p95_us": v + 1.0}}
            for m, v in (("sync", 20.0), ("lazy", 8.0))]
    return {"schema": "inference-engine.dispatch-cost-reference.v1",
            "config": CONFIG, "mlx_version": "0.0", "results": rows}


class SummarizeTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def path(self, name):
        return os.path.join(self.dir.name, name)

    def write(self, name, text):
        with open(self.path(name), "w") as f:
            f.write(text)
        return self.path(name)

    def run_script(self, *args):
        return subprocess.run([sys.executable, SCRIPT, *args],
                              capture_output=True, text=True)

    def inputs(self):
        return (self.write("r.json", json.dumps(rust_report())),
                self.write("p.json", json.dumps(py_report())))

    def test_passing_case(self):
        r, p = self.inputs()
        out = self.path("o.json")
        res = self.run_script(r, p, "--output", out)
        self.assertEqual(res.returncode, 0, res.stderr)
        row = json.loads(res.stdout.splitlines()[0])
        self.assertAlmostEqual(row["rust_over_py_sync"], 0.5)
        with open(out) as f:
            self.assertEqual(len(json.load(f)["rows"]), 1)

    def test_empty_output_fails(self):
        r, p = self.inputs()
        res = self.run_script(r, p, "--output", "")
        self.assertEqual(res.returncode, 1)
        self.assertIn("summarize: error:", res.stderr)
        self.assertEqual(
            res.stderr, "summarize: error: --output needs a non-empty file path\n")
        self.assertEqual(res.stdout, "")
        self.assertEqual(sorted(os.listdir(self.dir.name)), ["p.json", "r.json"])

    def test_duplicate_key_fails_in_each_input(self):
        good_r, good_p = self.inputs()
        dup = self.write("dup.json", '{"schema": "a", "schema": "b"}')
        for args in ((dup, good_p), (good_r, dup)):
            res = self.run_script(*args)
            self.assertEqual(res.returncode, 1)
            self.assertIn("summarize: error:", res.stderr)
            self.assertIn(f"{args[0] if args[0] == dup else args[1]} has a duplicate JSON key 'schema'", res.stderr)
            self.assertNotIn("not valid JSON", res.stderr)

    def test_nested_duplicate_key_fails(self):
        text = json.dumps(rust_report()).replace(
            '"median_us": 10.0', '"median_us": 10.0, "median_us": 11.0')
        r = self.write("r2.json", text)
        _, p = self.inputs()
        res = self.run_script(r, p)
        self.assertEqual(res.returncode, 1)
        self.assertIn("has a duplicate JSON key 'median_us'", res.stderr)


if __name__ == "__main__":
    unittest.main()
