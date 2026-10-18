"""Self-test for l0_gate.py. Run: python3 -m unittest test_l0_gate"""
import copy
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest

import l0_gate

HOST = {"load_1m": 1.0, "load_5m": 1.0, "load_15m": 1.0, "gpu_device_utilization_pct": 0}
HERE = os.path.dirname(os.path.abspath(__file__))


def report(per_call, release=10.0, gpu=0, load=1.0, layers=(1,), uid=0.0):
    host = dict(HOST, gpu_device_utilization_pct=gpu, load_1m=load)
    return {"config": {"hidden": 1024, "inter": 2048, "seq": 256}, "warmup": 10, "repetitions": 100,
            "release_per_layer_us": release, "adapter_release_us": 5.0 + uid,
            "host_start": host, "host_end": host,
            "results": [{"layers": n, "per_call_us_from_chain_median": per_call,
                         "per_op": {"qmm": {"median_us": per_call * 2}},
                         "output_stable": True, "output_finite": True,
                         "series_host": {"before": host, "after": host}} for n in layers]}


def five(**kw):
    return [report(10.0 + i * 0.1, uid=i * 0.01, **kw) for i in range(5)]


class GateTest(unittest.TestCase):
    def test_pass(self):
        r = l0_gate.evaluate(five())
        self.assertTrue(r["pass"])

    def test_ratio_over_limit_opens_gate(self):
        runs = five()
        runs[4] = report(13.0, uid=0.04)
        r = l0_gate.evaluate(runs)
        self.assertFalse(r["pass"])
        self.assertAlmostEqual(r["worst_ratio"], 1.3, places=1)

    def test_release_series_is_gated(self):
        runs = five()
        runs[4] = report(10.4, release=20.0, uid=0.04)
        self.assertFalse(l0_gate.evaluate(runs)["pass"])

    def test_too_few_runs(self):
        self.assertFalse(l0_gate.evaluate(five()[:4])["pass"])

    def test_busy_gpu_opens_gate(self):
        runs = five()
        runs[4] = report(10.4, gpu=40, uid=0.04)
        r = l0_gate.evaluate(runs)
        self.assertTrue(r["host"]["gpu_busy"])
        self.assertFalse(r["pass"])

    def test_busy_load_opens_gate(self):
        r = l0_gate.evaluate(five(load=40.0))
        self.assertTrue(r["host"]["load_busy"])
        self.assertFalse(r["pass"])

    def test_load_at_limit_passes(self):
        self.assertTrue(l0_gate.evaluate(five(load=l0_gate.LOAD_1M_IDLE_MAX))["pass"])

    def test_missing_gpu_samples_keep_gate_open(self):
        runs = five()
        for r in runs:
            for h in (r["host_start"], r["host_end"], r["results"][0]["series_host"]["before"],
                      r["results"][0]["series_host"]["after"]):
                h["gpu_device_utilization_pct"] = None
        r = l0_gate.evaluate(runs)
        self.assertTrue(r["host"]["gpu_unknown"])
        self.assertFalse(r["pass"])

    def test_missing_load_samples_keep_gate_open(self):
        runs = five()
        for r in runs:
            for h in (r["host_start"], r["host_end"], r["results"][0]["series_host"]["before"],
                      r["results"][0]["series_host"]["after"]):
                h["load_1m"] = None
        r = l0_gate.evaluate(runs)
        self.assertTrue(r["host"]["load_unknown"])
        self.assertFalse(r["pass"])

    def test_nan_in_later_run_is_rejected(self):
        runs = five()
        runs[1]["release_per_layer_us"] = math.nan
        with self.assertRaises(l0_gate.GateInput):
            l0_gate.evaluate(runs)

    def test_nan_in_first_run_is_rejected(self):
        runs = five()
        runs[0]["release_per_layer_us"] = math.nan
        with self.assertRaises(l0_gate.GateInput):
            l0_gate.evaluate(runs)

    def test_null_value_is_rejected(self):
        runs = five()
        runs[1]["release_per_layer_us"] = None
        with self.assertRaises(l0_gate.GateInput):
            l0_gate.evaluate(runs)

    def test_empty_input_is_rejected(self):
        with self.assertRaises(l0_gate.GateInput):
            l0_gate.evaluate([])

    def test_series_missing_in_first_run_is_rejected(self):
        runs = five(layers=(1, 8))
        del runs[0]["results"][1]
        with self.assertRaises(l0_gate.GateInput):
            l0_gate.evaluate(runs)

    def test_series_missing_in_later_run_is_rejected(self):
        runs = five(layers=(1, 8))
        del runs[2]["results"][1]
        with self.assertRaises(l0_gate.GateInput):
            l0_gate.evaluate(runs)

    def test_config_mismatch_is_rejected(self):
        runs = five()
        runs[0]["config"]["hidden"] = 4096
        with self.assertRaises(l0_gate.GateInput):
            l0_gate.evaluate(runs)

    def test_warmup_mismatch_is_rejected(self):
        runs = five()
        runs[3]["warmup"] = 1
        with self.assertRaises(l0_gate.GateInput):
            l0_gate.evaluate(runs)

    def test_repetitions_mismatch_is_rejected(self):
        runs = five()
        runs[3]["repetitions"] = 99
        with self.assertRaises(l0_gate.GateInput):
            l0_gate.evaluate(runs)

    def test_single_repetition_is_rejected(self):
        runs = five()
        for r in runs:
            r["repetitions"] = 1
        with self.assertRaises(l0_gate.GateInput):
            l0_gate.evaluate(runs)

    def test_unstable_output_is_rejected(self):
        runs = five()
        runs[2]["results"][0]["output_stable"] = False
        with self.assertRaises(l0_gate.GateInput):
            l0_gate.evaluate(runs)

    def test_non_finite_output_is_rejected(self):
        runs = five()
        runs[2]["results"][0]["output_finite"] = False
        with self.assertRaises(l0_gate.GateInput):
            l0_gate.evaluate(runs)

    def test_duplicate_reports_are_rejected(self):
        runs = [copy.deepcopy(report(10.0)) for _ in range(5)]
        with self.assertRaises(l0_gate.GateInput):
            l0_gate.evaluate(runs)

    def test_n1_missing_host_samples_in_later_runs_are_rejected(self):
        runs = five()
        for r in runs[1:]:
            del r["host_start"], r["host_end"]
            del r["results"][0]["series_host"]
        with self.assertRaises(l0_gate.GateInput):
            l0_gate.evaluate(runs)

    def test_n2_non_dict_host_samples_are_rejected(self):
        runs = five()
        runs[0]["host_start"] = "busy"
        runs[0]["host_end"] = [40]
        with self.assertRaises(l0_gate.GateInput):
            l0_gate.evaluate(runs)

    def test_n3_missing_host_end_is_rejected(self):
        runs = five()
        for r in runs:
            del r["host_end"]
        with self.assertRaises(l0_gate.GateInput):
            l0_gate.evaluate(runs)

    def test_missing_series_host_after_is_rejected(self):
        runs = five()
        del runs[2]["results"][0]["series_host"]["after"]
        with self.assertRaises(l0_gate.GateInput):
            l0_gate.evaluate(runs)

    def test_n4_negative_load_is_rejected(self):
        with self.assertRaises(l0_gate.GateInput):
            l0_gate.evaluate(five(load=-50.0))

    def test_n5_negative_gpu_is_rejected(self):
        with self.assertRaises(l0_gate.GateInput):
            l0_gate.evaluate(five(gpu=-10))

    def test_n12_duplicate_layers_rows_are_rejected(self):
        runs = five()
        dup = copy.deepcopy(runs[0]["results"][0])
        dup["per_call_us_from_chain_median"] = 50.0
        runs[0]["results"].append(dup)
        with self.assertRaises(l0_gate.GateInput):
            l0_gate.evaluate(runs)

    def test_nonstandard_limit_does_not_pass(self):
        r = l0_gate.evaluate(five(), limit=2.0)
        self.assertFalse(r["standard"])
        self.assertFalse(r["pass"])

    def test_nonstandard_min_runs_does_not_pass(self):
        r = l0_gate.evaluate(five()[:3], min_runs=2)
        self.assertFalse(r["pass"])


class CliTest(unittest.TestCase):
    def run_cli(self, args, runs=None):
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for i, r in enumerate(runs or []):
                p = os.path.join(d, f"r{i}.json")
                with open(p, "w") as f:
                    json.dump(r, f)
                paths.append(p)
            return subprocess.run([sys.executable, os.path.join(HERE, "l0_gate.py")] + paths + args,
                                  capture_output=True, text=True)

    def test_uncaught_exception_exits_2(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "r.json")
            with open(p, "w") as f:
                json.dump(five()[0], f)
            code = "import l0_gate; l0_gate.evaluate = lambda *a: 1 / 0; l0_gate.entry()"
            r = subprocess.run([sys.executable, "-c", code, p], cwd=HERE, capture_output=True, text=True)
            self.assertEqual(r.returncode, 2)
            self.assertIn("internal error", r.stderr)

    def test_keyboard_interrupt_is_not_exit_2(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "r.json")
            with open(p, "w") as f:
                json.dump(five()[0], f)
            code = ("import l0_gate\n"
                    "def boom(*a): raise KeyboardInterrupt\n"
                    "l0_gate.evaluate = boom\n"
                    "l0_gate.entry()")
            r = subprocess.run([sys.executable, "-c", code, p], cwd=HERE, capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 2)
            self.assertNotEqual(r.returncode, 0)

    def test_write_error_leaves_no_gate_json(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "g.json")
            code = ("import l0_gate, os; real = os.replace\n"
                    "def bad(*a): raise OSError('disk full')\n"
                    "os.replace = bad\n"
                    "l0_gate.entry()")
            paths = []
            for i, r in enumerate(five()):
                q = os.path.join(d, f"r{i}.json")
                with open(q, "w") as f:
                    json.dump(r, f)
                paths.append(q)
            r = subprocess.run([sys.executable, "-c", code] + paths + ["--output", out],
                               cwd=HERE, capture_output=True, text=True)
            self.assertEqual(r.returncode, 2)
            self.assertFalse(os.path.exists(out))
            self.assertEqual([n for n in os.listdir(d) if not n.startswith("r")], [])

    def test_option_without_value_exits_cleanly(self):
        p = self.run_cli(["--output"], five())
        self.assertNotEqual(p.returncode, 0)
        self.assertNotIn("Traceback", p.stderr)

    def test_corrupt_input_exits_nonzero_without_traceback(self):
        runs = five()
        runs[1]["release_per_layer_us"] = None
        p = self.run_cli([], runs)
        self.assertNotEqual(p.returncode, 0)
        self.assertNotIn("Traceback", p.stderr)

    def test_duplicate_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.json")
            with open(p, "w") as f:
                json.dump(report(10.0), f)
            q = subprocess.run([sys.executable, os.path.join(HERE, "l0_gate.py")] + [p] * 5,
                               capture_output=True, text=True)
        self.assertNotEqual(q.returncode, 0)

    def test_exit_codes_distinguish_verdicts(self):
        self.assertEqual(self.run_cli([], five()).returncode, 0)
        self.assertEqual(self.run_cli([], five(load=40.0)).returncode, 1)
        self.assertEqual(self.run_cli(["--limit", "2.0"], five()).returncode, 3)

    def test_missing_host_key_exits_2(self):
        for key in ("load_1m", "gpu_device_utilization_pct"):
            runs = five()
            del runs[0]["host_start"][key]
            self.assertEqual(self.run_cli([], runs).returncode, 2, key)

    def test_null_host_value_is_unknown_exit_1(self):
        runs = five()
        for r in runs:
            r["host_start"] = dict(r["host_start"], load_1m=None)
        self.assertEqual(self.run_cli([], runs).returncode, 1)

    def test_nonpositive_layers_exit_2(self):
        for n in (0, -1):
            self.assertEqual(self.run_cli([], [report(10.0 + i * 0.1, layers=(n,), uid=i * 0.01)
                                               for i in range(5)]).returncode, 2, n)

    def test_negative_times_exit_2(self):
        for edit in (lambda r: r["results"][0].__setitem__("per_call_us_from_chain_median", -1.0),
                     lambda r: r["results"][0]["per_op"]["qmm"].__setitem__("median_us", -1.0),
                     lambda r: r.__setitem__("release_per_layer_us", -1.0),
                     lambda r: r.__setitem__("adapter_release_us", -1.0)):
            runs = five()
            edit(runs[0])
            self.assertEqual(self.run_cli([], runs).returncode, 2)

    def test_series_host_with_rep_host_exits_2(self):
        runs = five()
        runs[0]["results"][0]["rep_host"] = [1]
        self.assertEqual(self.run_cli([], runs).returncode, 2)

    def test_no_arguments_and_bad_output_exit_2(self):
        self.assertEqual(self.run_cli([], []).returncode, 2)
        self.assertEqual(self.run_cli(["--output", "/nonexistent/dir/g.json"], five()).returncode, 2)

    def test_deep_json_exits_2(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.json")
            with open(p, "w") as f:
                f.write("[" * 100000)
            q = subprocess.run([sys.executable, os.path.join(HERE, "l0_gate.py"), p],
                               capture_output=True, text=True)
        self.assertEqual(q.returncode, 2)
        self.assertNotIn("Traceback", q.stderr)

    def test_huge_integer_exits_2(self):
        runs = five()
        runs[0]["release_per_layer_us"] = 10**400
        p = self.run_cli([], runs)
        self.assertEqual(p.returncode, 2)
        self.assertNotIn("Traceback", p.stderr)

    def test_zero_median_exits_2_and_writes_no_infinity(self):
        runs = [report(0.0, uid=i * 0.01) for i in range(5)]
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "g.json")
            p = self.run_cli(["--output", out], runs)
            self.assertEqual(p.returncode, 2)
            self.assertFalse(os.path.exists(out))

    def test_unwritable_output_prints_no_verdict(self):
        p = self.run_cli(["--output", "/nonexistent/dir/g.json"], five())
        self.assertEqual(p.returncode, 2)
        self.assertNotIn("gate PASS", p.stdout)

    def test_nonfinite_limit_exits_2_and_writes_no_file(self):
        for lim in ("inf", "nan", "1e400", "0", "-1"):
            with tempfile.TemporaryDirectory() as d:
                out = os.path.join(d, "g.json")
                p = self.run_cli(["--limit", lim, "--output", out], five())
                self.assertEqual(p.returncode, 2, lim)
                self.assertFalse(os.path.exists(out), lim)

    def test_infinite_ratio_exits_2_and_writes_no_file(self):
        runs = five()
        for i, r in enumerate(runs):
            r["adapter_release_us"] = 1e-320 if i == 0 else 5.0 + i
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "g.json")
            p = self.run_cli(["--output", out], runs)
            self.assertEqual(p.returncode, 2)
            self.assertFalse(os.path.exists(out))

    def test_good_input_passes(self):
        p = self.run_cli([], five())
        self.assertEqual(p.returncode, 0)
        self.assertIn("gate PASS", p.stdout)


if __name__ == "__main__":
    unittest.main()
