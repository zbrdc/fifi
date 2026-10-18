"""Self-tests of purity.py on small synthetic evidence directories (standard library only).
Run: python3 -I -m unittest discover -s qualification/mlx-purity -t qualification/mlx-purity"""
import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path

import purity

CONFIGS = ("steps", "chunk256")


def bank(n=3, p=0.999):
    prompts = []
    for i in range(n):
        res = {c: {"p_format": p - 0.0001 * i, "top1_id": 198, "top1_is_format": True,
                   "logit_format": 20.0 + i, "seconds": 1.0} for c in CONFIGS}
        prompts.append({"index": i, "n_tokens": 100, "source": "s", "results": res})
    return {"configs": list(CONFIGS), "prompts": prompts}


def route(sha="aa", cos=0.9999, floor=0.995):
    return {"min_logits_cosine": cos, "min_logits_cosine_floor": floor, "greedy_ids_equal_in_all_steps": True,
            "device_logits_sha256": sha, "host_logits_sha256": "bb", "cells_differing": 5,
            "cell_gap_max": 0.004, "cell_gap_median": 0.0002}


def write(root, banks, routes):
    for mode, j in banks.items():
        d = Path(root) / f"mxc6-bank-{mode}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "bank-fifi.json").write_text(json.dumps(j))
    for name, j in routes.items():
        d = Path(root) / f"mxc4-route-{name}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "route-parity.json").write_text(json.dumps(j))


def default_evidence():
    banks = {"off": bank(), "on": bank()}
    routes = {n: route(sha=n) for n in purity.ROUTES}
    return banks, routes


def run(args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = purity.main(["purity.py"] + args)
    return code, out.getvalue()


class PurityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.old = Path(self.tmp.name) / "old"
        self.new = Path(self.tmp.name) / "new"
        self.base = Path(self.tmp.name) / "base.json"
        banks, routes = default_evidence()
        write(self.old, banks, routes)
        self.banks, self.routes = banks, routes
        self.assertEqual(run(["freeze", str(self.old), str(self.base)])[0], 0)

    def new_with(self, banks=None, routes=None):
        write(self.new, banks or copy.deepcopy(self.banks), routes or copy.deepcopy(self.routes))

    def test_identical_evidence_passes(self):
        self.new_with()
        code, text = run(["compare", str(self.base), str(self.new)])
        self.assertEqual(code, 0, text)

    def test_a_changed_uncompiled_bank_value_breaks_purity(self):
        banks = copy.deepcopy(self.banks)
        banks["off"]["prompts"][1]["results"]["steps"]["logit_format"] += 1e-6
        self.new_with(banks=banks)
        code, text = run(["compare", str(self.base), str(self.new)])
        self.assertEqual(code, 1, text)
        self.assertIn("bank off/steps prompt 1", text)

    def test_timing_fields_do_not_count(self):
        banks = copy.deepcopy(self.banks)
        banks["off"]["prompts"][0]["results"]["steps"]["seconds"] = 99.0
        self.new_with(banks=banks)
        self.assertEqual(run(["compare", str(self.base), str(self.new)])[0], 0)

    def test_a_changed_uncompiled_route_hash_breaks_purity(self):
        routes = copy.deepcopy(self.routes)
        routes["bf16-off"]["device_logits_sha256"] = "cc"
        self.new_with(routes=routes)
        code, text = run(["compare", str(self.base), str(self.new)])
        self.assertEqual(code, 1, text)
        self.assertIn("route bf16-off", text)

    def test_a_compiled_shift_within_the_gates_passes_and_is_reported(self):
        banks = copy.deepcopy(self.banks)
        banks["on"]["prompts"][0]["results"]["chunk256"]["p_format"] = 0.995
        routes = copy.deepcopy(self.routes)
        routes["f32-on"]["device_logits_sha256"] = "dd"
        routes["bf16-on"]["min_logits_cosine"] = 0.998
        self.new_with(banks=banks, routes=routes)
        code, text = run(["compare", str(self.base), str(self.new)])
        self.assertEqual(code, 0, text)
        self.assertIn("shift bank on/p_format", text)

    def test_a_compiled_p_format_below_the_floor_fails_the_gate(self):
        banks = copy.deepcopy(self.banks)
        banks["on"]["prompts"][2]["results"]["steps"]["p_format"] = 0.98
        self.new_with(banks=banks)
        code, text = run(["compare", str(self.base), str(self.new)])
        self.assertEqual(code, 2, text)
        self.assertIn("below 0.99", text)

    def test_a_compiled_top1_loss_fails_the_gate(self):
        banks = copy.deepcopy(self.banks)
        banks["on"]["prompts"][0]["results"]["steps"]["top1_is_format"] = False
        self.new_with(banks=banks)
        self.assertEqual(run(["compare", str(self.base), str(self.new)])[0], 2)

    def test_a_compiled_cosine_below_its_floor_fails_the_gate(self):
        routes = copy.deepcopy(self.routes)
        routes["bf16-on"]["min_logits_cosine"] = 0.99
        self.new_with(routes=routes)
        code, text = run(["compare", str(self.base), str(self.new)])
        self.assertEqual(code, 2, text)
        self.assertIn("route bf16-on", text)

    def test_purity_failure_wins_over_a_gate_failure(self):
        banks = copy.deepcopy(self.banks)
        banks["off"]["prompts"][0]["results"]["steps"]["p_format"] = 0.9
        banks["on"]["prompts"][0]["results"]["steps"]["p_format"] = 0.9
        self.new_with(banks=banks)
        self.assertEqual(run(["compare", str(self.base), str(self.new)])[0], 1)

    def test_a_missing_file_is_invalid_input(self):
        self.new_with()
        (self.new / "mxc4-route-f32-on" / "route-parity.json").unlink()
        code, text = run(["compare", str(self.base), str(self.new)])
        self.assertEqual(code, 3, text)

    def test_a_different_prompt_count_is_invalid_input(self):
        banks = copy.deepcopy(self.banks)
        banks["off"]["prompts"].pop()
        self.new_with(banks=banks)
        self.assertEqual(run(["compare", str(self.base), str(self.new)])[0], 3)

    def test_freeze_records_the_source_hashes_and_the_meta(self):
        run(["freeze", str(self.old), str(self.base), "--mlx-version", "0.32.3", "--engine-head", "abc"])
        j = json.loads(self.base.read_text())
        self.assertEqual(j["meta"]["mlx_version"], "0.32.3")
        self.assertEqual(len(j["sources_sha256"]), 6)
        self.assertTrue(all(len(h) == 64 for h in j["sources_sha256"].values()))


if __name__ == "__main__":
    unittest.main()
