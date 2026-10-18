#!/usr/bin/env python3
"""Join the Rust and Python dispatch cost reports into one table.

Usage: summarize.py RUST.json PYTHON.json [--output SUMMARY.json]
Ratios use median chain wall time. All values are microseconds.
"""

import json
import math
import sys


def fail(message):
    sys.exit(f"summarize: error: {message}")


class DuplicateKey(ValueError):
    pass


def reject_duplicates(pairs):
    seen = set()
    for key, _ in pairs:
        if key in seen:
            raise DuplicateKey(f"duplicate JSON key {key!r}")
        seen.add(key)
    return dict(pairs)


def load(path):
    try:
        with open(path) as f:
            return json.load(f, object_pairs_hook=reject_duplicates)
    except OSError as e:
        fail(f"cannot read {path}: {e.strerror or e}")
    except DuplicateKey as e:
        fail(f"{path} has a {e}")
    except ValueError as e:
        fail(f"{path} is not valid JSON: {e}")


def check_number(value, what, positive=False, nonneg=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{what} is not a finite number: {value!r}")
    try:
        finite = float(value) - float(value) == 0
    except OverflowError:
        finite = False
    if not finite:
        fail(f"{what} is not a finite number")
    if positive and value <= 0:
        fail(f"{what} must be greater than 0: {value!r}")
    if nonneg and value < 0:
        fail(f"{what} must not be negative: {value!r}")


def check_count(value, what):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        fail(f"{what} is not an integer greater than 0: {value!r}")
    check_number(value, what)


PY_SCHEMA = "inference-engine.dispatch-cost-reference.v1"


def check_schema(report, name, schema):
    if not isinstance(report, dict):
        fail(f"{name} report is not a JSON object")
    for key in ("schema", "config", "results"):
        if key not in report:
            fail(f"{name} report lacks {key}")
    if report["schema"] != schema:
        fail(f"unexpected {name} schema {report['schema']!r}")
    if not isinstance(report["results"], list) or not report["results"]:
        fail(f"{name} report has no results")
    for row in report["results"]:
        if not isinstance(row, dict):
            fail(f"{name} row is not a JSON object")
        if name == "Python":
            if "layers" not in row:
                fail("Python row lacks layers")
            check_count(row["layers"], "Python row layers")


def find_row(py, mode, layers):
    rows = [p for p in py["results"]
            if p.get("mode") == mode and p.get("layers") == layers]
    if len(rows) != 1:
        fail(f"Python report has {len(rows)} {mode} rows for "
                 f"{layers} layers, expected 1")
    row = rows[0]
    for key in ("calls_per_chain", "chain"):
        if key not in row:
            fail(f"Python {mode} row for {layers} layers lacks {key}")
    for key in ("median_us", "p95_us"):
        if not isinstance(row["chain"], dict) or key not in row["chain"]:
            fail(f"Python {mode} row for {layers} layers lacks chain.{key}")
    what = f"Python {mode} row for {layers} layers"
    check_count(row["calls_per_chain"], f"{what} calls_per_chain")
    check_number(row["chain"]["median_us"], f"{what} chain.median_us", True)
    check_number(row["chain"]["p95_us"], f"{what} chain.p95_us", nonneg=True)
    return row


def main():
    args = sys.argv[1:]
    out = None
    if "--output" in args:
        k = args.index("--output")
        if k + 1 >= len(args) or args[k + 1].startswith("--"):
            fail("--output needs a file path")
        out = args[k + 1]
        if not out:
            fail("--output needs a non-empty file path")
        del args[k:k + 2]
    if len(args) != 2:
        sys.exit(__doc__)
    rust, py = load(args[0]), load(args[1])
    check_schema(rust, "Rust", "inference-engine.dispatch-cost.v1")
    check_schema(py, "Python", PY_SCHEMA)
    for rep, name, key in ((rust, "Rust", "setup_us"), (py, "Python", "mlx_version")):
        if key not in rep:
            fail(f"{name} report lacks {key}")
    if not isinstance(py["mlx_version"], str):
        fail(f"Python mlx_version is not a string: {type(py['mlx_version']).__name__}")
    check_number(rust["setup_us"], "Rust setup_us", nonneg=True)
    if rust["config"] != py["config"]:
        fail("configurations differ")
    rows = []
    seen = set()
    for r in rust["results"]:
        for key in ("layers", "calls_per_chain", "per_call_us_from_chain_median"):
            if key not in r:
                fail(f"Rust row lacks {key}")
        for key in ("median_us", "p95_us"):
            if not isinstance(r.get("chain"), dict) or key not in r["chain"]:
                fail(f"Rust row for {r['layers']} layers lacks chain.{key}")
        n = r["layers"]
        check_count(n, "Rust row layers")
        what = f"Rust row for {n} layers"
        check_count(r["calls_per_chain"], f"{what} calls_per_chain")
        check_number(r["per_call_us_from_chain_median"],
                     f"{what} per_call_us_from_chain_median", positive=True)
        check_number(r["chain"]["median_us"], f"{what} chain.median_us",
                     positive=True)
        check_number(r["chain"]["p95_us"], f"{what} chain.p95_us", nonneg=True)
        if n in seen:
            fail(f"Rust report repeats {n} layers")
        seen.add(n)
        sync = find_row(py, "sync", n)
        lazy = find_row(py, "lazy", n)
        try:
            row = {
                "layers": n,
                "calls_per_chain": r["calls_per_chain"],
                "rust_chain_median_us": r["chain"]["median_us"],
                "rust_chain_p95_us": r["chain"]["p95_us"],
                "rust_per_call_us": r["per_call_us_from_chain_median"],
            }
            row["py_sync_chain_median_us"] = sync["chain"]["median_us"]
            row["py_sync_per_call_us"] = sync["chain"]["median_us"] / sync["calls_per_chain"]
            row["rust_over_py_sync"] = row["rust_chain_median_us"] / sync["chain"]["median_us"]
            row["py_lazy_chain_median_us"] = lazy["chain"]["median_us"]
            row["py_lazy_chain_p95_us"] = lazy["chain"]["p95_us"]
            row["py_lazy_over_sync"] = lazy["chain"]["median_us"] / sync["chain"]["median_us"]
            row["py_sync_minus_lazy_per_call_us"] = (
                sync["chain"]["median_us"] - lazy["chain"]["median_us"]
            ) / sync["calls_per_chain"]
            row["rust_over_py_lazy"] = row["rust_chain_median_us"] / lazy["chain"]["median_us"]
        except ArithmeticError as e:
            fail(f"{what} gives an invalid ratio: {e}")
        for key, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                fail(f"{what} gives a non-finite {key}")
        rows.append(row)
    try:
        lines = "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows)
        text = json.dumps({"schema": "inference-engine.dispatch-cost-summary.v1",
                           "rust_setup_us": rust["setup_us"],
                           "mlx_version": py["mlx_version"], "rows": rows},
                          indent=2, allow_nan=False)
    except (ValueError, TypeError, RecursionError) as e:
        fail(f"cannot encode the summary: {type(e).__name__}")
    if out:
        write_output(out, text)
    sys.stdout.write(lines)


def write_output(out, text):
    try:
        with open(out, "w") as f:
            f.write(text)
    except OSError as e:
        fail(f"cannot write {out}: {e.strerror or e}")


if __name__ == "__main__":
    main()
