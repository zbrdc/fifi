#!/usr/bin/env python3
"""L0 gate over repeated dispatch-cost runs.

Usage: l0_gate.py RUN1.json RUN2.json ... [--output GATE.json] [--limit 1.25] [--min-runs 5]
Each input is one Rust report. A series is one measurement: the chain per-call time for a layer count,
one operation's median per-call time (including the intermediate release), or the layer release time.
The gate ratio of a series is the maximum over the minimum of that series' medians across the runs.
All times are microseconds.
The gate passes only when all of these hold:
- every input is valid: finite numbers, stable finite output, same series, config, warmup, and repetitions;
- there are at least MIN_RUNS different runs and each run has at least MIN_REPETITIONS repetitions;
- every ratio is within LIMIT;
- the load average is known and not above LOAD_1M_IDLE_MAX in every sample;
- the GPU utilization is known and not above GPU_BUSY_PCT in every sample.
Invalid input includes a host sample that is missing, not a dict, lacks `load_1m` or `gpu_device_utilization_pct`, or is negative.
It also includes a `layers` value below 1 or repeated, a negative or zero time, and a row with both `series_host` and `rep_host`.
A `null` host value means unknown and keeps the gate open. A zero median is invalid input.
Exit status: 0 for PASS, 1 for OPEN, 2 for invalid input, 3 for NONSTANDARD.
Invalid input stops the script with a nonzero exit status. An unknown host state keeps the gate open.
The limit and the minimum run count are fixed. Other values give a NONSTANDARD result that cannot pass.
"""
import hashlib
import json
import math
import os
import sys

LIMIT = 1.25
MIN_RUNS = 5
MIN_REPETITIONS = 5
GPU_BUSY_PCT = 5
# Worker choice. The plan gives no idle load value. The harness is expected to add about 1.0 to the 1 minute load. No run measured it.
# The user must confirm this value before an idle-host rerun counts as the baseline.
LOAD_1M_IDLE_MAX = 2.0


class GateInput(Exception):
    """The input is not valid for the gate."""


def check_number(value, what, non_negative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise GateInput(f"{what} is not a finite number: {value!r}")
    if non_negative and value < 0:
        raise GateInput(f"{what} is negative: {value!r}")
    return value


def series(report, name="report"):
    out = {}
    seen = set()
    try:
        for row in report["results"]:
            n = row["layers"]
            if isinstance(n, bool) or not isinstance(n, int) or n < 1 or n in seen:
                raise GateInput(f"{name}: layers value {n!r} is below 1, not an integer, or repeated")
            seen.add(n)
            if row.get("output_stable") is not True or row.get("output_finite") is not True:
                raise GateInput(f"{name}: output for {n} layers is not stable and finite")
            key = f"chain_per_call_us[{n} layers]"
            out[key] = check_number(row["per_call_us_from_chain_median"], f"{name} {key}", True)
            for op, s in row["per_op"].items():
                key = f"{op}_median_us[{n} layers]"
                out[key] = check_number(s["median_us"], f"{name} {key}", True)
        out["release_per_layer_us"] = check_number(report["release_per_layer_us"], f"{name} release_per_layer_us", True)
        out["adapter_release_us"] = check_number(report["adapter_release_us"], f"{name} adapter_release_us", True)
    except (KeyError, TypeError, AttributeError) as e:
        raise GateInput(f"{name}: missing or malformed field {e}") from e
    if not report["results"]:
        raise GateInput(f"{name}: no results")
    return out


def host_values(report, name="report"):
    """Return every host sample. Raise GateInput when a required sample is missing or not a dict."""
    hosts = [report.get("host_start"), report.get("host_end")]
    for row in report["results"]:
        sh = row.get("series_host")
        if isinstance(sh, dict) and "rep_host" in row:
            raise GateInput(f"{name}: row has both series_host and rep_host")
        if isinstance(sh, dict):
            hosts += [sh.get("before"), sh.get("after")]
        elif isinstance(row.get("rep_host"), list) and row["rep_host"]:
            for rep in row["rep_host"]:
                if not isinstance(rep, dict):
                    raise GateInput(f"{name}: malformed rep_host element")
                hosts += [rep.get("before"), rep.get("after")]
        else:
            raise GateInput(f"{name}: missing series_host for {row.get('layers')!r} layers")
    for h in hosts:
        if not isinstance(h, dict):
            raise GateInput(f"{name}: missing or non-dict host sample {h!r}")
        for key in ("load_1m", "gpu_device_utilization_pct"):
            if key not in h:
                raise GateInput(f"{name}: host sample lacks {key}")
    return hosts


def canonical_sha256(report):
    return hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()


def evaluate(reports, limit=LIMIT, min_runs=MIN_RUNS):
    if not reports:
        raise GateInput("no input reports")
    tables = [series(r, f"run {i + 1}") for i, r in enumerate(reports)]
    for i, r in enumerate(reports):
        for key in ("config", "warmup", "repetitions"):
            if r.get(key) != reports[0].get(key):
                raise GateInput(f"run {i + 1}: {key} differs from run 1")
        reps = r.get("repetitions")
        if isinstance(reps, bool) or not isinstance(reps, int) or reps < MIN_REPETITIONS:
            raise GateInput(f"run {i + 1}: repetitions {reps!r} is below {MIN_REPETITIONS}")
        if set(tables[i]) != set(tables[0]):
            raise GateInput(f"run {i + 1}: series set differs from run 1: "
                            f"{sorted(set(tables[i]) ^ set(tables[0]))}")
    hashes = [canonical_sha256(r) for r in reports]
    if len(set(hashes)) != len(hashes):
        raise GateInput("two inputs hold the same report")
    rows = []
    for name in tables[0]:
        vals = [t[name] for t in tables]
        lo, hi = min(vals), max(vals)
        if lo <= 0:
            raise GateInput(f"{name}: a run has a zero value")
        ratio = hi / lo
        if not math.isfinite(ratio):
            raise GateInput(f"{name}: ratio is not finite")
        rows.append({"series": name, "min": lo, "max": hi, "ratio": ratio, "within_limit": ratio <= limit})
    hosts = [h for i, r in enumerate(reports) for h in host_values(r, f"run {i + 1}")]
    loads = [check_number(h["load_1m"], "load_1m", True) for h in hosts if h.get("load_1m") is not None]
    gpus = [check_number(h["gpu_device_utilization_pct"], "gpu_device_utilization_pct", True) for h in hosts
            if h.get("gpu_device_utilization_pct") is not None]
    host = {
        "load_1m_min": min(loads) if loads else None,
        "load_1m_max": max(loads) if loads else None,
        "load_1m_idle_max": LOAD_1M_IDLE_MAX,
        "load_samples": len(loads),
        "load_unknown": not loads or len(loads) != len(hosts),
        "load_busy": bool(loads) and max(loads) > LOAD_1M_IDLE_MAX,
        "gpu_util_max_pct": max(gpus) if gpus else None,
        "gpu_samples": len(gpus),
        "gpu_unknown": not gpus or len(gpus) != len(hosts),
        "gpu_busy": bool(gpus) and max(gpus) > GPU_BUSY_PCT,
        "samples": len(hosts),
    }
    if not math.isfinite(limit) or limit <= 0:
        raise GateInput(f"limit {limit!r} is not a positive finite number")
    standard = limit == LIMIT and min_runs == MIN_RUNS
    host_ok = not (host["load_unknown"] or host["load_busy"] or host["gpu_unknown"] or host["gpu_busy"])
    ok = standard and len(reports) >= min_runs and all(r["within_limit"] for r in rows) and host_ok
    return {"schema": "inference-engine.dispatch-cost-l0-gate.v2", "runs": len(reports), "min_runs": min_runs,
            "limit": limit, "standard": standard, "input_canonical_sha256": hashes, "rows": rows,
            "host": host, "host_ok": host_ok, "pass": ok,
            "worst_ratio": max(r["ratio"] for r in rows)}


def parse_args(argv):
    opts = {"--output": None, "--limit": LIMIT, "--min-runs": MIN_RUNS}
    args = list(argv)
    for key in opts:
        while key in args:
            k = args.index(key)
            if k + 1 >= len(args):
                raise GateInput(f"option {key} needs a value")
            opts[key] = args[k + 1]
            del args[k:k + 2]
    return args, opts


def reject_duplicate_keys(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise GateInput(f"duplicate JSON key {key!r}")
        out[key] = value
    return out


def load_report(path):
    with open(path, "rb") as f:
        data = f.read()
    try:
        report = json.loads(data, object_pairs_hook=reject_duplicate_keys)
    except ValueError as e:
        raise GateInput(f"{path}: {e}") from e
    return report, hashlib.sha256(data).hexdigest()


def main():
    try:
        args, opts = parse_args(sys.argv[1:])
        if not args:
            raise GateInput("no input files. Usage: l0_gate.py RUN1.json ... [--output F]")
        loaded = [load_report(p) for p in args]
        file_hashes = [h for _, h in loaded]
        if len(set(file_hashes)) != len(file_hashes):
            raise GateInput("two input files have the same SHA-256")
        result = evaluate([r for r, _ in loaded], float(opts["--limit"]), int(opts["--min-runs"]))
    except (GateInput, OSError, ValueError, RecursionError, OverflowError) as e:
        print(f"l0_gate: invalid input: {e}", file=sys.stderr)
        sys.exit(2)
    result["input_files"] = dict(zip(args, file_hashes))
    try:
        text = json.dumps(result, indent=2, allow_nan=False)
    except ValueError as e:
        print(f"l0_gate: result is not valid JSON: {e}", file=sys.stderr)
        sys.exit(2)
    if opts["--output"]:
        # The output path is a fresh file in a new directory, so the symlink and mode
        # concerns of item 96 do not apply. A temporary file and a rename leave no partial file.
        tmp = f"{opts['--output']}.tmp{os.getpid()}"
        try:
            with open(tmp, "w") as f:
                f.write(text)
            os.replace(tmp, opts["--output"])
        except OSError as e:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            print(f"l0_gate: cannot write output: {e}", file=sys.stderr)
            sys.exit(2)
    for r in result["rows"]:
        print(f"{r['series']:44} min {r['min']:12.3f} max {r['max']:12.3f} ratio {r['ratio']:6.3f} {'ok' if r['within_limit'] else 'OVER'}")
    print(json.dumps(result["host"]))
    if result["pass"]:
        verdict = "PASS"
    elif not result["standard"]:
        verdict = "NONSTANDARD"
    else:
        verdict = "OPEN"
    print(f"runs {result['runs']} worst_ratio {result['worst_ratio']:.3f} host_ok {result['host_ok']} gate {verdict}")
    sys.exit({"PASS": 0, "OPEN": 1, "NONSTANDARD": 3}[verdict])


def entry():
    """Run main. A crash exits 2 (invalid), never 1 (OPEN)."""
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print(f"l0_gate: internal error: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    entry()
