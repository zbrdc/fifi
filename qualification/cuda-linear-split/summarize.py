#!/usr/bin/env python3
"""Summarizes the P6-M1 bench CSVs of one machine.

Usage: python3 -I summarize.py <results-dir>

<results-dir> holds cold-ev0.csv, warm-ev0.csv, cold-ev1.csv and warm-ev1.csv (see
bench.cu). It writes summary.json and summary.md next to them. Times are
nanoseconds in the CSVs and microseconds (per call) or milliseconds (per token)
in the output. The median is the middle value; p90 is the nearest-rank 90th
percentile, sorted[ceil(0.9 * n) - 1].

Per-token figures sum calls-per-token times a per-call statistic. Per-call times are
right-skewed, so the sum of means is the expected per-token cost and the sum of
medians understates it. The sum of per-call p90s is a loose upper bound, not a
per-token p90.
"""
import csv
import json
import math
import pathlib
import statistics
import sys

MIN_CALLS = 200

# Calls of inference_cuda_linear per decoded token of Qwen3.6-35B-A3B (40 layers: 10
# full attention at 3,7,..,39 and 30 DeltaNet; 8 of 256 routed experts per layer):
#   attn_q       10 attn_q + 30 attn_qkv
#   attn_kv      10 x (attn_k + attn_v) + 40 x (shared gate + shared up)
#   attn_o       10 attn_output + 30 ssm_out
#   shexp_down   40
#   attn_gate    30 DeltaNet attn_gate
#   ssm_ab       30 x (ssm_alpha + ssm_beta)
#   exp_gate_up  40 x 8 x (gate + up)
#   exp_down     40 x 8
#   lm_head      1 output.weight
COUNTS = {
    "attn_q": 40,
    "attn_kv": 100,
    "attn_o": 40,
    "shexp_down": 40,
    "attn_gate": 30,
    "ssm_ab": 60,
    "exp_gate_up": 640,
    "exp_down": 320,
    "lm_head": 1,
}
CALLS_PER_TOKEN = 1271
assert sum(COUNTS.values()) == CALLS_PER_TOKEN

MALLOCS = ("malloc_x_ns", "malloc_xf_ns", "malloc_y_ns")
FREES = ("free_y_ns", "free_xf_ns", "free_x_ns")
PHASES = MALLOCS + ("h2d_ns", "memset_ns", "sync_ns", "d2h_ns") + FREES
TIMED = ("total_ns",) + PHASES + ("other_ns", "memset_win_ns", "kernel_win_ns", "floor_ns")
MODES = ("cold", "warm")


def median(values):
    return statistics.median(values)


def p90(values):
    ordered = sorted(values)
    return ordered[math.ceil(0.9 * len(ordered)) - 1]


def stat(values):
    return {"median": median(values), "mean": statistics.fmean(values), "p90": p90(values), "n": len(values)}


def load_csv(path, min_calls):
    """Returns {shape: [row dict of ints]} and checks every shape has min_calls rows."""
    shapes = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            parsed = {name: int(row[name]) for name in TIMED}
            parsed["expert"] = int(row["expert"])
            shapes.setdefault(row["shape"], []).append(parsed)
    for shape in COUNTS:
        rows = shapes.get(shape, [])
        if len(rows) < min_calls:
            raise SystemExit(f"{path}: shape {shape} has {len(rows)} calls, need {min_calls}")
    for shape, rows in shapes.items():
        if shape not in COUNTS:
            raise SystemExit(f"{path}: unknown shape {shape}")
        for row in rows:
            row["malloc_ns"] = sum(row[name] for name in MALLOCS)
            row["free_ns"] = sum(row[name] for name in FREES)
            row["alloc_free_ns"] = row["malloc_ns"] + row["free_ns"]
            row["non_alloc_ns"] = row["total_ns"] - row["alloc_free_ns"]
    return shapes


def per_shape(rows, names):
    return {name: stat([row[name] for row in rows]) for name in names}


def shape_summary(rows):
    names = TIMED + ("malloc_ns", "free_ns", "alloc_free_ns", "non_alloc_ns")
    return per_shape(rows, names)


def per_token(shapes, field):
    """Per-token milliseconds: sum over shapes of calls per token times the statistic."""
    out = {}
    for which in ("median", "mean", "p90"):
        total_ns = sum(COUNTS[name] * shapes[name][field][which] for name in COUNTS)
        out[which] = total_ns / 1e6
    return out


def summarize(results_dir, min_calls=MIN_CALLS):
    results_dir = pathlib.Path(results_dir)
    result = {"calls_per_token": CALLS_PER_TOKEN, "counts": COUNTS, "modes": {}}
    for mode in MODES:
        host = load_csv(results_dir / f"{mode}-ev0.csv", min_calls)
        events = load_csv(results_dir / f"{mode}-ev1.csv", min_calls)
        shapes = {name: shape_summary(host[name]) for name in COUNTS}
        gpu = {name: shape_summary(events[name]) for name in COUNTS}
        result["modes"][mode] = {
            "shapes": shapes,
            "events_run": gpu,
            "per_token_ms": {
                "total": per_token(shapes, "total_ns"),
                "alloc_free": per_token(shapes, "alloc_free_ns"),
                "non_alloc": per_token(shapes, "non_alloc_ns"),
                "in_call_gpu_window": per_token(gpu, "kernel_win_ns"),
                "floor": per_token(shapes, "floor_ns"),
            },
        }
    return result


# Per-token table: statistic key -> column header, and quantity key -> row label.
TOKEN_COLUMNS = {
    "median": "median-based (sum of per-call medians)",
    "mean": "mean-based (sum of per-call means, the expected cost)",
    "p90": "sum of per-call p90 (loose upper bound, not a per-token p90)",
}
TOKEN_ROWS = {
    "total": "all calls as measured",
    "alloc_free": "of which 6 cudaMalloc/cudaFree per call",
    "non_alloc": ("without malloc/free = cost of a persistent-workspace call "
                  "(H2D, memset, launch, sync, D2H still paid)"),
    "in_call_gpu_window": "GPU pack+matmul window inside the call (events run, GPU time only)",
    "floor": ("GPU pack+matmul on persistent buffers launched right after the L2 flush "
              "(cold: biased high by L2/TLB misses)"),
}
FLOOR_NOTE = ("floor = GPU pack+matmul on persistent buffers launched right after the L2 flush "
              "(cold: biased high by L2/TLB misses); it excludes H2D, D2H, sync wake-up and host "
              "launch cost. The in-call window of the events run is the unbiased GPU time.")
TOKEN_NOTE = ("Per-call times are right-skewed, so the mean-based column is the expected "
              "per-token cost and the median-based column understates it.")


def us(value):
    return f"{value / 1000:.1f}"


def markdown(result):
    lines = ["# P6-M1 summary", "", f"Calls per token: {result['calls_per_token']}", ""]
    for mode, data in result["modes"].items():
        lines += [f"## {mode}: per call, microseconds (median / p90)", ""]
        header = ["shape", "n", "total"] + [p[:-3] for p in PHASES] + ["other", "floor"]
        lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        for name, stats in data["shapes"].items():
            cells = [name, str(stats["total_ns"]["n"])]
            for field in ("total_ns",) + PHASES + ("other_ns", "floor_ns"):
                cells.append(f"{us(stats[field]['median'])} / {us(stats[field]['p90'])}")
            lines.append("| " + " | ".join(cells) + " |")
        lines += ["", FLOOR_NOTE]
        lines += ["", f"## {mode}: derived, microseconds (median / p90)", ""]
        lines += ["| shape | malloc x3 | free x3 | alloc+free | rest (total - alloc+free) | alloc+free share |",
                  "|---|---|---|---|---|---|"]
        for name, stats in data["shapes"].items():
            share = stats["alloc_free_ns"]["median"] / stats["total_ns"]["median"]
            cells = [name]
            for field in ("malloc_ns", "free_ns", "alloc_free_ns", "non_alloc_ns"):
                cells.append(f"{us(stats[field]['median'])} / {us(stats[field]['p90'])}")
            cells.append(f"{share * 100:.0f}%")
            lines.append("| " + " | ".join(cells) + " |")
        lines += ["", f"## {mode}: GPU windows from the events run, microseconds (median / p90)", ""]
        lines += ["| shape | memset window | pack+matmul window | host total of events run |", "|---|---|---|---|"]
        for name, stats in data["events_run"].items():
            cells = [name]
            for field in ("memset_win_ns", "kernel_win_ns", "total_ns"):
                cells.append(f"{us(stats[field]['median'])} / {us(stats[field]['p90'])}")
            lines.append("| " + " | ".join(cells) + " |")
        lines += ["", f"## {mode}: implied cost per token, milliseconds", ""]
        lines += [f"| quantity | {' | '.join(TOKEN_COLUMNS.values())} |", "|---|---|---|---|"]
        for key, label in TOKEN_ROWS.items():
            values = data["per_token_ms"][key]
            cells = " | ".join(f"{values[stat_name]:.2f}" for stat_name in TOKEN_COLUMNS)
            lines.append(f"| {label} | {cells} |")
        lines += ["", TOKEN_NOTE]
        lines += ["", "| shape | calls per token | median per call (us) | ms per token |", "|---|---|---|---|"]
        for name, count in COUNTS.items():
            med = data["shapes"][name]["total_ns"]["median"]
            lines.append(f"| {name} | {count} | {us(med)} | {count * med / 1e6:.2f} |")
        lines.append("")
    return "\n".join(lines)


def main(argv):
    if len(argv) != 2:
        raise SystemExit("usage: summarize.py <results-dir>")
    results_dir = pathlib.Path(argv[1])
    result = summarize(results_dir)
    (results_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (results_dir / "summary.md").write_text(markdown(result))
    print(f"wrote {results_dir / 'summary.json'} and {results_dir / 'summary.md'}")


if __name__ == "__main__":
    main(sys.argv)
