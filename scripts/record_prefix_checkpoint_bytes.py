#!/usr/bin/env python3
"""Record committed prefix checkpoint bytes for a completed prefix-return benchmark."""

import argparse
import json
from pathlib import Path

from bench_engines import nonnegative_int
from bench_long_context import matching_server_metrics, write_json


TURN_ORDER = ("cold_a", "adjacent_a", "competing_b", "return_a")


def benchmark_turns(record):
    if record["state"] != "complete":
        raise ValueError("the benchmark did not complete")
    depths = record["depths"]
    if [depth["context_tokens"] for depth in depths] != record["contexts"]:
        raise ValueError("the benchmark did not report every requested depth")
    measured = []
    for depth in depths:
        if depth["state"] == "capacity_refused":
            continue
        turns = depth.get("turns", [])
        if depth["state"] != "passed" or tuple(t["turn"] for t in turns) != TURN_ORDER:
            raise ValueError(f"incomplete results at {depth['context_tokens']} tokens")
        for turn in turns:
            if turn["recall_passed"] is not True or turn["reuse_passed"] is not True:
                raise ValueError("a benchmark turn failed recall or reuse")
            measured.append((depth["context_tokens"], turn))
    ids = [turn["request_id"] for _, turn in measured]
    if not ids or any(not isinstance(rid, str) or not rid for rid in ids):
        raise ValueError("no measured requests with valid IDs")
    if len(set(ids)) != len(ids):
        raise ValueError("benchmark request IDs are not unique")
    return measured


def record_checkpoint_bytes(benchmark, request_log):
    measured = benchmark_turns(benchmark)
    by_id = {turn["request_id"]: [] for _, turn in measured}
    model = benchmark["model"]
    for line in request_log:
        # The service journal can include startup and admission diagnostics.
        if not line.lstrip().startswith("{"):
            continue
        row = json.loads(line)
        if row.get("endpoint") == "/v1/chat/completions" and row.get("model") == model:
            rows = by_id.get(row.get("request_id"))
            if rows is not None:
                rows.append(row)

    samples = []
    usage_keys = ("prompt_tokens", "completion_tokens", "reused_tokens")
    for context, turn in measured:
        rid = turn["request_id"]
        usage = {key: turn[key] for key in usage_keys}
        stream = dict(usage=usage, request_id=rid, finish_reason=turn["finish_reason"])
        rows = by_id[rid]
        if matching_server_metrics(rows, model, stream) is None:
            raise ValueError(f"{rid} has no unique matching completed chat log record")
        row = rows[0]
        entries, size = row.get("prefix_entries"), row.get("prefix_bytes")
        if not nonnegative_int(entries) or not nonnegative_int(size):
            raise ValueError(f"{rid} is missing valid committed checkpoint counters")
        samples.append({
            "context_tokens": context,
            "turn": turn["turn"],
            "request_id": rid,
            **usage,
            "prefix_entries": entries,
            "prefix_bytes": size,
        })

    return {
        "model": model,
        "expect_return": benchmark["expect_return"],
        "server_context_limit": benchmark["server_context_limit"],
        "capacity_refused_contexts": [
            depth["context_tokens"]
            for depth in benchmark["depths"]
            if depth["state"] == "capacity_refused"
        ],
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--request-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
        with args.request_log.open(encoding="utf-8") as log:
            result = record_checkpoint_bytes(benchmark, log)
        write_json(args.output, result, create=True)
    except (OSError, TypeError, ValueError, KeyError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    main()
