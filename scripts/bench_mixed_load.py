#!/usr/bin/env python3
"""Measure two overlapping chat clients on a running FiFi endpoint."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
from threading import Event
import time

import bench_engines as bench
import bench_long_context as common


@dataclass
class Prompt:
    messages: list
    tokens: int
    budget: int
    fingerprint: str


def prompt(client, model, messages, budget):
    encoded = client.json("/v1/token/encode", {"model": model, "messages": messages,
                                                "x-thinking": "off"})
    ids = encoded["tokens"]
    if not isinstance(ids, list) or len(ids) != encoded["length"] or not ids:
        raise ValueError("invalid tokenized fixture")
    digest = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
    return Prompt(messages, len(ids), budget, digest)


def make_pair(client, model, depth, seed):
    rng = random.Random(seed)
    codes = tuple(str(n) for n in rng.sample(range(100000, 1000000), 3))
    answer = "|".join(codes)
    cold, _, _ = common.size_prompt(client, model, depth - 192, 128, seed, codes)
    short, _, _ = common.size_prompt(client, model, 640, 64, seed + "-count", codes)
    conversation = common.messages(short) + [
        {"role": "assistant", "content": answer},
        {"role": "user", "content": "Count from 1 to 200, one number per line, without skipping numbers."},
    ]
    return (prompt(client, model, common.messages(cold), 64),
            prompt(client, model, conversation, 128), answer)


def timed_stream(client, model, item, notify):
    body = {"model": model, "messages": item.messages, "max_tokens": item.budget,
            "temperature": 0, "stream": True, "x-thinking": "off"}
    fragments, arrivals = [], []
    start = time.monotonic()
    with client.request("/v1/chat/completions", body) as response:
        headers = time.monotonic()
        notify("admitted")

        def record(content):
            arrivals.append(time.monotonic())
            fragments.append(content)
            if len(arrivals) == 1:
                notify("visible")

        stream = bench.parse_stream(response, start, on_content=record)
        end = time.monotonic()
    usage = stream["usage"]
    if not arrivals or usage["prompt_tokens"] != item.tokens or not 1 <= usage["completion_tokens"] <= item.budget:
        raise ValueError("missing visible output or invalid usage for this fixture")
    return dict(stream=stream, pieces=fragments, times=arrivals, start=start,
                admitted=headers, end=end)


def run_pair(client, model, prompts, first_name):
    second_name = "cold" if first_name == "count" else "count"
    phase = "visible" if first_name == "count" else "admitted"
    future = None
    with ThreadPoolExecutor(max_workers=1) as pool:
        def notify(event):
            nonlocal future
            if event == phase and future is None:
                future = pool.submit(timed_stream, client, model, prompts[second_name], lambda _: None)

        first = timed_stream(client, model, prompts[first_name], notify)
        if future is None:
            raise ValueError("first client did not reach the launch phase")
        second = future.result()
    begin = first["times"][0] if first_name == "count" else first["admitted"]
    cutoff = first["times"][-1] if first_name == "count" else first["times"][0]
    if not begin <= second["start"] <= second["admitted"] < cutoff:
        raise ValueError("second request was not admitted within the intended overlap interval")
    return {first_name: first, second_name: second}


def checked_reuse(usage):
    reused = bench.reused_tokens(usage)
    if reused is None:
        raise ValueError("missing cached-token count")
    return reused


def p95(values):
    return sorted(values)[math.ceil(0.95 * len(values)) - 1] if len(values) >= 20 else None


def summary(client, model, result, item, reference):
    stream, times = result["stream"], result["times"]
    usage = stream["usage"]
    metrics = result.get("metrics")
    if metrics is None:
        metrics = common.recent_server_metrics(client, model, stream)
    if metrics["server_queue_wait_ms"] is None:
        raise ValueError("no matching server queue measurement for " + stream["request_id"])
    def offsets(t):
        return round((t - reference) * 1000, 3)

    reused = checked_reuse(usage)
    fresh = item.tokens - reused
    prefill_ms = metrics["server_prefill_ms"]
    gaps = [(b - a) * 1000 for a, b in zip(times, times[1:])]
    return {"id": stream["request_id"], "prompt_sha256": item.fingerprint,
            "prompt_tokens": item.tokens, "reused_tokens": reused, "fresh_tokens": fresh,
            "completion_tokens": usage["completion_tokens"], "finish": stream["finish_reason"],
            "reply_sha256": hashlib.sha256("".join(result["pieces"]).encode()).hexdigest(),
            "submit_ms": offsets(result["start"]), "admitted_ms": offsets(result["admitted"]),
            "first_visible_ms": offsets(times[0]), "last_visible_ms": offsets(times[-1]),
            "done_ms": offsets(result["end"]), "ttft_ms": (times[0] - result["start"]) * 1000,
            "visible_gaps": len(gaps), "p95_visible_gap_ms": p95(gaps),
            "server_time_to_first_token_ms": metrics.get("server_time_to_first_token_ms"),
            "server_queue_wait_ms": metrics["server_queue_wait_ms"], "server_prefill_ms": prefill_ms,
            "fresh_prefill_tokens_per_second": fresh * 1000 / prefill_ms if prefill_ms else None,
            "server_decode_tokens_per_second": metrics["server_decode_tokens_per_second"]}


def valid_count(result):
    text = "".join(result["pieces"])
    expected = "\n".join(map(str, range(1, 201)))
    return (len(text.splitlines()) >= 20 and expected.startswith(text)
            and result["stream"]["usage"]["completion_tokens"] == 128
            and result["stream"]["finish_reason"] == "length")


def prepared_control(client, model, depth, seed):
    cold, count, answer = make_pair(client, model, depth, seed)
    if max(cold.tokens + cold.budget, count.tokens + count.budget) > depth:
        raise ValueError("fixtures exceed the requested context budget")
    control = timed_stream(client, model, count, lambda _: None)
    if not valid_count(control):
        raise ValueError("the serial counting control did not produce the expected sequence")
    control["metrics"] = common.recent_server_metrics(client, model, control["stream"])
    return cold, count, answer, control


def measure(client, model, depth, seed, first):
    cold, count, answer, control = prepared_control(client, model, depth, seed)
    prompts = {"cold": cold, "count": count}
    results = run_pair(client, model, prompts, first)
    recall, counting = results["cold"], results["count"]
    if ("".join(recall["pieces"]).strip() != answer or recall["stream"]["finish_reason"] != "stop"
            or not valid_count(counting) or "".join(counting["pieces"]) != "".join(control["pieces"])):
        raise ValueError("concurrent output failed the recall or serial counting control")
    origin = results[first]["start"]
    clients = {name: summary(client, model, result, prompts[name], origin)
               for name, result in results.items()}
    if len({r["stream"]["request_id"] for r in (control, recall, counting)}) != 3:
        raise ValueError("expected a distinct response ID for each request")
    if clients["cold"]["reused_tokens"] or (first == "count" and not clients["count"]["reused_tokens"]):
        raise ValueError("measured cache state did not match the workload")
    duration = max(row["end"] for row in results.values()) - origin
    return {"context": depth, "first": first, "seed": seed, "clients": clients,
            "serial_count_control": summary(client, model, control, count, control["start"]),
            "pair_seconds": duration,
            "aggregate_output_tokens_per_second": sum(r["completion_tokens"] for r in clients.values()) / duration}


def run_backlog(client, model, cold, count, max_requests):
    """Start cold work while one primed client streams and a second has been admitted."""
    cold_done = Event()
    cold_future = second_future = None

    def cold_request():
        try:
            result = timed_stream(client, model, cold, lambda _: None)
        finally:
            cold_done.set()
        result["metrics"] = common.recent_server_metrics(client, model, result["stream"])
        return result

    def sequence(signal):
        results = []
        for _ in range(max_requests):
            result = timed_stream(client, model, count, signal if not results else lambda _: None)
            result["metrics"] = common.recent_server_metrics(client, model, result["stream"])
            results.append(result)
            if cold_done.is_set():
                break
        return results

    with ThreadPoolExecutor(max_workers=2) as pool:
        def submit_cold(phase):
            nonlocal cold_future
            if phase == "admitted":
                cold_future = pool.submit(cold_request)

        def submit_second(phase):
            nonlocal second_future
            if phase == "visible":
                second_future = pool.submit(sequence, submit_cold)

        first = sequence(submit_second)
        if second_future is None:
            raise ValueError("first counting client never streamed")
        second = second_future.result()
        if cold_future is None:
            raise ValueError("second counting client was never admitted")
        return first, second, cold_future.result()


def warm_backlog(client, model, depth, seed, max_requests):
    cold, count, answer, control = prepared_control(client, model, depth, seed)
    first, second, recall = run_backlog(client, model, cold, count, max_requests)
    if not (first[0]["times"][0] <= second[0]["admitted"] <= recall["start"]
            <= recall["admitted"] < first[0]["times"][-1]):
        raise ValueError("cold request missed the initial warm backlog")
    if ("".join(recall["pieces"]).strip() != answer or recall["stream"]["finish_reason"] != "stop"
            or checked_reuse(recall["stream"]["usage"]) != 0):
        raise ValueError("cold request failed recall or reused a prefix")
    expected = "".join(control["pieces"])
    if not all(valid_count(r) and "".join(r["pieces"]) == expected for r in first + second):
        raise ValueError("a counting client diverged from the serial control")
    if not checked_reuse(first[0]["stream"]["usage"]):
        raise ValueError("first counting client did not reuse its primed prefix")
    all_results = [control, recall, *first, *second]
    if len({r["stream"]["request_id"] for r in all_results}) != len(all_results):
        raise ValueError("expected a distinct response ID for each request")

    origin = first[0]["start"]
    clients = {name: [summary(client, model, r, count, origin) for r in results]
               for name, results in [("first", first), ("second", second)]}
    measured = [recall, *first, *second]
    duration = max(r["end"] for r in measured) - origin
    tokens = sum(r["stream"]["usage"]["completion_tokens"] for r in measured)
    return {"context": depth, "seed": seed, "mode": "warm_backlog",
            "workload_seconds": duration, "aggregate_output_tokens_per_second": tokens / duration,
            "max_requests_per_counting_client": max_requests,
            "serial_count_control": summary(client, model, control, count, control["start"]),
            "cold": summary(client, model, recall, cold, origin), "counting_clients": clients,
            **backlog_overlap(recall, first + second)}


def backlog_overlap(cold, warm):
    """Union of admitted warm intervals while the cold request was outstanding."""
    begin, end = cold["admitted"], cold["end"]
    intervals = sorted((max(begin, r["admitted"]), min(end, r["end"])) for r in warm)
    covered, cursor = 0.0, begin
    for start, stop in intervals:
        if stop > max(start, cursor):
            covered += stop - max(start, cursor)
            cursor = stop
    return {"warm_admitted_during_cold": sum(begin < r["admitted"] < end for r in warm),
            "warm_completed_during_cold": sum(begin < r["end"] < end for r in warm),
            "cold_interval_with_warm_outstanding_fraction": covered / (end - begin)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--contexts", type=int, nargs="+", required=True)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--warm-backlog", type=int, default=0, metavar="REQUESTS",
                        help="run two counting clients until cold completes, up to REQUESTS per client")
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--api-key-env")
    parser.add_argument("--output", type=Path, required=True, help="new JSONL file, one row per workload")
    args = parser.parse_args()
    if args.warm_backlog < 0 or args.warm_backlog == 1:
        parser.error("--warm-backlog requires at least two requests per counting client")
    if not args.seed or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("use a nonempty seed and a positive finite timeout")
    if args.repetitions < 1 or min(args.contexts) < 2048 or len(set(args.contexts)) != len(args.contexts):
        parser.error("use positive repetitions and distinct context budgets of at least 2048")
    client = common.Client(args.url, os.environ[args.api_key_env] if args.api_key_env else None, args.timeout)
    limit = common.context_limit(client, args.model)
    with args.output.open("x", buffering=1) as out:
        for depth in args.contexts:
            if depth > limit:
                out.write(json.dumps({"model": args.model, "context": depth, "server_limit": limit,
                                      "state": "above_advertised_limit"}) + "\n")
                continue
            for i in range(args.repetitions):
                if args.warm_backlog:
                    seed = f"{args.seed}/{depth}/{i}/backlog"
                    row = warm_backlog(client, args.model, depth, seed, args.warm_backlog)
                    out.write(json.dumps({"model": args.model, **row}, allow_nan=False) + "\n")
                    continue
                for first in (("count", "cold") if i % 2 == 0 else ("cold", "count")):
                    seed = f"{args.seed}/{depth}/{i}/{first}"
                    row = measure(client, args.model, depth, seed, first)
                    out.write(json.dumps({"model": args.model, **row}, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
