#!/usr/bin/env python3
"""Measure cold A, adjacent A, competing B, and returning A on a running FiFi server."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile

import bench_long_context as recall


def conversation(prompt):
    return [
        *recall.messages(prompt),
        {"role": "assistant", "content": "I have the archive."},
        {"role": "user", "content": "Give the three requested codes in order."},
    ]


def fixture(client, args, target, nonce, values):
    prompt, count, _ = recall.size_prompt(
        client, args.model, target, args.tolerance, nonce, values, render=conversation
    )
    ids = client.encode_ids(
        args.model, messages=conversation(prompt), **{"x-thinking": "off"}
    )
    if len(ids) != count:
        raise ValueError("the token count changed after sizing the conversation")
    digest = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
    return prompt, ids, digest, "|".join(values)


def run_depth(client, args, row, limit, request_ids, save):
    depth = row["context_tokens"]
    # Reserve the tokenizer tolerance as well as the reply at the largest admitted depth.
    target = depth - args.max_tokens - args.tolerance
    rng = random.Random(f"{args.seed}/{depth}")
    values = [str(n) for n in rng.sample(range(100_000, 1_000_000), 6)]
    a, b = [
        fixture(client, args, target, f"{rng.getrandbits(96):024x}", codes)
        for codes in (values[:3], values[3:])
    ]
    shared = next(
        (i for i, (x, y) in enumerate(zip(a[1], b[1])) if x != y),
        min(len(a[1]), len(b[1])),
    )
    if shared == min(len(a[1]), len(b[1])):
        raise ValueError("the tokenized conversations have no independent branches")
    row.update(common_prefix_tokens=shared, turns=[])
    save()
    adjacent = 0
    for label, (prompt, ids, digest, answer) in [
        ("cold_a", a),
        ("adjacent_a", a),
        ("competing_b", b),
        ("return_a", a),
    ]:
        if recall.context_limit(client, args.model) != limit:
            raise ValueError("the serving context limit changed during the comparison")
        result = recall.measure(
            client,
            args.model,
            prompt,
            len(ids),
            limit,
            args.max_tokens,
            answer,
            render=conversation,
            require_cold=False,
        )
        reused = result["reused_tokens"]
        if label == "cold_a":
            correct_reuse = reused == 0
        elif label == "adjacent_a":
            adjacent = reused
            correct_reuse = adjacent > shared
        elif label == "competing_b" or args.expect_return == "evicted":
            correct_reuse = reused <= shared
        else:
            correct_reuse = reused >= adjacent
        fresh = len(ids) - reused
        ms = result["server_prefill_ms"]
        del (
            result["estimated_prompt_tokens_per_second"],
            result["server_prompt_tokens_per_second"],
        )
        result.update(
            client_new_prompt_tokens_per_second_estimate=fresh
            / result["first_visible_seconds"],
            turn=label,
            prompt_ids_sha256=digest,
            new_prompt_tokens=fresh,
            server_new_prompt_tokens_per_second=1000 * fresh / ms if ms else None,
            reuse_passed=correct_reuse,
        )
        row["turns"].append(result)
        rid = result["request_id"]
        if rid in request_ids:
            raise ValueError("the server repeated a request ID during the benchmark")
        request_ids.add(rid)
        save()
        if not result["recall_passed"] or not correct_reuse:
            row["state"] = "failed"
            return
    row["state"] = (
        "passed"
        if all(
            t["server_time_to_first_token_ms"] is not None
            and t["server_decode_tokens_per_second"] is not None
            for t in row["turns"]
        )
        else "timing_incomplete"
    )


def run(args):
    if (
        args.tolerance < 0
        or args.max_tokens <= 0
        or not args.seed
        or not math.isfinite(args.timeout)
        or args.timeout <= 0
        or len(set(args.contexts)) != len(args.contexts)
        or any(n <= args.max_tokens + args.tolerance for n in args.contexts)
    ):
        raise ValueError(
            "invalid context sizes, reply reservation, tolerance, seed or timeout"
        )
    key = os.environ.get(args.api_key_env) if args.api_key_env else None
    if args.api_key_env and not key:
        raise ValueError("the requested API key environment variable is empty")
    client = recall.Client(args.url, key, args.timeout)
    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + ".partial")
    if out.exists():
        raise FileExistsError("the output already exists")
    record = {
        "state": "running",
        "model": args.model,
        "seed": args.seed,
        "expect_return": args.expect_return,
        "contexts": args.contexts,
        "max_tokens": args.max_tokens,
        "tolerance": args.tolerance,
        "depths": [],
    }
    recall.write_json(partial, record, create=True)

    def save():
        with tempfile.TemporaryDirectory(dir=out.parent) as directory:
            tmp = Path(directory) / "record.json"
            recall.write_json(tmp, record)
            tmp.replace(partial)

    try:
        limit = recall.context_limit(client, args.model)
        record["server_context_limit"] = limit
        request_ids = set()
        for depth in args.contexts:
            row = {"context_tokens": depth, "state": "running"}
            record["depths"].append(row)
            if depth > limit:
                row["state"] = "capacity_refused"
            else:
                run_depth(client, args, row, limit, request_ids, save)
            save()
            if row["state"] == "failed":
                break
        record["state"] = "complete"
        record["passed"] = all(d["state"] == "passed" for d in record["depths"])
        save()
        os.link(partial, out)
        partial.unlink()
        return record["passed"]
    except Exception as error:
        record.update(state="failed", error=str(error))
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18792")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--expect-return", choices=["retained", "evicted"], required=True
    )
    parser.add_argument(
        "--seed", required=True, help="use the same seed on both configurations"
    )
    parser.add_argument(
        "--contexts", nargs="+", type=int, default=[2048 << n for n in range(8)]
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--tolerance", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=21600)
    parser.add_argument("--api-key-env")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        return 0 if run(args) else 2
    except (OSError, ValueError, RuntimeError) as error:
        print(f"prefix return benchmark failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
