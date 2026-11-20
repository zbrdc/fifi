#!/usr/bin/env python3
"""Measure a cold synthetic recall request at a server-counted prompt length.

Example (run only in an admitted, isolated device window):
  python3 scripts/bench_long_context.py --model fifi-q36 --prompt-tokens 131072 --output /path/to/scratch/recall-128k.json

The prompt and reply share the server's context limit. Reserve reply space with
--max-tokens. This is a synthetic recall check, not a coding-quality evaluation.
The prefill rate is an end-to-end estimate to the first visible content; it
includes tokenization, queueing, and response overhead. Server timing and decode
rate are reported only when a matching completed request record is available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import sys
import time
from urllib.parse import urlsplit
import urllib.request

import bench_engines as bench

FILLER = "The sky is blue. A cool wind crosses the empty stone valley.\n"
NAMES = ("MAPLE", "COBALT", "FJORD")


def prompt_text(repeats: int, nonce: str, values: tuple[str, str, str]) -> str:
    if repeats < 4 or not nonce or len(values) != len(NAMES):
        raise ValueError("a nonce, three values, and at least four repeats are required")
    a, b, c = repeats // 10, repeats * 4 // 10, repeats * 4 // 10
    d = repeats - a - b - c
    return (
        f"Test case {nonce}. Remember the three archive markers below.\n"
        + FILLER * a + f"\nARCHIVE MARKER {NAMES[0]}: {values[0]}.\n"
        + FILLER * b + f"\nARCHIVE MARKER {NAMES[1]}: {values[1]}.\n"
        + FILLER * c + f"\nARCHIVE MARKER {NAMES[2]}: {values[2]}.\n"
        + FILLER * d
        + "\nReturn only the codes for MAPLE, COBALT, FJORD in that order, "
          "separated by vertical bars, with no other text.\n"
    )


def messages(prompt: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt}]


class Client:
    def __init__(self, base: str, key: str | None, timeout: float):
        parts = urlsplit(base)
        if (parts.scheme not in ("http", "https") or not parts.netloc
                or parts.username or parts.password or parts.query or parts.fragment):
            raise ValueError("--url must be an HTTP(S) base URL without credentials or a query")
        self.base = base.rstrip("/")
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}
        if key:
            self.headers["Authorization"] = "Bearer " + key

    def request(self, path: str, body: dict | None = None, *, timeout: float | None = None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, headers=self.headers)
        return urllib.request.urlopen(req, timeout=min(self.timeout, timeout) if timeout else self.timeout)

    def json(self, path: str, body: dict | None = None) -> dict:
        timeout = 180 if path == "/v1/token/encode" else 5
        with self.request(path, body, timeout=timeout) as response:
            value = json.load(response)
        if not isinstance(value, dict):
            raise ValueError(f"{path} did not return an object")
        return value

    def encode_ids(self, model: str, **source) -> list[int]:
        value = self.json("/v1/token/encode", {"model": model, **source})
        count, ids = value.get("length"), value.get("tokens")
        if not bench.nonnegative_int(count) or not isinstance(ids, list) or len(ids) != count:
            raise ValueError("server tokenization returned an inconsistent count")
        if any(not bench.nonnegative_int(token) or token > 0xffffffff for token in ids):
            raise ValueError("server tokenization returned an invalid token ID")
        return ids

    def encode(self, model: str, **source) -> int:
        return len(self.encode_ids(model, **source))


def size_prompt(client: Client, model: str, target: int, tolerance: int,
                nonce: str, values: tuple[str, str, str], *, render=messages) -> tuple[str, int, int]:
    per = client.encode(model, text=FILLER * 3) - client.encode(model, text=FILLER * 2)
    if per <= 0:
        raise ValueError("the server tokenizer did not advance on the filler")
    repeats = max(4, target // per)
    visited = set()
    for _ in range(12):
        if repeats in visited:
            break
        visited.add(repeats)
        prompt = prompt_text(repeats, nonce, values)
        count = client.encode(model, messages=render(prompt), **{"x-thinking": "off"})
        if target <= count <= target + tolerance:
            return prompt, count, repeats
        change = max(1, abs(target - count) // per)
        repeats = max(4, repeats + (change if count < target else -change))
    raise ValueError("could not reach the requested prompt range; try a larger tolerance")


def context_limit(client: Client, model: str) -> int:
    data = client.json("/v1/system")
    engines = data.get("engines")
    if not isinstance(engines, list):
        raise ValueError("/v1/system did not return an engine list")
    entries = [entry for entry in engines
               if isinstance(entry, dict) and entry.get("model") == model]
    if len(entries) != 1 or entries[0].get("state") != "ready":
        raise ValueError("the requested model is not uniquely ready in /v1/system")
    limit = entries[0].get("context_limit")
    if not bench.nonnegative_int(limit) or limit == 0:
        raise ValueError("the server did not report a valid context limit")
    return limit


def matching_server_metrics(rows, model: str, stream: dict) -> dict | None:
    """Accept timings only from the unique completed record for this response."""
    if not isinstance(rows, list):
        return None
    matches = [row for row in rows if isinstance(row, dict)
               and row.get("request_id") == stream["request_id"]
               and row.get("model") == model]
    if len(matches) != 1:
        return None
    row, usage = matches[0], stream["usage"]
    if (row.get("status") != 200 or row.get("outcome") != "ok"
            or row.get("stream") is not True
            or row.get("finish_reason") != stream["finish_reason"]
            or not bench.nonnegative_int(row.get("prompt_tokens"))
            or not bench.nonnegative_int(row.get("completion_tokens"))
            or not bench.nonnegative_int(row.get("reused_tokens"))
            or row["prompt_tokens"] != usage["prompt_tokens"]
            or row["completion_tokens"] != usage["completion_tokens"]
            or row["reused_tokens"] != bench.reused_tokens(usage)):
        return None

    def milliseconds(value, *, allow_zero: bool) -> float | None:
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero)):
            return None
        return float(value)

    prefill_ms = milliseconds(row.get("prefill_ms"), allow_zero=False)
    return {
        "server_time_to_first_token_ms": milliseconds(row.get("time_to_first_token_ms"), allow_zero=True),
        "server_queue_wait_ms": milliseconds(row.get("queue_wait_ms"), allow_zero=True),
        "server_prefill_ms": prefill_ms,
        "server_prompt_tokens_per_second": (1000 * usage["prompt_tokens"] / prefill_ms
                                             if prefill_ms is not None else None),
        "server_decode_tokens_per_second": bench.matching_server_rate(rows, model, stream),
    }


def recent_server_metrics(client: Client, model: str, stream: dict) -> dict:
    """The request log may arrive just after the final SSE event."""
    deadline = time.monotonic() + 5
    empty = {"server_time_to_first_token_ms": None, "server_queue_wait_ms": None, "server_prefill_ms": None,
             "server_prompt_tokens_per_second": None,
             "server_decode_tokens_per_second": None}
    while True:
        try:
            data = client.json("/v1/recent-requests")
        except (OSError, ValueError):
            return empty
        metrics = matching_server_metrics(data.get("requests"), model, stream)
        if metrics is not None:
            return metrics
        if time.monotonic() >= deadline:
            return empty
        time.sleep(0.1)


def measure(client: Client, model: str, prompt: str, count: int, limit: int,
            max_tokens: int, expected: str, *, render=messages, require_cold=True) -> dict:
    if count + max_tokens > limit:
        raise ValueError("the counted prompt and reserved reply exceed the context limit")
    body = {"model": model, "messages": render(prompt), "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "x-thinking": "off"}
    chunks: list[str] = []
    started = time.monotonic()
    with client.request("/v1/chat/completions", body) as response:
        stream = bench.parse_stream(response, started, on_content=chunks.append)
    usage = stream["usage"]
    if usage["prompt_tokens"] != count or not 1 <= usage["completion_tokens"] <= max_tokens:
        raise ValueError("the completed response usage does not match the counted prompt")
    reused = bench.reused_tokens(usage)
    if require_cold and reused != 0:
        raise ValueError("the server did not confirm a cold prompt with zero reused tokens")
    if reused is None:
        raise ValueError("the server did not report prefix reuse")
    first = stream["first_delta_seconds"]
    if first is None or not math.isfinite(first) or first <= 0:
        raise ValueError("the response did not contain a timed visible delta")
    answer = "".join(chunks).strip()
    passed = stream["finish_reason"] == "stop" and answer == expected
    return {
        "prompt_tokens": count,
        "completion_tokens": usage["completion_tokens"],
        "reused_tokens": reused,
        "finish_reason": stream["finish_reason"],
        "first_visible_seconds": first,
        "estimated_prompt_tokens_per_second": count / first,
        **recent_server_metrics(client, model, stream),
        "request_id": stream["request_id"],
        "expected_answer": expected,
        "answer": answer,
        "recall_passed": passed,
    }


def write_json(path: Path, data: dict, *, create: bool = False) -> None:
    with path.open("x" if create else "w") as out:
        json.dump(data, out, indent=2, allow_nan=False)
        out.write("\n")


def run(args: argparse.Namespace) -> bool:
    if (args.prompt_tokens < 1 or args.max_tokens < 1 or args.tolerance < 0
            or not math.isfinite(args.timeout) or args.timeout <= 0):
        raise ValueError("token counts and timeout must be positive; tolerance cannot be negative")
    key = os.environ.get(args.api_key_env) if args.api_key_env else None
    if args.api_key_env and not key:
        raise ValueError("the requested API key environment variable is empty")
    client = Client(args.url, key, args.timeout)
    output = args.output
    partial = output.with_name(output.name + ".partial")
    if output.exists():
        raise FileExistsError("the output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {"state": "started", "model": args.model, "target_prompt_tokens": args.prompt_tokens,
              "tolerance": args.tolerance, "max_completion_tokens": args.max_tokens}
    write_json(partial, result, create=True)
    try:
        limit = context_limit(client, args.model)
        if args.prompt_tokens + args.max_tokens > limit:
            raise ValueError("the requested prompt and reserved reply exceed the context limit")
        nonce = secrets.token_hex(12)
        values = tuple(str(n) for n in secrets.SystemRandom().sample(range(100_000, 1_000_000), 3))
        prompt, count, repeats = size_prompt(client, args.model, args.prompt_tokens,
                                              args.tolerance, nonce, values)
        result.update({"state": "counted", "context_limit": limit, "prompt_tokens": count,
                       "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                       "filler_repeats": repeats, "nonce": nonce,
                       "expected_answer": "|".join(values)})
        write_json(partial, result)
        if context_limit(client, args.model) != limit:
            raise ValueError("the engine context changed after token counting")
        print(f"Counted {count} prompt tokens; sending the synthetic recall request.", flush=True)
        result.update(measure(client, args.model, prompt, count, limit, args.max_tokens, "|".join(values)))
        result.update({"state": "complete", "measurement_note":
                       "Synthetic cold recall; first-visible time includes request overhead. "
                       "Decode comes only from matching server telemetry. "
                       "This result alone does not verify coding quality, device isolation, or build identity."})
        write_json(partial, result)
        os.link(partial, output)
        partial.unlink()
        return result["recall_passed"]
    except Exception as error:
        result.update({"state": "failed", "error_type": type(error).__name__, "error": str(error)})
        write_json(partial, result)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:18792")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-tokens", required=True, type=int,
                        help="minimum server-counted prompt tokens, including the chat template")
    parser.add_argument("--tolerance", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=64, help="reply space to reserve")
    parser.add_argument("--timeout", type=float, default=21600)
    parser.add_argument("--api-key-env", help="environment variable containing an optional API key")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        passed = run(args)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"long-context check failed: {error}", file=sys.stderr)
        return 1
    print(f"Synthetic recall {'passed' if passed else 'failed'}; evidence: {args.output}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
