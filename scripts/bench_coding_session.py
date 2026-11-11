#!/usr/bin/env python3
"""Check a synthetic two-turn coding session at a server-counted prompt length.

This benchmark does not execute model-generated code, reserve devices, or prove
real-world coding quality. Run only in an admitted and isolated device window.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import sys
import time

import bench_engines as bench
import bench_long_context as common

FOLLOWUP = ("Change your previous replacement to normalize the key to uppercase instead. "
            "Use the same JSON shape and return only the replacement statement.")


def fragment(index: int) -> str:
    return (f"# app/module_{index:06d}.py\n"
            f"def clean_field_{index:06d}(value: str) -> str:\n"
            "    return value.strip()\n\n")


def prompt_text(repeats: int, nonce: str) -> str:
    if repeats < 4 or not nonce or not nonce.isalnum():
        raise ValueError("a nonce and at least four source fragments are required")
    cut = max(1, repeats // 10)
    project = "".join(fragment(i) for i in range(cut))
    project += ("# app/keys.py\n"
                "def normalize_key(value: str) -> str:\n"
                "    return value.strip()\n\n")
    project += "".join(fragment(i) for i in range(cut, repeats))
    return (f"Project snapshot {nonce}. Review the following Python files.\n\n"
            + project
            + "Update app/keys.py so normalize_key removes leading and trailing whitespace "
              "and then normalizes the key to lowercase. Return only a JSON object with "
              'one string field named "replacement" containing the single Python return '
              "statement that replaces the function body. Do not include markdown.\n")


def messages(prompt: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt}]


def size_prompt(client: common.Client, model: str, target: int, tolerance: int,
                nonce: str) -> tuple[str, int, int]:
    per = client.encode(model, text=fragment(2))
    if per <= 0:
        raise ValueError("the server tokenizer did not advance on source fragments")
    repeats = max(4, target // per)
    visited = set()
    for _ in range(20):
        if repeats in visited:
            break
        visited.add(repeats)
        prompt = prompt_text(repeats, nonce)
        count = client.encode(model, messages=messages(prompt), **{"x-thinking": "off"})
        if target <= count <= target + tolerance:
            return prompt, count, repeats
        change = max(1, abs(target - count) // per)
        repeats = max(4, repeats + (change if count < target else -change))
    raise ValueError("could not reach the requested prompt range; try a larger tolerance")


def valid_replacement(answer: str, operation: str) -> bool:
    """Require the requested statement's AST without executing returned code."""
    try:
        value = json.loads(answer)
        if (not isinstance(value, dict) or set(value) != {"replacement"}
                or not isinstance(value["replacement"], str)
                or len(value["replacement"]) > 256):
            return False
        candidate = ast.parse("def _candidate(value):\n    " + value["replacement"].strip() + "\n")
        expected = ast.parse(f"def _candidate(value):\n    return value.strip().{operation}()\n")
        return ast.dump(candidate, include_attributes=False) == ast.dump(expected, include_attributes=False)
    except (SyntaxError, ValueError, TypeError, RecursionError):
        return False


def turn(client: common.Client, model: str, conversation: list[dict[str, str]],
         count: int, max_tokens: int, *, cold: bool) -> dict:
    body = {"model": model, "messages": conversation, "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "x-thinking": "off"}
    chunks: list[str] = []
    started = time.monotonic()
    with client.request("/v1/chat/completions", body) as response:
        def capture():
            for raw in response:
                line = raw.decode("utf-8").strip()
                if line.startswith("data: ") and line[6:] != "[DONE]":
                    obj = json.loads(line[6:])
                    for choice in obj.get("choices") or []:
                        content = (choice.get("delta") or {}).get("content")
                        if content is not None:
                            if not isinstance(content, str):
                                raise ValueError("stream content is not text")
                            chunks.append(content)
                yield raw
        stream = bench.parse_stream(capture(), started)
    usage = stream["usage"]
    if usage["prompt_tokens"] != count or not 1 <= usage["completion_tokens"] <= max_tokens:
        raise ValueError("response usage does not match the counted prompt")
    reused = bench.reused_tokens(usage)
    if reused is None or (cold and reused != 0):
        raise ValueError("the prompt's reuse was not confirmed cold" if cold
                         else "the follow-up's reuse was not reported")
    first = stream["first_delta_seconds"]
    if first is None or not math.isfinite(first) or first <= 0:
        raise ValueError("the response did not contain a timed visible delta")
    telemetry = common.recent_server_metrics(client, model, stream)
    # The common helper's rate uses all prompt tokens and is valid only for a
    # cold request. For a reused prefix, expose the newly processed-token rate.
    telemetry.pop("server_prompt_tokens_per_second", None)
    prefill_ms = telemetry["server_prefill_ms"]
    return {"prompt_tokens": count, "completion_tokens": usage["completion_tokens"],
            "reused_tokens": reused, "finish_reason": stream["finish_reason"],
            "request_id": stream["request_id"], "answer": "".join(chunks).strip(),
            "first_visible_seconds": first,
            "client_uncached_tokens_per_second_estimate": (count - reused) / first,
            **telemetry,
            "server_uncached_tokens_per_second": (
                1000 * (count - reused) / prefill_ms if prefill_ms is not None else None)}


def run(args: argparse.Namespace) -> bool:
    if (args.prompt_tokens < 1 or args.max_tokens < 1 or args.tolerance < 0
            or args.min_reused_tokens < 1 or not math.isfinite(args.timeout) or args.timeout <= 0):
        raise ValueError("invalid token count, reuse threshold, tolerance, or timeout")
    key = os.environ.get(args.api_key_env) if args.api_key_env else None
    if args.api_key_env and not key:
        raise ValueError("the requested API key environment variable is empty")
    client = common.Client(args.url, key, args.timeout)
    output = args.output
    partial = output.with_name(output.name + ".partial")
    if output.exists():
        raise FileExistsError("the output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {"state": "started", "model": args.model, "target_prompt_tokens": args.prompt_tokens,
              "tolerance": args.tolerance, "max_completion_tokens": args.max_tokens,
              "reuse_mode": args.reuse_mode,
              "min_reused_tokens": args.min_reused_tokens if args.reuse_mode == "required" else None,
              "turns": []}
    common.write_json(partial, result, create=True)
    try:
        limit = common.context_limit(client, args.model)
        if args.prompt_tokens + 2 * args.max_tokens > limit:
            raise ValueError("the prompt and two reply reservations exceed the context limit")
        nonce = secrets.token_hex(12)
        prompt, count, repeats = size_prompt(client, args.model, args.prompt_tokens,
                                             args.tolerance, nonce)
        result.update({"state": "counted", "context_limit": limit, "filler_repeats": repeats,
                       "nonce": nonce, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()})
        common.write_json(partial, result)
        if count + 2 * args.max_tokens > limit:
            raise ValueError("the counted prompt and two reply reservations exceed the context limit")
        if common.context_limit(client, args.model) != limit:
            raise ValueError("the engine context changed after token counting")
        print(f"Counted {count} prompt tokens; sending the first coding turn.", flush=True)
        conversation = messages(prompt)
        first = turn(client, args.model, conversation, count, args.max_tokens, cold=True)
        first["edit_passed"] = first["finish_reason"] == "stop" and valid_replacement(first["answer"], "lower")
        result["turns"].append(first)
        common.write_json(partial, result)
        passed = first["edit_passed"]
        if passed:
            conversation += [{"role": "assistant", "content": first["answer"]},
                             {"role": "user", "content": FOLLOWUP}]
            followup_count = client.encode(args.model, messages=conversation, **{"x-thinking": "off"})
            if followup_count + args.max_tokens > limit:
                raise ValueError("the follow-up and reserved reply exceed the context limit")
            if common.context_limit(client, args.model) != limit:
                raise ValueError("the engine context changed before the follow-up")
            print(f"Counted {followup_count} follow-up tokens; sending the revision.", flush=True)
            second = turn(client, args.model, conversation, followup_count, args.max_tokens, cold=False)
            second["edit_passed"] = (second["finish_reason"] == "stop"
                                     and valid_replacement(second["answer"], "upper"))
            second["reuse_passed"] = (second["reused_tokens"] >= args.min_reused_tokens
                                      if args.reuse_mode == "required" else None)
            second["reuse_check_passed"] = (second["reuse_passed"] if args.reuse_mode == "required"
                                            else second["reused_tokens"] == 0)
            result["turns"].append(second)
            passed = second["edit_passed"] and second["reuse_check_passed"]
        result.update({"state": "complete", "passed": passed,
                       "measurement_note": "Synthetic two-turn code edit; AST checked without execution. "
                       "Not general coding quality, device isolation, or build identity."})
        common.write_json(partial, result)
        os.link(partial, output)
        partial.unlink()
        return passed
    except Exception as error:
        result.update({"state": "failed", "error_type": type(error).__name__, "error": str(error)})
        common.write_json(partial, result)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18792")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-tokens", type=int, required=True)
    parser.add_argument("--tolerance", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--reuse-mode", choices=("required", "disabled"), default="required",
                        help="require warm prefix reuse, or verify that reuse is disabled")
    parser.add_argument("--min-reused-tokens", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=21600)
    parser.add_argument("--api-key-env")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        passed = run(args)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"coding-session check failed: {error}", file=sys.stderr)
        return 1
    print(f"Synthetic coding session {'passed' if passed else 'failed'}; evidence: {args.output}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
