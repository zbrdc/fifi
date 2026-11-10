#!/usr/bin/env python3
"""Answer-quality corpus for the chunked-vs-per-token prefill A/B.

Usage:
  bench_corpus_accuracy.py BASE_URL API_KEY MODEL OUT.json     # run the corpus
  bench_corpus_accuracy.py --compare A.json B.json             # diff two runs

Each prompt carries an expected substring (case-insensitive) or None for
free-form prompts a human reviews. A run writes one JSON record per prompt
(answer text, usage, timing); the compare mode prints the two answers side by
side plus pass/agreement counts. Prompts are fixed and deterministic: the same
corpus must produce the same lengths in both engine configs, so a mismatch in
prompt_tokens is itself a finding. The long class (~1k tokens) crosses the
965-row piece boundary of Q36_TIERED_CHUNK=1 — the exact path the
qualification exercised.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request

PARAGRAPH = (
    "The mountain weather shifted through the afternoon: thin cloud over the ridge, "
    "a cold wind funneling up the valley, and by evening the first spindrift off the summit. "
)

# The facts each context prompt asks about, stated once inside the filler.
FACTS = (
    "The beacon keeper this season is named Aldric. The supply crate count at the "
    "high camp is 47, and the rope coils stowed in the lean-to number 13. "
)


def context(tokens: int, fact_at: str) -> str:
    """ filler → FACT → filler ; a PARAGRAPH is ~25 tokens (bench_engines' unit)."""
    paragraphs = tokens // 25
    head = PARAGRAPH * (paragraphs // 2)
    tail = PARAGRAPH * (paragraphs - paragraphs // 2)
    if fact_at == "start":
        return FACTS + head + tail
    if fact_at == "end":
        return head + tail + FACTS
    return head + FACTS + tail


def corpus() -> list[dict]:
    return [
        {
            "id": "short-multiply",
            "cls": "short",
            "prompt": "What is 17 × 23? Reply with only the number.",
            "expect": "391",
            "max_tokens": 16,
        },
        {
            "id": "short-leap",
            "cls": "short",
            "prompt": "How many days are in a leap year? Reply with only the number.",
            "expect": "366",
            "max_tokens": 16,
        },
        {
            "id": "short-speed",
            "cls": "short",
            "prompt": "A train travels 60 km in 45 minutes. "
            "What is its average speed in km/h? Reply with only the number.",
            "expect": "80",
            "max_tokens": 16,
        },
        {
            "id": "medium-retrieval",
            "cls": "medium",
            "prompt": context(350, "middle")
            + "Who is the beacon keeper this season? Reply with only the name.",
            "expect": "aldric",
            "max_tokens": 16,
        },
        {
            "id": "medium-arithmetic",
            "cls": "medium",
            "prompt": context(350, "start")
            + "How many supply items does the high camp hold in total "
            "(crates plus rope coils)? Reply with only the number.",
            "expect": "60",
            "max_tokens": 16,
        },
        {
            "id": "medium-freeform",
            "cls": "medium",
            "prompt": context(350, "end")
            + "In two sentences, summarize the weather the passage describes.",
            "expect": None,
            "max_tokens": 96,
        },
        {
            "id": "long-mid-fact",
            "cls": "long",
            "prompt": context(1000, "middle")
            + "Who is the beacon keeper this season? Reply with only the name.",
            "expect": "aldric",
            "max_tokens": 16,
        },
        {
            "id": "long-start-fact",
            "cls": "long",
            "prompt": context(1000, "start")
            + "How many rope coils are stowed in the lean-to? Reply with only the number.",
            "expect": "13",
            "max_tokens": 16,
        },
        {
            "id": "long-freeform",
            "cls": "long",
            "prompt": context(1000, "end")
            + "Write one sentence naming the two weather details the passage repeats most.",
            "expect": None,
            "max_tokens": 96,
        },
    ]


def ask(base: str, headers: dict, model: str, item: dict) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": item["prompt"]}],
        "temperature": 0,
        "x-thinking": "off",
        "max_tokens": item["max_tokens"],
    }
    started = time.monotonic()
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        json.dumps(body).encode(),
        headers=headers,
    )
    with urllib.request.urlopen(req, timeout=600) as response:
        payload = json.load(response)
    answer = ""
    for choice in payload.get("choices") or []:
        message = choice.get("message") or {}
        answer = message.get("content") or choice.get("text") or ""
        if answer:
            break
    usage = payload.get("usage") or {}
    expect = item["expect"]
    low = answer.lower()
    return {
        "id": item["id"],
        "cls": item["cls"],
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "finish_reason": (payload.get("choices") or [{}])[0].get("finish_reason"),
        "answer": answer,
        "expect": expect,
        "pass": None if expect is None else expect in low,
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }


def run(base: str, key: str, model: str, out_path: str) -> None:
    headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
    records = [ask(base, headers, model, item) for item in corpus()]
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=1, ensure_ascii=False)
    checked = [r for r in records if r["pass"] is not None]
    passed = sum(1 for r in checked if r["pass"])
    print(f"corpus: {passed}/{len(checked)} factual checks passed "
          f"({len(records) - len(checked)} free-form, in {out_path})")
    for record in records:
        mark = {True: "ok", False: "FAIL", None: "review"}[record["pass"]]
        print(f"  {mark:>6} {record['id']:<20} {record['prompt_tokens']} tok "
              f"-> {record['answer'][:60]!r}")


def compare(a_path: str, b_path: str) -> None:
    with open(a_path, encoding="utf-8") as handle:
        a = {r["id"]: r for r in json.load(handle)}
    with open(b_path, encoding="utf-8") as handle:
        b = {r["id"]: r for r in json.load(handle)}
    if set(a) != set(b):
        sys.exit("the two runs hold different prompt ids")
    same_tokens = same_answer = 0
    for pid in a:
        ra, rb = a[pid], b[pid]
        if ra["prompt_tokens"] == rb["prompt_tokens"]:
            same_tokens += 1
        else:
            print(f"  TOKENS DIFFER {pid}: {ra['prompt_tokens']} vs {rb['prompt_tokens']}")
        if ra["answer"].strip() == rb["answer"].strip():
            same_answer += 1
            continue
        print(f"  {pid} ({ra['cls']}):")
        print(f"    A: {ra['answer'][:100]!r} pass={ra['pass']}")
        print(f"    B: {rb['answer'][:100]!r} pass={rb['pass']}")
    print(f"compare: {same_tokens}/{len(a)} same prompt tokens, "
          f"{same_answer}/{len(a)} identical answers "
          f"(divergence is expected across kernel paths; the corpus pass "
          f"counts are the merge signal)")


def main() -> None:
    if len(sys.argv) == 4 and sys.argv[1] == "--compare":
        compare(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 5:
        run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
