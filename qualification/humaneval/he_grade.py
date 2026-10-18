"""HumanEval grader: official protocol (prompt + completion + test +
check(entry_point)), subprocess per problem, pass@1 (greedy)."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def extract_block(text):
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if not blocks:
        return None
    return blocks[-1].strip()


def final_text(text):
    if "<channel|>" in text:
        return text.rsplit("<channel|>", 1)[1]
    if "<|channel>" in text:
        return text.rsplit("<|channel>", 1)[1]
    return text


def passes(problem, raw_response):
    code = extract_block(final_text(raw_response))
    if code is None:
        return False, "no-block"
    if f"def {problem['entry_point']}(" not in code:
        return False, "no-def"
    program = (problem["prompt"] + "\n" + code + "\n" +
               problem["test"] + f"\ncheck({problem['entry_point']})\n"
               "print('PASSED')\n")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.py"
        p.write_text(program)
        try:
            r = subprocess.run(
                [sys.executable, "-I", str(p)], capture_output=True,
                text=True, timeout=10)
        except subprocess.TimeoutExpired:
            return False, "timeout"
        if r.returncode == 0 and "PASSED" in r.stdout:
            return True, "ok"
        return False, f"fail:{r.returncode}"


REQUIRED_FIELDS = ("task_id", "prompt", "test", "entry_point")
EXPECTED_PROBLEMS = 164


def fail(msg):
    sys.exit(f"he_grade: error: {msg}")


def read_jsonl(path, what):
    """Return a list of (physical line number, object) pairs."""
    try:
        with open(path) as f:
            lines = f.readlines()
    except (OSError, UnicodeDecodeError) as e:
        fail(f"cannot read {what} file {path}: {e}")
    rows = []
    for k, l in enumerate(lines, 1):
        if not l.strip():
            continue
        try:
            row = json.loads(l)
        except ValueError as e:
            fail(f"{what} file line {k} is not valid JSON: {e}")
        if not isinstance(row, dict):
            fail(f"{what} file line {k} is not a JSON object")
        rows.append((k, row))
    return rows


def load_problems(path):
    rows = read_jsonl(path, "problems")
    problems = {}
    for k, r in rows:
        for field in REQUIRED_FIELDS:
            if not isinstance(r.get(field), str):
                fail(f"problems line {k} lacks string field {field}")
        if r["task_id"] in problems:
            fail(f"problems line {k} repeats task_id {r['task_id']}")
        problems[r["task_id"]] = r
    if len(problems) != EXPECTED_PROBLEMS:
        fail(f"problems file holds {len(problems)} problems, "
             f"expected {EXPECTED_PROBLEMS}")
    return problems


def load_responses(path, problems):
    rows = read_jsonl(path, "response")
    if not rows:
        fail("response file is empty")
    seen = set()
    for k, r in rows:
        tid = r.get("task_id")
        if not isinstance(tid, str):
            fail(f"response line {k} lacks string field task_id")
        if not isinstance(r.get("response"), str):
            fail(f"response line {k} lacks string field response")
        if tid not in problems:
            fail(f"response line {k} has unknown task_id {tid!r}")
        if tid in seen:
            fail(f"response line {k} repeats task_id {tid}")
        seen.add(tid)
    return [r for _, r in rows]


def main(path, label, problems_path):
    problems = load_problems(problems_path)
    rows = load_responses(path, problems)
    n = len(problems)
    missing = n - len(rows)
    passed = 0
    fails = []
    kinds = {}
    chars = 0
    for r in rows:
        p = problems[r["task_id"]]
        ok, kind = passes(p, r["response"])
        passed += ok
        chars += len(r["response"])
        if not ok:
            fails.append((r["task_id"], kind))
            kinds[kind.split(":")[0]] = kinds.get(kind.split(":")[0], 0) + 1
    print(f"== {label}: pass@1 {passed}/{n} = {passed / n:.1%} "
          f"(avg chars {chars // len(rows)})")
    print(f"   missing responses (counted as failures): {missing}")
    print(f"   fail kinds: {kinds}")
    print(f"   failed ids: {[t for t, _ in fails][:20]}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("usage: he_grade.py RESPONSES LABEL PROBLEMS")
    main(sys.argv[1], sys.argv[2], sys.argv[3])
