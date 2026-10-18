#!/usr/bin/env python3
"""CPU self-test of report_text.py: the rendered record contains what the gate needs to show, built from evaluated results of the
fabricated stand-in files (no assembler involved). Usage: python3 -I selftest_render.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a5_eval as ev  # noqa: E402
import clockcheck as ck  # noqa: E402
import fixtures_files as fx  # noqa: E402
from report_text import LABEL, hold_records, render  # noqa: E402

results = []


def check(name, cond):
    results.append(cond)
    print(("ok   " if cond else "FAIL ") + name)


def evaluated(ref, docs):
    free = [fx.make_run(ref, "cuda/free", "P1", "free", 50, k) for k in range(5)]
    res = {"a5": {}, "free": {}, "det": {}, "tf_free": {}, "cap_eq": {}, "speed": ev.eval_speed(free), "speed_others": {p: 20.0 for p in ev.PROMPTS[1:]}}
    for pid in ev.PROMPTS:
        run = fx.make_run(ref, "cuda/follow", pid, "forced")
        res["a5"][pid] = ev.eval_a5(run, ref.block(pid)["prompt_ids_sha256"], ref, pid)
        res["free"][pid] = ev.eval_free([fx.make_run(ref, "cuda/free", pid, "free")], ref, pid)
        res["det"][pid] = {"runs": 3, "ok": True}
        res["tf_free"][pid] = {"ok": True}
        res["cap_eq"][pid] = True
    clocks = {}
    for name in ("C1", "B1"):
        d = docs[name]
        watch = fx.make_watch(d, busy_util=10 if name == "B1" else 98)
        for p in d["prompts"]:
            r = p["runs"][0]
            clocks[(name, p["id"])] = ck.judge_window(watch, r["timed_start_unix_ns"], r["timed_end_unix_ns"], d["identities"]["pid"])
    return res, clocks


def main():
    ref = ev.Reference()
    docs = fx.make_all(ref)
    res, clocks = evaluated(ref, docs)
    ident, hashes = docs["A1"]["identities"], {"data A1.json": "ab" * 32}
    text = render(res, ident, hashes, [], "2026-10-10", False, ref, "", ["call-C1.log: HOLD post {}"], clocks)
    check("normal record: title, three tables, hold records appended by script", "# CUDA-RD stage 1 qualification run 2026-10-10" in text and "## Table 1" in text and "## Table 2" in text and "## Table 3" in text and "call-C1.log: HOLD post" in text)
    check("the excluded share is printed for information (P3 12/128, 9.38 percent), no share rule", "12/128 (9.38 percent)" in text and "information only" in text)
    check("the clock table shows total and busy sample counts, minimum, median and the median rule", "| C1 | P1 | 13 | 13 | 1800 | 1800 | evaluated | no |" in text)
    check("a run with no busy sample shows 'not evaluated' and a dash, not a void", "not evaluated (0 busy samples, fewer than 10)" in text and "| - |" in text)
    check("the first timed step is printed beside the median step", "First timed step (P1 median run)" in text)
    check("the record reports BOTH syncs per token and host writes per token", "Host-blocking syncs per token: 1 and host writes per token: 1" in text)
    check("verdict PASS with no misses", "| 1 | 18 |" in text and "PASS |" in text and "Misses:" not in text)
    bad = render(res, ident, hashes, ["speed 16.00 tok/s < 18"], "2026-10-10", False, ref, "", [], clocks)
    check("a miss is listed and the verdict is FAIL", "Misses: speed 16.00 tok/s < 18" in bad and "FAIL |" in bad)
    pre = render(res, ident, hashes, [], "2026-10-10", True, ref, "", [], clocks)
    check("pre-run: the label is in the title, all three table headings and the verdict", pre.count(LABEL) >= 5 and f"# CUDA-RD stage 1 {LABEL}" in pre)
    slow = dict(res["speed"]); slow["first_step_flagged"] = [2]; slow["first_step_ms"] = slow["median_step_ms"] * 3
    flagged = render({**res, "speed": slow}, ident, hashes, [], "2026-10-10", False, ref, "", [], clocks)
    check("pre-run: Table 3 never says PASS for stage 1 (the cell reads `tables pass (information only, never a stage 1 result)`), and the capture outcome Eager is stated", "| PASS |" not in pre and "tables pass (information only, never a stage 1 result)" in pre and "capture outcome of every run (RD-9d `CaptureOutcome`): Eager" in pre)
    pre_bad = render(res, ident, hashes, ["speed 16.00 tok/s < 18"], "2026-10-10", True, ref, "", [], clocks)
    check("pre-run with a miss: `tables FAIL (information only...)`, still labelled", "tables FAIL (information only, never a stage 1 result)" in pre_bad and pre_bad.count(LABEL) >= 5)
    check("a first step over 2x the median is flagged and the run is named", "FLAG: over 2x" in flagged and "['C3']" in flagged)
    check("hold_records keeps only the HOLD lines of the call logs", hold_records("/nonexistent") == [])
    print("RENDER SELFTEST", "OK" if all(results) else "FAILED", f"({len(results)} checks)")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
