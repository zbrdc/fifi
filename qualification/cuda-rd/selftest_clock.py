#!/usr/bin/env python3
"""CPU self-test of clockcheck.py: the clock void conditions, the warm-up conditions and the first-step flag.
Run: python3 -I selftest_clock.py. One line per check; exit 1 on any unexpected result."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a5_eval as ev  # noqa: E402
import clockcheck as ck  # noqa: E402
import fixtures_files as fx  # noqa: E402
import rawcheck as rw  # noqa: E402

results = []


def check(name, cond):
    results.append(cond)
    print(("ok   " if cond else "FAIL ") + name)


def raises(exc, fn, *a):
    try:
        fn(*a)
    except exc:
        return True
    return False


def main():
    ref = ev.Reference()
    docs = fx.make_all(ref)
    d0 = docs["C1"]; r0 = d0["prompts"][0]["runs"][0]; t0, t1, pid = r0["timed_start_unix_ns"], r0["timed_end_unix_ns"], d0["identities"]["pid"]
    def judge(**kw):
        return ck.judge_window(fx.make_watch(d0, **kw), t0, t1, pid)
    check("clock: 1800 MHz busy all through the timed region -> not voided", not judge()["voided"])
    j = judge()
    check(f"clock: minimum {j['min_mhz']:.0f} and busy median {j['median_busy_mhz']:.0f} are reported", j["min_mhz"] == 1800 and j["median_busy_mhz"] == 1800)
    check("clock: exactly 1395 MHz is not below the floor", not judge(floor_dip=1395)["voided"])
    check("clock VOID: ANY sample inside the timed region at 1394 MHz", judge(floor_dip=1394)["voided"])
    check("clock VOID: a single idle-level sample (210 MHz) inside the timed region", judge(floor_dip=210)["voided"])
    check("clock VOID: busy median 1500 MHz (all samples above the floor)", judge(busy_mhz=1500)["voided"])
    check("clock: busy median exactly 1600 MHz is accepted", not judge(busy_mhz=1600)["voided"])
    check("clock VOID: busy median 1599 MHz", judge(busy_mhz=1599)["voided"])
    j = judge(busy_util=20)
    check("clock: no busy sample -> the busy-median rule is NOT EVALUATED, not a void (lane M Q12)", not j["voided"] and j["median_rule"].startswith("not evaluated") and j["n_busy"] == 0)
    def busy_run(n_busy, mhz=1500, extra=()):
        mid = (t0 + t1) // 2
        s = [{"t": (mid + k * 100_000_000) / 1e9, "clocks.sm": str(mhz), "utilization.gpu": "98", "apps": [pid]} for k in range(n_busy)]
        s += [{"t": (mid - 200_000_000 - k * 100_000_000) / 1e9, "clocks.sm": "1800", "utilization.gpu": "10", "apps": [pid]} for k in range(3)]
        return ev_judge(s + list(extra))
    ev_judge = lambda s: ck.judge_window(s, t0, t1, pid)  # noqa: E731
    j9, j10 = busy_run(9), busy_run(10)
    check("clock: 9 busy samples at 1500 MHz -> median rule not evaluated, NOT voided", not j9["voided"] and j9["n_busy"] == 9 and "not evaluated" in j9["median_rule"])
    check("clock: 10 busy samples at 1500 MHz -> median rule evaluated and VOIDED", j10["voided"] and j10["n_busy"] == 10 and j10["median_rule"] == "evaluated")
    check("clock: 10 busy samples at 1600 MHz -> evaluated, accepted", not busy_run(10, 1600)["voided"])
    check("clock: the 1395 floor still applies with only 9 busy samples (one idle-level sample voids)", busy_run(9, 1700, [{"t": (t0 + 50_000_000) / 1e9, "clocks.sm": "300", "utilization.gpu": "5", "apps": [pid]}])["voided"])
    check("clock: total and busy sample counts are reported", j10["n"] == 13 and j10["n_busy"] == 10)
    check("clock VOID: another compute process appeared", judge(foreign_pid=777)["voided"])
    partial = fx.make_watch(d0)
    partial[len(partial) // 2].pop("apps")     # a sample inside the timed region
    check("clock: a card sample without the apps list is a typed STOP, never a silent skip of the foreign-process rule",
          raises(ev.Stop, ck.judge_window, partial, t0, t1, pid))
    check("clock VOID: fewer than 3 samples inside the timed region", ck.judge_window(fx.make_watch(d0)[:2], t0, t1, pid)["voided"])
    outside = [s for s in fx.make_watch(d0) if not (t0 <= s["t"] * 1e9 <= t1)]
    check("clock: idle samples OUTSIDE the timed region (pre-check, post-run) never void or count", ck.judge_window(outside, t0, t1, pid)["n"] == 0)
    check("clock: samples before and after the region at 210 MHz do not void a good run", not judge()["voided"])
    check("clock: a missing watch file is a typed STOP", raises(ev.Stop, ck.read_watch, "/nonexistent/x.jsonl"))
    check("warm-up: a good set of files has no problems", ck.warmup_problems(docs) == [])
    d2 = fx.clone(docs); d2["C2"]["prompts"][0]["runs"][0]["graph_ready_before_timed"] = False
    check("warm-up FAIL: graph not instantiated before the timed region (graph mode)", len(ck.warmup_problems(d2)) == 1)
    d2 = fx.clone(docs); d2["B1"]["prompts"][1]["runs"][0]["prompt_synced_unix_ns"] += 5_000_000_000
    check("warm-up FAIL: timed region started before the prompt pass synchronized", len(ck.warmup_problems(d2)) == 1)
    ed = fx.make_all(ref, capture="eager")
    check("warm-up: eager data needs no graph (graph_ready false is fine)", ck.warmup_problems(ed) == [])
    for label, mut in (("timed_start_unix_ns missing", lambda r: r.pop("timed_start_unix_ns")), ("window shorter than decode_ns", lambda r: r.__setitem__("timed_end_unix_ns", r["timed_start_unix_ns"] + 1)),
                       ("step_ns missing", lambda r: r.pop("step_ns")), ("step_ns of 100 entries", lambda r: r.__setitem__("step_ns", r["step_ns"][:100])),
                       ("step_ns adding up to twice decode_ns", lambda r: r.__setitem__("step_ns", [2 * x for x in r["step_ns"]])), ("graph_ready_before_timed not a bool", lambda r: r.__setitem__("graph_ready_before_timed", 1))):
        d2 = fx.clone(docs); mut(d2["C1"]["prompts"][0]["runs"][0])
        check(f"validate refuses: {label}", raises(ev.Stop, rw.validate_raw, "C1", d2["C1"], ref))
    runs = [fx.make_run(ref, "cuda/free", "P1", "free", 50, k) for k in range(5)]
    sp = ev.eval_speed(runs)
    check(f"first step: {sp['first_step_ms']:.2f} ms beside median step {sp['median_step_ms']:.2f} ms, not flagged (1.2x)", sp["first_step_flagged"] == [])
    runs[3]["step_ns"][0] = runs[3]["step_ns"][1] * 3
    sp = ev.eval_speed(runs)
    check("first step: 3x the median step is flagged, and still counted in decode_ns (never dropped)", sp["first_step_flagged"] == [3] and runs[3]["decode_ns"] > 0)
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
