#!/usr/bin/env python3
"""End-to-end CPU self-test of assemble_report.py with fabricated raw files (llama.cpp data in the 'fifi' slot): gate, clock, warm-up and
capture cases plus --check-raw. Usage: python3 -I selftest_report.py SCRATCH_DIR   (writes under SCRATCH_DIR only)
Each case runs the real command line and checks the exit code: 0 PASS, 1 FAIL, 2 STOP, 3 REFUSED, 4 PRE-RUN complete, 5 VOIDED, 7 BLOCKED."""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_case import *  # noqa: F401,F403,E402
from report_case import HERE, LANE_C, case, fx, ref, raw, release, results, scratch, wrapper_log  # noqa: E402,F401


def main():
    scratch.mkdir(parents=True, exist_ok=True)
    case("pass-stand-in", 0)
    case("fail-wrapper-line-missing", 1, wlog=wrapper_log(lines=LANE_C[:3] + LANE_C[4:]))
    case("fail-wrapper-equal-replaced-by-differ", 1, wlog=wrapper_log(lines=[x.replace("equal", "differ") if "lm_head" in x else x for x in LANE_C]))
    docs = fx.make_all(ref)
    w = {"C2": fx.make_watch(docs["C2"], floor_dip=1394)}
    case("VOIDED-any-sample-below-1395", 5, docs, watch=w)
    w = {"C2": fx.make_watch(docs["C2"], busy_mhz=1500)}
    case("VOIDED-busy-median-1500", 5, docs, watch=w)
    w = {"B1": fx.make_watch(docs["B1"], busy_util=10)}
    case("pass-no-busy-sample-median-not-evaluated", 0, docs, watch=w)
    w = {"A1": fx.make_watch(docs["A1"], foreign_pid=4242)}
    case("VOIDED-foreign-compute-process", 5, docs, watch=w)
    w = {"C3": fx.make_watch(docs["C3"])[:2]}
    case("VOIDED-too-few-samples", 5, docs, watch=w)
    case("pass-idle-clocks-outside-the-window-do-not-void", 0, docs)
    case("stop-missing-watch-file", 2, docs, drop_watch=("C1",))
    case("BLOCKED-three-consecutive-voided-runs", 7, docs, blocked=("C1",))
    docs = fx.make_all(ref)
    docs["C2"]["prompts"][0]["runs"][0]["graph_ready_before_timed"] = False
    case("fail-graph-not-instantiated-before-timed-region", 1, docs)
    docs = fx.make_all(ref)
    docs["B2"]["prompts"][0]["runs"][0]["prompt_synced_unix_ns"] += 9_000_000_000
    case("fail-timed-region-started-before-prompt-sync", 1, docs)
    docs = fx.make_all(ref)
    r = docs["C3"]["prompts"][0]["runs"][0]
    e = r["step_ns"][1]; extra = 3 * e - r["step_ns"][0]; r["step_ns"][0] = 3 * e
    for i in range(1, 128):
        r["step_ns"][i] -= extra // 127
    case("pass-first-step-3x-flagged-never-dropped", 0, docs)
    flagged = "runs whose first step is over 2x the median: ['C3']" in (scratch / "pass-first-step-3x-flagged-never-dropped.md").read_text()
    results.append(flagged); print(f"{'ok  ' if flagged else 'FAIL'} the report flags the slow first step")
    case("fail-slow-60ms", 1, fx.make_all(ref, ms=60.0))
    case("fail-no-wrapper-log", 1, wlog="")
    case("fail-wrapper-running-0-tests", 1, wlog="engine_commit c0ffee0\nprofile release\nrunning 0 tests\ntest result: ok. 0 passed; 0 failed; 0 ignored\n")
    case("fail-wrapper-wrong-commit", 1, wlog=wrapper_log("deadbee"))
    case("fail-release-json-empty-object", 1, rel={})
    case("fail-release-json-debug-build-info", 1, rel=release(build_info={"profile": "debug", "opt_level": "0", "debug": True, "debug_assertions": True}))
    case("fail-release-device-debug-miss", 1, rel=release(misses=[".nv_debug_info_reg_"]))
    case("fail-release-json-other-binary", 1, rel=release(exe="a" * 64))
    case("fail-selftest-rc-1", 1, selftest={"rc": 1, "scripts": __import__("assemble_report").script_hashes()})
    case("fail-selftest-result-for-other-scripts", 1, selftest={"rc": 0, "scripts": {"a5_eval.py": "0" * 64}})
    docs = fx.make_all(ref)
    docs["B2"]["prompts"][0]["runs"][0]["steps"][9]["hash"] = "0" * 16
    case("fail-nondeterministic", 1, docs)
    docs = fx.make_all(ref)
    docs["E2"]["prompts"][1]["runs"][0]["steps"][40]["hash"] = "1" * 16
    case("fail-capture-mismatch", 1, docs)
    docs = fx.make_all(ref)
    docs["C3"]["prompts"][0]["runs"][0]["captured"] = False
    case("fail-silent-capture-fallback", 1, docs)
    case("fail-killed-call", 1, killed=("E1",))
    case("fail-stopped-early-skipped-calls", 1, killed=("C1",), skipped=("C2", "C3", "C4", "C5", "B1"))
    docs = fx.make_all(ref)
    docs["C4"]["incomplete"] = True; docs["C4"]["elapsed_s"] = 270; docs["C4"]["partial"] = {"steps_done": 900, "elapsed_ns": 270_000_000_000}
    case("fail-too-slow-incomplete", 1, docs)
    docs = fx.make_all(ref)
    step = ref.block("P1")["gated"][5]["step"]
    docs["A1"]["prompts"][0]["runs"][0]["steps"][step]["argmax"] += 1
    docs["A1"]["prompts"][0]["runs"][0]["steps"][step]["top32"][0][0] += 1
    case("fail-a5-top1-differs-at-gated-step", 1, docs)
    docs = fx.make_all(ref)
    docs["A1"]["prompts"][0]["runs"][0]["steps"][ref.block("P1")["excluded"][0]["step"]]["lse"] = None
    case("stop-null-lse-at-excluded-step", 2, docs)
    case("pass-prerun-eager", 4, fx.make_all(ref, capture="eager"), prerun=True)
    ptext = (scratch / "pass-prerun-eager.md").read_text()
    t3 = ptext.split("## Table 3")[1].split("## Hold records")[0]
    ok = "PASS" not in t3 and "tables pass (information only" in t3 and "Eager" in ptext
    results.append(ok); print(f"{'ok  ' if ok else 'FAIL'} the eager pre-run report never prints PASS in Table 3 and states the capture outcome Eager")
    # --check-raw
    fx.write_all(fx.make_all(ref), raw)
    for tag, f, want in (("check-raw A1 valid", raw / "A1.json", 0), ("check-raw missing file", raw / "ZZ.json", 2)):
        p = subprocess.run([sys.executable, "-I", str(HERE / "assemble_report.py"), "--check-raw", str(f)], capture_output=True, text=True)
        ok = p.returncode == want and "Traceback" not in p.stderr
        results.append(ok); print(f"{'ok  ' if ok else 'FAIL'} {tag}: rc={p.returncode} (want {want})")
    inc = fx.make_all(ref)
    inc["C1"]["incomplete"] = True; inc["C1"]["elapsed_s"] = 270; inc["C1"]["partial"] = {"steps_done": 800}
    fx.write_all(inc, raw)
    p = subprocess.run([sys.executable, "-I", str(HERE / "assemble_report.py"), "--check-raw", str(raw / "C1.json")], capture_output=True, text=True)
    ok = p.returncode == 1 and "INCOMPLETE" in p.stdout
    results.append(ok); print(f"{'ok  ' if ok else 'FAIL'} check-raw incomplete file: rc={p.returncode} (want 1)")
    slow = fx.make_all(ref, ms=300.0)
    fx.write_all(slow, raw)
    p = subprocess.run([sys.executable, "-I", str(HERE / "assemble_report.py"), "--check-raw", str(raw / "A1.json")], capture_output=True, text=True)
    ok = p.returncode == 2 and "3.6" in p.stdout
    results.append(ok); print(f"{'ok  ' if ok else 'FAIL'} check-raw below 3.6 tok/s: rc={p.returncode} (want 2)")
    print("SELFTEST", "OK" if all(results) else "FAILED", f"({len(results)} cases)")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
