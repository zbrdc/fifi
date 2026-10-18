#!/usr/bin/env python3
"""CPU self-test of wrapperlog.py on lane C's RD-2e log format.
Run: python3 -I selftest_wrapper.py. One line per check; exit 1 on any unexpected result."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a5_eval as ev  # noqa: E402
import wrapperlog as wl  # noqa: E402

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
    shapes = ev.read_json(ev.HERE / "rd2e_shapes.json")
    lane_c = ["q_qkv Q80 8192x2048 y_bytes=32768 second_stream=equal", "kv_shared_gate_up Q80 512x2048 y_bytes=2048 equal",
              "z_gate Q80 4096x2048 y_bytes=16384 equal", "o_ssm_out Q80 2048x4096 y_bytes=8192 equal",
              "alpha_beta Q80 32x2048 y_bytes=128 equal", "shared_down Q80 2048x512 y_bytes=8192 equal",
              "lm_head Q6K 248320x2048 y_bytes=993280 equal", "expert_gate_up Q4K 512x2048 repeats=1000 y_bytes=16384 slots=8",
              "expert_down Q4K 2048x512 repeats=1000 y_bytes=65536 slots=8"]
    head = ["engine_commit abc1234", "profile release", "running 5 tests"]
    tests = [f"test family::x::projection_repeat_{n} ... ok" for n in ("dense", "expert", "perturbed", "comparator", "splitk_guard")]
    tail = ["test result: ok. 5 passed; 0 failed; 0 ignored"]
    def wlog(lines, tests=tests):
        return "\n".join(head + tests + lines + tail)
    check("wrapper log: all 9 lane C lines and 5 passed -> accepted", wl.parse_wrapper_log(wlog(lane_c), shapes, "abc1234") == [])
    nocap = [f"test family::x::projection_repeat_{n} ... {line}" if k == 0 else f"test family::x::projection_repeat_{n} ... ok" for k, (n, line) in enumerate(zip(("dense", "expert", "perturbed", "comparator", "splitk_guard"), lane_c))]
    check("wrapper log: --nocapture output on the `test ... ` line itself is still matched", wl.parse_wrapper_log("\n".join(head + nocap + lane_c[1:] + tail), shapes, "abc1234") == [])
    check("wrapper log: dense lines need no repeats=1000", all("repeats" not in x for x in lane_c[:7]) and wl.parse_wrapper_log(wlog(lane_c), shapes, "abc1234") == [])
    for i, name in enumerate(s["name"] for s in shapes):
        missing = [x for k, x in enumerate(lane_c) if k != i]
        check(f"wrapper log: line for {name} missing -> problem", any(name in b for b in wl.parse_wrapper_log(wlog(missing), shapes, "abc1234")))
    check("wrapper log: `equal` replaced by `differ` on a dense line -> problem", bool(wl.parse_wrapper_log(wlog([x.replace("equal", "differ") if "q_qkv" in x else x for x in lane_c]), shapes, "abc1234")))
    check("wrapper log: expert line without repeats=1000 -> problem", bool(wl.parse_wrapper_log(wlog([x.replace("repeats=1000", "") if "expert_down" in x else x for x in lane_c]), shapes, "abc1234")))
    check("wrapper log: running 0 tests -> problems", bool(wl.parse_wrapper_log("running 0 tests\ntest result: ok. 0 passed; 0 failed; 0 ignored", shapes, "abc1234")))
    ign = wlog(lane_c).replace("test family::x::projection_repeat_dense ... ok", "test family::x::projection_repeat_dense ... ignored")
    check("wrapper log: an #[ignore]d projection_repeat_ test (it would not run without --ignored) -> problem", any("ignore" in b for b in wl.parse_wrapper_log(ign, shapes, "abc1234")))
    check("wrapper log: 4 passed -> problem", bool(wl.parse_wrapper_log(wlog(lane_c).replace("5 passed", "4 passed"), shapes, "abc1234")))
    check("wrapper log: a failing test -> problem", bool(wl.parse_wrapper_log(wlog(lane_c).replace("0 failed", "1 failed"), shapes, "abc1234")))
    check("wrapper log: wrong commit -> problem", bool(wl.parse_wrapper_log(wlog(lane_c), shapes, "other")))
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
