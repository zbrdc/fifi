#!/usr/bin/env python3
"""Shared helpers of the end-to-end report self-tests: fabricated raw files, release JSON, wrapper logs, and `case`, which runs the real
assemble_report.py command line and checks its exit code. The scratch directory is argv[1]. Design guidance only, no code copied."""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a5_eval as ev  # noqa: E402
import assemble_report as ar  # noqa: E402
import fixtures_files as fx  # noqa: E402

scratch = Path(sys.argv[1]) / "reportcases"
raw = scratch / "raw"
ref = ev.Reference()
shapes = ev.read_json(HERE / "rd2e_shapes.json")
results = []


LANE_C = ["q_qkv Q80 8192x2048 y_bytes=32768 second_stream=equal", "kv_shared_gate_up Q80 512x2048 y_bytes=2048 equal",
          "z_gate Q80 4096x2048 y_bytes=16384 equal", "o_ssm_out Q80 2048x4096 y_bytes=8192 equal",
          "alpha_beta Q80 32x2048 y_bytes=128 equal", "shared_down Q80 2048x512 y_bytes=8192 equal",
          "lm_head Q6K 248320x2048 y_bytes=993280 equal", "expert_gate_up Q4K 512x2048 repeats=1000 y_bytes=16384 slots=8",
          "expert_down Q4K 2048x512 repeats=1000 y_bytes=65536 slots=8"]


def wrapper_log(commit=fx.COMMIT, tests=5, lines=None):
    out = [f"engine_commit {commit}", "profile release", f"running {tests} tests"]
    out += [f"test projection_repeat_{n} ... ok" for n in ("dense", "expert", "perturbed", "comparator", "splitk_guard")[:tests]]
    out += (LANE_C if lines is None else lines) + [f"test result: ok. {tests} passed; 0 failed; 0 ignored"]
    return "\n".join(out) + "\n"


def release(exe=fx.EXE, **over):
    r = {"build_info": {"profile": "release", "opt_level": "3", "debug": False, "debug_assertions": False},
         "cuobjdump_version": "Cuda compilation tools, release 12.2, V12.2.140", "misses": [], "exe_sha256": exe}
    r.update(over)
    return r


def case(tag, want, docs=None, rel=None, wlog=None, selftest=None, prerun=False, drop=(), killed=(), junk=None, skipped=(), watch=None, blocked=(), drop_watch=(), head=fx.COMMIT):
    docs = docs or fx.make_all(ref)
    fx.write_all(docs, raw, watch)
    for n in drop_watch:
        (raw / f"{n}.watch.jsonl").unlink()
    for n in blocked:
        (raw / f"{n}.blocked").write_text("blocked\n")
    for n in drop:
        (raw / f"{n}.json").unlink()
    for n in killed:
        (raw / f"{n}.json").unlink()
        (raw / f"{n}.killed").write_text("killed at 300 s\n")
    for n in skipped:
        (raw / f"{n}.json").unlink()
        (raw / f"{n}.skipped").write_text("stopped early\n")
    if junk:
        (raw / "C1.json").write_text(junk)
    (scratch / "release.json").write_text(json.dumps(rel if rel is not None else release()))
    (scratch / "wrapper.log").write_text(wlog if wlog is not None else wrapper_log())
    (scratch / "selftest.json").write_text(json.dumps(selftest if selftest is not None else {"rc": 0, "scripts": __import__("assemble_report").script_hashes()}))
    cmd = [sys.executable, "-I", str(HERE / "assemble_report.py"), "--raw", str(raw), "--release-check", str(scratch / "release.json"),
           "--wrapper-log", str(scratch / "wrapper.log"), "--selftest-result", str(scratch / "selftest.json"), "--head", head, "--date", "selftest",
           "--out", str(scratch / f"{tag}.md")] + (["--prerun"] if prerun else [])
    p = subprocess.run(cmd, capture_output=True, text=True)
    ok = p.returncode == want and "Traceback" not in p.stderr + p.stdout
    results.append(ok)
    print(f"{'ok  ' if ok else 'FAIL'} {tag}: rc={p.returncode} (want {want}) {p.stdout.strip()[:150]}")
