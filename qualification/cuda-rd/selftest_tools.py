#!/usr/bin/env python3
"""CPU self-test of the run-path tools that execute inside a GPU hold: find_test_exes.py, check_gguf.py, idle_check.py and hold.sh.
A fake `nvidia-smi` on PATH and a dummy binary stand in for the card and for cuda-rd-run. Usage: python3 -I selftest_tools.py SCRATCH_DIR"""
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
S = Path(sys.argv[1]) / "tools"
S.mkdir(parents=True, exist_ok=True)
results = []


def check(name, cond):
    results.append(cond)
    print(("ok   " if cond else "FAIL ") + name)


def run(cmd, env=None, **kw):
    e = {**os.environ, **(env or {})}
    return subprocess.run(cmd, capture_output=True, text=True, env=e, **kw)


def script(path, text):
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def runner():
    import run_selftests as rs
    rs.ar.script_hashes = lambda: {}          # manifest gate = finding 1, not this case's subject
    rs.TESTS = [["selftest_tokens.py"]]       # one fast, dependency-free test
    d = Path(sys.argv[1]) / "runner"
    shutil.rmtree(d, ignore_errors=True)
    rc = rs.main(str(d / "missing" / "scratch"), str(d / "out.json"))
    check("run_selftests: a scratch dir that does not exist is created; log and result file are written",
          rc == 0 and (d / "missing/scratch/selftest_tokens.py.log").is_file() and (d / "out.json").is_file())
    # 16c F10: a self-test that hangs or cannot run is recorded as a typed failure, never an uncaught exception.
    # The patched run executes a real 30 s sleep under the shrunk timeout, so TimeoutExpired raises at the same site.
    real_run = rs.subprocess.run
    rs.subprocess.run = lambda cmd, **k: real_run(["sleep", "30"], **{**k, "timeout": 0.2})
    try:
        rc = rs.main(str(d / "missing2" / "scratch"), str(d / "out2.json"))
        logged = (d / "missing2/scratch/selftest_tokens.py.log").read_text()
        result = json.loads((d / "out2.json").read_text())
    finally:
        rs.subprocess.run = real_run
    check("run_selftests: a hung self-test is a typed failure (rc 1, a STOP line in its log, no traceback)",
          rc == 1 and "STOP" in logged and "Traceback" not in logged and result["results"]["selftest_tokens.py"] == 2)


def finder():
    lib = {"reason": "compiler-artifact", "executable": "/t/deps/inference_engine-abc", "target": {"name": "inference_engine", "kind": ["lib"]}, "profile": {"test": True}}
    other = {"reason": "compiler-artifact", "executable": "/t/deps/other-1", "target": {"name": "other", "kind": ["test"]}, "profile": {"test": True}}
    nontest = {"reason": "compiler-artifact", "executable": None, "target": {"name": "inference_engine", "kind": ["lib"]}, "profile": {"test": False}}
    for label, items, rc in (("one lib test exe", [lib, other, nontest], 0), ("no lib test exe", [other, nontest], 2), ("two lib test exes", [lib, {**lib, "executable": "/t/deps/inference_engine-def"}], 2)):
        log = S / "cargo.json"
        log.write_text("\n".join(json.dumps(x) for x in items) + "\nnot json\n")
        r = run([sys.executable, "-I", str(HERE / "find_test_exes.py"), str(log)])
        check(f"find_test_exes: {label} -> rc {rc}", r.returncode == rc and (rc != 0 or r.stdout.split()[0] == "/t/deps/inference_engine-abc"))
    r = run([sys.executable, "-I", str(HERE / "find_test_exes.py"), str(S / "missing")])
    check("find_test_exes: missing log -> typed STOP", r.returncode == 2 and "Traceback" not in r.stderr)


def gguf():
    f = S / "small.gguf"
    f.write_bytes(b"x" * 100)
    r = run([sys.executable, "-I", str(HERE / "check_gguf.py"), str(f)])
    check("check_gguf: a file that is not the frozen GGUF -> STOP (size and sha256 differ)", r.returncode == 2 and "frozen identity" in r.stdout)
    r = run([sys.executable, "-I", str(HERE / "check_gguf.py"), str(S / "missing.gguf")])
    check("check_gguf: missing file -> typed STOP", r.returncode == 2 and "Traceback" not in r.stderr)


def card():
    fake = script(S / "bin" / "nvidia-smi", '''#!/usr/bin/env bash
[ -n "${FAKE_SMI_DEAD:-}" ] && exit 3
case "$*" in
  *query-compute-apps*) printf '%s' "${FAKE_APPS:-}" ;;
  *query-gpu=name*) [ -n "${FAKE_NAME_EMPTY:-}" ] || echo "NVIDIA GeForce RTX 3090" ;;
  *query-gpu=driver_version*) echo "580.159.03" ;;
  *query-gpu=utilization*)
    if [ -n "${FAKE_GPU_ONCE:-}" ] && [ ! -e "$FAKE_GPU_ONCE" ]; then touch "$FAKE_GPU_ONCE"; echo "98, 414, 1800, 405, 37, 24, 580.159.03, P0"
    else echo "${FAKE_GPU:-0, 414, 210, 405, 37, 24.26, 580.159.03, P8}"; fi ;;
  *) exit 3 ;;
esac
''') if (S / "bin").mkdir(exist_ok=True) is None else None
    env = {"PATH": f"{S / 'bin'}:{os.environ['PATH']}", "RD10_IDLE_SETTLE_SECS": "0.1"}
    idle = lambda extra: run([sys.executable, "-I", str(HERE / "idle_check.py"), "pre"], {**env, **extra})  # noqa: E731
    check("idle_check pre: an idle card -> exit 0 and a HOLD pre line", idle({}).returncode == 0 and "HOLD pre" in idle({}).stdout)
    check("idle_check pre: a compute process present -> exit 1", idle({"FAKE_APPS": "4242, 900 MiB\n"}).returncode == 1)
    check("idle_check pre: utilization 50 -> exit 1", idle({"FAKE_GPU": "50, 414, 1800, 405, 37, 24, 580.159.03, P0"}).returncode == 1)
    once = S / "gpu-once"; once.unlink(missing_ok=True)
    settled = idle({"FAKE_GPU_ONCE": str(once)})
    check("idle_check pre: busy on the first sample only (the previous call just ended) -> settles and exits 0", settled.returncode == 0 and "HOLD settle 1" in settled.stdout)
    check("idle_check pre: a compute process present is not retried (no settle line)", "HOLD settle" not in idle({"FAKE_APPS": "4242, 900 MiB\n"}).stdout)
    check("idle_check pre: 80 C -> exit 1", idle({"FAKE_GPU": "0, 414, 210, 405, 80, 24, 580.159.03, P8"}).returncode == 1)
    hang = S / "bin-hang"; hang.mkdir(exist_ok=True)
    script(hang / "nvidia-smi", "#!/usr/bin/env bash\nsleep 30\n")
    import idle_check as ic
    real_run = ic.subprocess.run
    ic.subprocess.run = lambda cmd, **k: real_run(cmd, **{**k, "timeout": 0.2})
    saved = os.environ["PATH"]; os.environ["PATH"] = f"{hang}:{saved}"
    code = None
    try:
        ic.smi(["--query-gpu=utilization.gpu"])
    except SystemExit as e:
        code = e.code
    finally:
        ic.subprocess.run = real_run; os.environ["PATH"] = saved
    check("idle_check: a hung nvidia-smi is a typed STOP (exit 2), never a traceback", code == 2)
    return fake, env


def hold(env):
    dummy = script(S / "dummy-bin", f'#!/usr/bin/env bash\necho "$@" > {S}/args.txt\nsleep 1.3\nexit ${{DUMMY_RC:-0}}\n')
    sha = hashlib.sha256(dummy.read_bytes()).hexdigest()
    watch = S / "watch.jsonl"
    base = ["bash", str(HERE / "hold.sh")]
    e = {**env, "TMPDIR": str(S)}
    r = run([*base, sha, str(dummy), str(watch), "run", "--out", "x.json"], e)
    args = (S / "args.txt").read_text()
    samples = [json.loads(ln) for ln in watch.read_text().splitlines()]
    check("hold.sh: a normal call exits 0 and the binary got the card identity and the inner deadline",
          r.returncode == 0 and "--gpu-name NVIDIA GeForce RTX 3090" in args and "--driver 580.159.03" in args and "--deadline-secs 270" in args)
    check("hold.sh: the card was sampled while the binary ran (field t, clocks.sm, utilization.gpu, apps)",
          len(samples) >= 2 and all({"t", "clocks.sm", "utilization.gpu", "apps"} <= set(s) for s in samples))
    check("hold.sh: prints the executable sha and the hold seconds", f"HOLD exe_sha256 {sha}" in r.stdout and "HOLD seconds" in r.stdout and "HOLD post" in r.stdout)
    r = run([*base, sha, str(dummy), str(watch), "run"], {**e, "DUMMY_RC": "3"})
    check("hold.sh: the binary's exit 3 (REFUSED) is passed through", r.returncode == 3)
    r = run([*base, sha, str(dummy), str(watch), "run"], {**e, "DUMMY_RC": "2"})
    check("hold.sh: the binary's exit 2 (STOP) is passed through", r.returncode == 2)
    r = run([*base, "0" * 64, str(dummy), str(watch), "run"], e)
    check("hold.sh: a different executable -> exit 6 and the binary never ran", r.returncode == 6 and "not the one the release check examined" in r.stdout)
    (S / "args.txt").unlink()
    r = run([*base, sha, str(dummy), str(watch), "run"], {**e, "FAKE_APPS": "4242, 900 MiB\n"})
    check("hold.sh: a card that is not idle -> exit 4 and the binary never ran", r.returncode == 4 and not (S / "args.txt").exists())
    (S / "args.txt").unlink(missing_ok=True)
    r = run([*base, sha, str(dummy), str(watch), "run"], {**e, "FAKE_SMI_DEAD": "1"})
    check("hold.sh: nvidia-smi dead at the pre check -> exit 2 (infrastructure STOP), never 4",
          r.returncode == 2 and "STOP" in r.stdout)
    r = run([*base, sha, str(dummy), str(watch), "run"], {**e, "FAKE_NAME_EMPTY": "1"})
    check("hold.sh: no GPU name from nvidia-smi -> exit 2 and the binary never runs",
          r.returncode == 2 and not (S / "args.txt").exists())


def main():
    runner()
    finder()
    gguf()
    _, env = card()
    hold(env)
    print("TOOLS SELFTEST", "OK" if all(results) else "FAILED", f"({len(results)} checks)")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
