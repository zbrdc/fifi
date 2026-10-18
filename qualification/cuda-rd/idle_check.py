#!/usr/bin/env python3
"""RD-10 GPU state tool, used only by hold.sh INSIDE a gpu_run.sh hold. Read-only nvidia-smi queries.
  idle_check.py pre                 exit 0 = idle enough to time, 1 = not idle, 2 = nvidia-smi failed
  idle_check.py watch FILE PID      sample the card every 0.5 s into FILE (JSON lines, field t = CLOCK_REALTIME seconds) until PID ends
  idle_check.py post                record temperature and clocks after the run (never fails the call)
The samples are JUDGED later, by the evaluator, against the timed region of each run only (a5_eval.judge_window): the clock
floor never comes from the pre-run check, because an idle 3090 downclocks. Every state line printed starts with HOLD.
Idle = no compute process, utilization <= 15 percent (the desktop alone is 5 to 9), temperature <= 75 C, memory used <= 3 GiB.
"""
import json
import os
import subprocess
import sys
import time

Q = "utilization.gpu,memory.used,clocks.sm,clocks.mem,temperature.gpu,power.draw,driver_version,pstate"
INTERVAL = 0.5
# A call that starts right after another one reads the previous process's last utilization for about a second:
# when utilization is the only reason for "not idle", sample again after a short settle (5 tries of 5 s = 25 s).
SETTLE_TRIES = 5
SETTLE_SECONDS = float(os.environ.get("RD10_IDLE_SETTLE_SECS", "5"))
UTILIZATION_ONLY = ["utilization above 15"]


def smi(args):
    try:
        p = subprocess.run(["nvidia-smi", *args, "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as e:
        print("STOP: nvidia-smi failed:", e)
        sys.exit(2)
    if p.returncode != 0:
        print("STOP: nvidia-smi failed:", p.stderr.strip())
        sys.exit(2)
    return [ln.strip() for ln in p.stdout.splitlines() if ln.strip()]


def sample():
    d = dict(zip(Q.split(","), [x.strip() for x in smi([f"--query-gpu={Q}"])[0].split(",")]))
    d["apps"] = sorted(int(ln.split(",")[0]) for ln in smi(["--query-compute-apps=pid,used_memory"]))
    d["t"] = time.time()
    return d


def why_not_idle(d):
    bad = []
    if d["apps"]:
        bad.append("compute process present")
    if float(d["utilization.gpu"]) > 15:
        bad.append("utilization above 15")
    if float(d["temperature.gpu"]) > 75:
        bad.append("temperature above 75")
    if float(d["memory.used"]) > 3072:
        bad.append("memory used above 3 GiB")
    return bad


def pre():
    for attempt in range(SETTLE_TRIES + 1):
        d = sample()
        bad = why_not_idle(d)
        if bad != UTILIZATION_ONLY or attempt == SETTLE_TRIES:
            break
        print(f"HOLD settle {attempt + 1}: utilization {d['utilization.gpu']} percent, sampling again in {SETTLE_SECONDS:g} s")
        time.sleep(SETTLE_SECONDS)
    d["idle"], d["why_not"] = not bad, bad
    print("HOLD pre " + json.dumps(d, sort_keys=True))
    return 0 if not bad else 1


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def watch(path, pid):
    with open(path, "w") as f:
        while alive(pid):
            f.write(json.dumps(sample(), sort_keys=True) + "\n")
            f.flush()
            time.sleep(INTERVAL)
    return 0


def post():
    print("HOLD post " + json.dumps(sample(), sort_keys=True))
    return 0


if __name__ == "__main__":
    a = sys.argv[1:] or ["pre"]
    sys.exit({"pre": lambda: pre(), "post": lambda: post(), "watch": lambda: watch(a[1], int(a[2]))}[a[0]]())
