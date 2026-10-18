#!/usr/bin/env python3
"""Stand-in run data for the self-tests (the part the rule tests need), built from the oracle's stored runs (llama.cpp data in the 'fifi' slot).
The three frozen references are used: cpu/clean (committed here), cpu/threads6 and cuda/follow, cuda/free (scratch oracle)."""
import copy
import json
from pathlib import Path

import a5_eval as ev

SCRATCH_ORACLE = Path("/media/dan/nvme/fifi/scrch/c62fd02/oracle/qwen36")
EXE = "e" * 64
COMMIT = "c0ffee0".ljust(40, "0")
GGUF = "671e47e0ec53c665d048b98c3ecbfd5236b5ca9c3e02ed19fc8f81f7b85140c7"


def _frozen_sha(rel):
    """The sha256 the stand-in file must have: the one frozen in the gated file when it lists the file (cpu/clean, cpu/threads6,
    cuda/follow), else the one in the scratch oracle's own MANIFEST.json (cuda/free)."""
    frozen = ev.read_json(ev.HERE / "a5-gated-steps.json")["inputs_sha256"]
    if rel in frozen:
        return frozen[rel]
    return ev.read_json(SCRATCH_ORACLE / "MANIFEST.json")["files"][rel]["sha256"]


def oracle_steps(run_dir, pid):
    base = SCRATCH_ORACLE if run_dir != "cpu/clean" else ev.HERE / "oracle"
    path = base / run_dir / pid / "run.json"
    rel = f"{run_dir}/{pid}/run.json"
    if ev.sha_file(path) != _frozen_sha(rel):
        raise ev.Stop(f"stand-in file {rel} does not have the frozen sha256")
    return json.loads(path.read_text())["steps"]


BASE_NS = 1_700_000_000_000_000_000   # a CLOCK_REALTIME origin for the fabricated windows


def make_run(ref, run_dir, pid, mode, ms=50.0, salt=0, captured=True, start_ns=BASE_NS):
    """One valid run of `pid`; mode forced = teacher-forced on the reference stream, free = own argmax stream."""
    st = oracle_steps(run_dir, pid)
    toks = [s["token"] for s in ref.streams[pid]]
    steps = []
    for i, s in enumerate(st):
        if i == 0:
            fed = ref.last_prompt_token[pid]
        else:
            fed = toks[i - 1] if mode == "forced" else st[i - 1]["argmax"]
        fl = None
        if mode == "forced":
            fl = next(v for t, v in s["top"] if t == toks[i])
        steps.append({"in_token": fed, "argmax": s["argmax"], "argmax_logit": s["top"][0][1], "top32": s["top"],
                      "forced_logit": fl, "hash": s["hash"], "lse": s["lse"]})
    ns = int(ms * 128 * 1e6) + salt
    ph = {"replay": int(ns * .9), "readback": int(ns * .03), "host": int(ns * .07)}
    each = int(ns * 0.99 / 128.2)
    step_ns = [int(each * 1.2)] + [each] * 127            # a first step a little slower than the rest (under 2x)
    return {"captured": captured, "graph_ready_before_timed": captured, "decode_ns": ns, "phase_ns": ph, "step_ns": step_ns,
            "analysis_ns": 1_800_000 * 128, "syncs": 128, "host_writes": 128, "prompt_synced_unix_ns": start_ns,
            "timed_start_unix_ns": start_ns + 1_000_000_000, "timed_end_unix_ns": start_ns + 1_000_000_000 + ns, "steps": steps}


def clone(d):
    return copy.deepcopy(d)
