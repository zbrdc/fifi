#!/usr/bin/env python3
"""RD-10 evaluator library: A5 exactness (frozen gated step sets), determinism, speed, raw-file validation.
CPU only, stdlib only, run as `python3 -I`. Design guidance only, no code copied.

Every input is a tracked file next to this script (qualification/cuda-rd/): a5-gated-steps.json and oracle/ (the five
cpu/clean run.json files and the prompt id files). Nothing is read from scratch. The rule is the one frozen by REPLY-40,
REPLY-41 and the 0047 amendment 2026-10-10: there is NO excluded-share rule (REPLY-69); the share is information only.

Exit codes of every command-line tool of this packet:
  0 PASS   1 FAIL (closed result, report written)   2 STOP (input or infrastructure problem, message "STOP: ...", no verdict)
  3 REFUSED (admission refusal such as not enough free memory after the load; report written, not a FAIL)
  4 PRE-RUN complete (eager pre-run, never a stage gate; the console line never contains PASS)   5 VOIDED (a clock rule voided a run: repeat the call, never a FAIL)
  7 BLOCKED (three consecutive voided runs: a clock or thermal problem; never a FAIL and never a pass)

Raw input schema "cuda-rd-raw/2" (written by cuda-rd-run, one file per GPU call):
 {"schema": "cuda-rd-raw/2",
  "identities": {"pid", "started_unix_ns", "capture_requested": "graph"|"eager", "exe_sha256", "engine_commit", "gguf_sha256", "profile", "opt_level",
                 "debug", "kv_mode": "fp16", "gpu", "driver", "mem_total", "mem_free_before", "mem_free_after_load"},
  "mode": "free"|"forced", "incomplete": bool, "refused": null | {"kind": "memory", ...},
  "prompts": [{"id": "P1", "ids_sha256": "...", "n_prompt": 1024, "forced_stream_sha256": "... (forced files)",
     "runs": [{"captured": bool, "decode_ns": int, "phase_ns": {"replay","readback","host"}, "analysis_ns": int, "syncs": int, "host_writes": int,
               "steps": [{"in_token": int, "argmax": int, "argmax_logit": float, "top32": [[id, logit] x32 (at least 5)],
                          "forced_logit": float|null, "hash": "fnv1a64 hex", "lse": float}] x128,
               "graph_ready_before_timed": bool, "step_ns": [int] x128, "prompt_synced_unix_ns": int,
               "timed_start_unix_ns": int, "timed_end_unix_ns": int}]}]}
Per call the run script also keeps NAME.watch.jsonl: one card sample every 0.5 s while the binary ran (see idle_check.py).
Step i is the logit row after the prompt and i generated tokens (oracle numbering). decode_ns covers the 128 decode steps
(device step, token read-back and the copy of the 993 KB row into a preallocated host buffer); the host analysis (top-32,
hash, lse) runs after the clock stops and is reported as analysis_ns.
"""

import hashlib
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW_SCHEMA = "cuda-rd-raw/2"
PROMPTS = ["P1", "P2", "P3", "P4", "P5"]
GATED_SHA256 = "18d329a693be4983c73caddba470e5643943b0f014b68aa43670669bf726d49a"
GATE_TOKS = {"1": 18.0, "1b": 35.0, "2": 100.0}
BREAKDOWN_TOL = 0.03
N_STEPS = 128
PHASES = ("replay", "readback", "host")
MIN_FREE_AFTER_LOAD = 1 << 30
FIRST_STEP_FLAG = 2.0           # the first timed step is flagged (never dropped) when over this multiple of the median step
IDENT_KEYS = ("exe_sha256", "engine_commit", "gguf_sha256", "profile", "kv_mode", "gpu", "driver")
# file name -> (mode, prompts, runs per prompt, eager comparison file)
FILES = {
    "A1": ("forced", ["P1", "P3"], 1, False), "A2": ("forced", ["P2", "P4", "P5"], 1, False),
    "E1": ("forced", ["P1", "P3"], 1, True), "E2": ("forced", ["P2", "P4", "P5"], 1, True),
    "B1": ("free", ["P2", "P3", "P4"], 1, False), "B2": ("free", ["P2", "P3", "P4"], 1, False),
    "B3": ("free", ["P2", "P3", "P4"], 1, False), "B4": ("free", ["P5"], 1, False), "B5": ("free", ["P5"], 1, False),
    "B6": ("free", ["P5"], 1, False),
    "C1": ("free", ["P1"], 1, False), "C2": ("free", ["P1"], 1, False), "C3": ("free", ["P1"], 1, False),
    "C4": ("free", ["P1"], 1, False), "C5": ("free", ["P1"], 1, False),
}


class Stop(Exception):
    """Input or infrastructure problem: exit 2, no verdict."""


class Refused(Exception):
    """Admission refusal (memory): exit 3, reported as REFUSED."""


class Voided(Exception):
    """A clock rule voided a run (environmental, not a FAIL): exit 5, the call is repeated."""


class Blocked(Exception):
    """Three consecutive voided runs: exit 7, BLOCKED, never a FAIL and never a pass."""


def sha_file(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError as e:
        raise Stop(f"cannot read {path}: {e.strerror}") from e


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as e:
        raise Stop(f"cannot read JSON {path}: {e}") from e


def _finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


class Reference:
    """The frozen gated set plus the reference stream derived from the hashed cpu/clean run files."""

    def __init__(self, gated_path=HERE / "a5-gated-steps.json", oracle=HERE / "oracle", expect=GATED_SHA256):
        got = sha_file(gated_path)
        if got != expect:
            raise Stop(f"a5-gated-steps.json sha256 {got} != frozen {expect}")
        self.gated_sha256 = got
        self.gated = read_json(gated_path)
        if self.gated.get("schema") != "a5-gated-steps/1":
            raise Stop("unexpected gated-steps schema")
        self.oracle = Path(oracle)
        self.verified = {}
        self.not_committed = []
        for rel, want in sorted(self.gated["inputs_sha256"].items()):
            p = self.oracle / rel
            if not p.exists():
                self.not_committed.append(rel)
                continue
            have = sha_file(p)
            if have != want:
                raise Stop(f"{rel}: sha256 {have} != the one frozen in a5-gated-steps.json {want}")
            self.verified[rel] = have
        for pid in PROMPTS:
            for rel in (f"cpu/clean/{pid}/run.json", f"prompts/{pid}.ids.json"):
                if rel not in self.verified:
                    raise Stop(f"required reference input {rel} is not committed or not hashed")
        self.last_prompt_token = {}
        for pid in PROMPTS:
            ids = (self.oracle / f"prompts/{pid}.ids").read_text().split()
            self.last_prompt_token[pid] = int(ids[-1])
        self.streams, self.lse = {}, {}
        for pid in PROMPTS:
            steps = read_json(self.oracle / "cpu/clean" / pid / "run.json")["steps"]
            self.streams[pid] = [{"token": s["token"], "top": s["top"], "margin": s["top"][0][1] - s["top"][1][1]}
                                 for s in steps]
            self.lse[pid] = [s["lse"] for s in steps]
            for g in self.gated["prompts"][pid]["gated"]:
                st = self.streams[pid][g["step"]]
                if st["token"] != g["reference_token"] or abs(st["top"][0][1] - g["ref_top1_logit"]) > 1e-6:
                    raise Stop(f"{pid} step {g['step']}: gated set and cpu/clean run disagree")

    def block(self, pid):
        return self.gated["prompts"][pid]

    def excluded_share(self, pid):
        b = self.block(pid)
        return b["n_excluded"], b["n_steps"], b["excluded_share_percent"]


def eval_a5(run, ids_sha256, ref, pid):
    """Grade ONE teacher-forced run of one prompt. 'ok' is the A5 verdict: every gated step must pass."""
    block = ref.block(pid)
    steps = run["steps"]
    out = {"n_steps": len(steps), "n_gated": len(block["gated"]), "fails": [], "max_logit_dev": 0.0,
           "top5_set_agree": 0, "lse_abs_diff_max": 0.0}
    if ids_sha256 != block["prompt_ids_sha256"]:
        raise Stop("prompt id list hash differs from the frozen one")
    if len(steps) != block["n_steps"]:
        out["fails"].append({"step": None, "reason": f"{len(steps)} steps, expected {block['n_steps']}"})
        out["ok"] = False
        return out
    for g in block["gated"]:
        s = steps[g["step"]]
        al, fl = s.get("argmax_logit"), s.get("forced_logit")
        if s["argmax"] != g["reference_token"]:
            out["fails"].append({"step": g["step"], "reason": "top1 differs", "got": s["argmax"], "want": g["reference_token"]})
            continue
        if not (_finite(fl) and _finite(al)):
            out["fails"].append({"step": g["step"], "reason": "non-finite logit"})
            continue
        if fl != al or s["top32"][0][0] != s["argmax"] or s["top32"][0][1] != al:
            out["fails"].append({"step": g["step"], "reason": "argmax_logit, forced_logit and top32[0] disagree"})
            continue
        dev = abs(al - g["ref_top1_logit"])
        out["max_logit_dev"] = max(out["max_logit_dev"], dev)
        if dev > g["tolerance"]:
            out["fails"].append({"step": g["step"], "reason": "top1 logit outside 2 S_t", "dev": dev, "tolerance": g["tolerance"]})
    for i, s in enumerate(steps):
        if {t for t, _ in s["top32"][:5]} == {t for t, _ in ref.streams[pid][i]["top"]}:
            out["top5_set_agree"] += 1
        if _finite(s.get("lse")):
            out["lse_abs_diff_max"] = max(out["lse_abs_diff_max"], abs(s["lse"] - ref.lse[pid][i]))
    out["ok"] = not out["fails"]
    return out


def eval_free(runs, ref, pid):
    """First divergence of the free-run greedy stream from the reference stream (information, not the gate)."""
    res = []
    for run in runs:
        toks = [s["argmax"] for s in run["steps"]]
        want = [s["token"] for s in ref.streams[pid]]
        div = next((i for i, (a, b) in enumerate(zip(toks, want)) if a != b), None)
        res.append({"first_divergence": div, "margin_at_divergence": None if div is None else ref.streams[pid][div]["margin"]})
    return res


def eval_determinism(runs, need):
    """At least `need` runs, all with identical token ids and identical row-hash sequences."""
    sig = [(tuple(s["argmax"] for s in r["steps"]), tuple(s["hash"] for s in r["steps"])) for r in runs]
    return {"runs": len(runs), "ok": len(runs) >= max(need, 2) and all(x == sig[0] for x in sig)}


def eval_tf_vs_free(free_run, forced_run, first_div):
    """Same context up to and including the first divergence row, so those row hashes must be equal."""
    last = N_STEPS - 1 if first_div is None else first_div
    bad = [i for i in range(last + 1) if free_run["steps"][i]["hash"] != forced_run["steps"][i]["hash"]]
    return {"rows_compared": last + 1, "mismatch_rows": bad, "ok": not bad}


def counts_ok(run, field):
    """True when the step's own `field` counter ("syncs" or "host_writes") of a run is what the gate expects: exactly one per
    token on a plain run (the read-back wait; the start-block write). A tiered run (it carries a `tiered` member) also waits
    on the host in every layer and at its bank fills, so it needs at least one per token and no more is checked here.
    The `tiered` member must be a dict (a bare key is not enough). TCAP C6 makes a tiered token report one sync again;
    this lower bound goes back to exact then."""
    value = run.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        return False
    return value >= N_STEPS if isinstance(run.get("tiered"), dict) else value == N_STEPS


def eval_speed(runs, stage="1"):
    """Median tok/s over the runs (decode wall time only) and the ms/token breakdown of the median run. Every run is
    checked, by `counts_ok`, for its syncs and host_writes counters (the step's own, self-reports): a plain run must
    report exactly 128 of each (one host-blocking wait and one start-block write per token), a run that carries the
    `tiered` member (a dict) at least 128 of each (its layers add waits and blocking writes).
    The first timed step of every run is reported beside the median step and flagged when over 2x (never dropped)."""
    rates = [N_STEPS / (r["decode_ns"] / 1e9) for r in runs]
    med = statistics.median(rates)
    order = sorted(range(len(runs)), key=lambda i: rates[i])
    mid = runs[order[len(order) // 2]]
    ph = {k: mid["phase_ns"][k] / 1e6 / N_STEPS for k in PHASES}
    ms_tok = 1000.0 / med
    err = abs(sum(ph.values()) - ms_tok) / ms_tok
    first = [(r["step_ns"][0] / 1e6, statistics.median(r["step_ns"]) / 1e6) for r in runs]
    return {"tok_s": rates, "median": med, "gate": GATE_TOKS[stage], "pass": med >= GATE_TOKS[stage], "ms_per_token": ms_tok,
            "breakdown_ms": ph, "breakdown_err": err, "breakdown_ok": err <= BREAKDOWN_TOL,
            "syncs_ok": all(counts_ok(r, "syncs") for r in runs),
            "host_writes_ok": all(counts_ok(r, "host_writes") for r in runs),
            "analysis_ms": mid["analysis_ns"] / 1e6 / N_STEPS, "spread": (max(rates) - min(rates)) / med,
            "first_step_ms": first[order[len(order) // 2]][0], "median_step_ms": first[order[len(order) // 2]][1],
            "first_step_flagged": [i for i, (f, m) in enumerate(first) if f > FIRST_STEP_FLAG * m]}
