#!/usr/bin/env python3
"""Stand-in raw FILES for the self-tests: whole calls, card samples and the files on disk (llama.cpp data in the 'fifi' slot).
Re-exports the run builders of fixtures.py so a test needs only `import fixtures_files as fx`."""
import json

import a5_eval as ev
from fixtures import BASE_NS, COMMIT, EXE, GGUF, clone, make_run, oracle_steps  # noqa: F401  (re-exported)


def make_file(ref, name, ms=50.0, salt=0, capture="graph", exe=EXE, pid=1000, free_dir="cuda/free", forced_dir="cuda/follow"):
    mode, prompts, nruns, eager = ev.FILES[name]
    requested = "eager" if (eager or capture == "eager") else "graph"
    plist = []
    for k, p in enumerate(prompts):
        run = make_run(ref, forced_dir if mode == "forced" else free_dir, p, mode, ms, salt + k,
                       captured=(requested == "graph"), start_ns=BASE_NS + k * 60_000_000_000)
        blk = ref.block(p)
        plist.append({"id": p, "ids_sha256": blk["prompt_ids_sha256"], "n_prompt": blk["n_prompt_tokens"], "runs": [run],
                      **({"forced_stream_sha256": ref.verified[f"cpu/clean/{p}/run.json"]} if mode == "forced" else {})})
    return {"schema": ev.RAW_SCHEMA, "mode": mode, "incomplete": False, "refused": None,
            "identities": {"capture_requested": requested, "exe_sha256": exe, "engine_commit": COMMIT, "gguf_sha256": GGUF,
                           "profile": "release", "opt_level": "3", "debug": False, "kv_mode": "fp16", "gpu": "NVIDIA GeForce RTX 3090",
                           "driver": "580.159.03", "mem_total": 25_769_803_776, "mem_free_before": 24_195_497_984,
                           "mem_free_after_load": 3_887_726_592, "pid": pid, "started_unix_ns": 1_000_000 + pid},
            "prompts": plist}


def make_all(ref, ms=50.0, capture="graph"):
    docs = {}
    for k, name in enumerate(sorted(ev.FILES)):
        docs[name] = make_file(ref, name, ms=ms + (0.05 * k if name.startswith("C") else 0), salt=1000 * (k + 1),
                               capture=capture if not ev.FILES[name][3] else "eager", pid=2000 + k)
    return docs


def make_watch(doc, busy_mhz=1800, idle_mhz=210, floor_dip=None, foreign_pid=None, busy_util=98):
    """Card samples every 0.5 s around every run: idle before and after (an idle 3090 downclocks), busy inside the timed region."""
    out = []
    for p in doc.get("prompts", []):
        r = p["runs"][0]
        t0, t1 = r["timed_start_unix_ns"], r["timed_end_unix_ns"]
        t = t0 - 3_000_000_000
        while t <= t1 + 3_000_000_000:
            inside = t0 <= t <= t1
            s = {"t": t / 1e9, "clocks.sm": str(busy_mhz if inside else idle_mhz), "utilization.gpu": str(busy_util if inside else 0),
                 "apps": [doc["identities"].get("pid", 0)] + ([foreign_pid] if (inside and foreign_pid) else [])}
            out.append(s)
            t += 500_000_000
        if floor_dip is not None:
            mid = (t0 + t1) // 2
            out.append({"t": mid / 1e9, "clocks.sm": str(floor_dip), "utilization.gpu": str(busy_util), "apps": [doc["identities"].get("pid", 0)]})
    return sorted(out, key=lambda s: s["t"])


def write_all(docs, raw, watch=None):
    raw.mkdir(parents=True, exist_ok=True)
    for old in raw.glob("*"):
        old.unlink()
    for n, d in docs.items():
        (raw / f"{n}.json").write_text(json.dumps(d))
        w = (watch or {}).get(n)
        if w is None and not d.get("incomplete"):
            w = make_watch(d)
        if w is not None:
            (raw / f"{n}.watch.jsonl").write_text("\n".join(json.dumps(s) for s in w) + "\n")
