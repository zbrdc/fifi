"""RD-10 raw-file validation: typed STOP unless a raw file is exactly what the plan says its call produces.
CPU only, stdlib only, `python3 -I`. Design guidance only, no code copied.
"""
import json
import re

import a5_eval as ev



def validate_raw(name, doc, ref):
    """Typed STOP unless the raw file is exactly what the plan says call `name` produces."""
    mode, prompts, nruns, eager = ev.FILES[name]
    if doc.get("schema") != ev.RAW_SCHEMA:
        raise ev.Stop(f"{name}: schema {doc.get('schema')} != {ev.RAW_SCHEMA}")
    if doc.get("refused"):
        raise ev.Refused(f"{name}: refused {json.dumps(doc['refused'], sort_keys=True)}")
    if doc.get("mode") != mode:
        raise ev.Stop(f"{name}: mode {doc.get('mode')} != {mode}")
    ident = doc.get("identities") or {}
    for k in ev.IDENT_KEYS + ("pid", "started_unix_ns"):
        if not ident.get(k):
            raise ev.Stop(f"{name}: identity {k} missing")
    if not re.fullmatch(r"[0-9a-f]{40}", str(ident["engine_commit"])):
        raise ev.Stop(f"{name}: engine_commit {ident['engine_commit']!r} is not a full 40-character commit (unknown or abbreviated)")
    if "3090" not in str(ident["gpu"]) or not re.fullmatch(r"\d+\.\d+(\.\d+)?", str(ident["driver"])):
        raise ev.Stop(f"{name}: gpu {ident['gpu']!r} / driver {ident['driver']!r} (need an RTX 3090 name and a numeric driver version)")
    if ident["kv_mode"] != "fp16" or ident["profile"] != "release":
        raise ev.Stop(f"{name}: kv_mode {ident['kv_mode']} / profile {ident['profile']} (stage gates need fp16 and release)")
    if ident["gguf_sha256"] != ref.gated["gguf"]["sha256"]:
        raise ev.Stop(f"{name}: gguf_sha256 differs from the frozen GGUF {ref.gated['gguf']['sha256']}")
    free_after = ident.get("mem_free_after_load")
    if not isinstance(free_after, int) or not isinstance(ident.get("mem_free_before"), int):
        raise ev.Stop(f"{name}: cudaMemGetInfo figures missing")
    if free_after < ev.MIN_FREE_AFTER_LOAD:
        raise ev.Refused(f"{name}: only {free_after} B free after the load (< 1 GiB headroom, decision 0040 A2)")
    if ident.get("capture_requested") not in ("graph", "eager") or (eager and ident["capture_requested"] != "eager"):
        raise ev.Stop(f"{name}: capture_requested {ident.get('capture_requested')}")
    if doc.get("incomplete"):
        return
    got = [p.get("id") for p in doc.get("prompts", [])]
    if got != prompts:
        raise ev.Stop(f"{name}: prompts {got} != {prompts}")
    for p in doc["prompts"]:
        block = ref.block(p["id"])
        if p.get("ids_sha256") != block["prompt_ids_sha256"] or p.get("n_prompt") != block["n_prompt_tokens"]:
            raise ev.Stop(f"{name} {p['id']}: ids_sha256 or n_prompt differs from the frozen prompt set")
        if mode == "forced" and p.get("forced_stream_sha256") != ref.verified[f"cpu/clean/{p['id']}/run.json"]:
            raise ev.Stop(f"{name} {p['id']}: forced_stream_sha256 is not the hash of the frozen cpu/clean stream")
        if len(p.get("runs", [])) != nruns:
            raise ev.Stop(f"{name} {p['id']}: {len(p.get('runs', []))} runs != {nruns}")
        for r in p["runs"]:
            _validate_run(f"{name} {p['id']}", p["id"], r, (eager, mode), ref)


def _validate_run(where, pid, r, kind, ref):
    eager, mode = kind
    if len(r.get("steps", [])) != ev.N_STEPS:
        raise ev.Stop(f"{where}: {len(r.get('steps', []))} steps != {ev.N_STEPS}")
    relation = ">=" if isinstance(r.get("tiered"), dict) else "=="
    if not ev.counts_ok(r, "syncs"):
        raise ev.Stop(f"{where}: syncs {r.get('syncs')} not {relation} {ev.N_STEPS} (one id-free sync per token; a tiered run adds its in-layer waits)")
    if not ev.counts_ok(r, "host_writes"):
        raise ev.Stop(f"{where}: host_writes {r.get('host_writes')} not {relation} {ev.N_STEPS} (one host write per token: the start-block write, lane C Q-C6; a tiered run adds its blocking writes)")
    if not isinstance(r.get("captured"), bool) or (eager and r["captured"]):
        raise ev.Stop(f"{where}: captured flag {r.get('captured')} not allowed here")
    for k in ("prompt_synced_unix_ns", "timed_start_unix_ns", "timed_end_unix_ns"):
        if not isinstance(r.get(k), int) or r[k] <= 0:
            raise ev.Stop(f"{where}: {k} missing or not a positive integer")
    if not r["timed_start_unix_ns"] < r["timed_end_unix_ns"] or \
            r["timed_end_unix_ns"] - r["timed_start_unix_ns"] < 0.97 * r.get("decode_ns", 0):
        raise ev.Stop(f"{where}: the timed window is shorter than decode_ns")
    if not isinstance(r.get("graph_ready_before_timed"), bool):
        raise ev.Stop(f"{where}: graph_ready_before_timed missing")
    sn = r.get("step_ns")
    if not isinstance(sn, list) or len(sn) != ev.N_STEPS or not all(isinstance(x, int) and x >= 0 for x in sn) \
            or not 0.9 * r.get("decode_ns", 0) <= sum(sn) <= r.get("decode_ns", 0):
        raise ev.Stop(f"{where}: step_ns must be 128 integers adding up to between 90 and 100 percent of decode_ns")
    ph = r.get("phase_ns", {})
    if set(ph) != set(ev.PHASES) or not all(isinstance(v, int) and v >= 0 for v in ph.values()):
        raise ev.Stop(f"{where}: phase_ns malformed")
    if not isinstance(r.get("decode_ns"), int) or r["decode_ns"] <= 0 or not isinstance(r.get("analysis_ns"), int):
        raise ev.Stop(f"{where}: decode_ns/analysis_ns malformed")
    if abs(sum(ph.values()) - r["decode_ns"]) > ev.BREAKDOWN_TOL * r["decode_ns"]:
        raise ev.Stop(f"{where}: phases do not sum to decode_ns within 3 percent")
    ref_tokens = [s["token"] for s in ref.streams[pid]]
    for i, s in enumerate(r["steps"]):
        _validate_step(f"{where} step {i}", s, mode)
    if r["steps"][0]["in_token"] != ref.last_prompt_token[pid]:
        raise ev.Stop(f"{where}: step 0 was not fed the last prompt token")
    for i in range(ev.N_STEPS - 1):
        want = ref_tokens[i] if mode == "forced" else r["steps"][i]["argmax"]
        if r["steps"][i + 1]["in_token"] != want:
            raise ev.Stop(f"{where}: input of step {i + 1} is not the {'reference' if mode == 'forced' else 'own argmax'} token (teacher forcing broken)")


def _validate_step(where, s, mode):
    t = s.get("top32")
    if not isinstance(t, list) or len(t) < 5 or len(s.get("hash", "")) != 16 or not isinstance(s.get("in_token"), int):
        raise ev.Stop(f"{where}: top32 shorter than 5, hash or in_token malformed")
    if not all(isinstance(e, list) and len(e) == 2 and ev._finite(e[1]) for e in t):
        raise ev.Stop(f"{where}: non-finite or malformed top32 entry")
    if any(t[k][1] < t[k + 1][1] for k in range(len(t) - 1)):
        raise ev.Stop(f"{where}: top32 not in descending order")
    if t[0][0] != s.get("argmax") or t[0][1] != s.get("argmax_logit") or not ev._finite(s.get("argmax_logit")):
        raise ev.Stop(f"{where}: top32[0] disagrees with argmax/argmax_logit (or non-finite)")
    if not ev._finite(s.get("lse")):
        raise ev.Stop(f"{where}: lse not finite")
    if mode == "forced" and not ev._finite(s.get("forced_logit")):
        raise ev.Stop(f"{where}: forced_logit not finite")
    if mode == "free" and s.get("forced_logit") is not None:
        raise ev.Stop(f"{where}: forced_logit present in a free run")


def check_identities(docs):
    """All raw files come from one binary, one commit, one GGUF, and are distinct calls and distinct runs."""
    first, seen_calls, seen_ns = None, set(), {}
    for name, d in sorted(docs.items()):
        ident = d["identities"]
        if first is None:
            first = (name, ident)
        for k in ev.IDENT_KEYS:
            if ident[k] != first[1][k]:
                raise ev.Stop(f"{name}: identity {k} differs from {first[0]} ({ident[k]} vs {first[1][k]}): mixed binaries or commits")
        call = (ident["pid"], ident["started_unix_ns"])
        if call in seen_calls:
            raise ev.Stop(f"{name}: same pid and start time as another file (a copied file)")
        seen_calls.add(call)
        for p in d.get("prompts", []):
            for r in p["runs"]:
                if r["decode_ns"] in seen_ns:
                    raise ev.Stop(f"{name} {p['id']}: decode_ns equal to the run in {seen_ns[r['decode_ns']]} (a copied run)")
                seen_ns[r["decode_ns"]] = f"{name} {p['id']}"
    return first[1]


def capture_state(docs, prerun_flag):
    """Derive the pre-run label from the DATA. Returns (prerun, fail_reasons)."""
    gate = {n: d for n, d in docs.items() if not ev.FILES[n][3]}
    eager_files = {n: d for n, d in docs.items() if ev.FILES[n][3]}
    req = {d["identities"]["capture_requested"] for d in gate.values()}
    if len(req) != 1:
        raise ev.Stop(f"gate files disagree on capture_requested: {sorted(req)}")
    for n, d in eager_files.items():
        if d["identities"]["capture_requested"] != "eager":
            raise ev.Stop(f"{n}: the eager comparison must request eager")
    runs = [(n, p["id"], r) for n, d in gate.items() if not d.get("incomplete") for p in d["prompts"] for r in p["runs"]]
    uncaptured = [(n, pid) for n, pid, r in runs if not r["captured"]]
    if req == {"eager"}:
        if len(uncaptured) != len(runs):
            raise ev.Stop("eager requested but some runs report captured = true")
        if not prerun_flag:
            raise ev.Stop("every run is eager but --prerun was not given: refusing to label eager data as a gate run")
        return True, []
    if prerun_flag:
        raise ev.Stop("--prerun given but the data asks for graph mode")
    why = [f"graph mode ran uncaptured in {n} {pid} (silent fallback; relabel as an eager pre-run)" for n, pid in uncaptured]
    return False, why


def check_head(ident, head):
    """The commit the binary was built from is the commit under test (full 40-character comparison)."""
    if not re.fullmatch(r"[0-9a-f]{40}", head):
        raise ev.Stop(f"--head {head!r} is not a full 40-character commit")
    if ident["engine_commit"] != head:
        raise ev.Stop(f"engine_commit {ident['engine_commit']} is not the commit under test {head}")
