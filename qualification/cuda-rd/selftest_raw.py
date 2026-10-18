#!/usr/bin/env python3
"""CPU self-test of rawcheck.py: every validation of a raw file, the cross-file checks and the capture state.
Run: python3 -I selftest_raw.py. One line per check; exit 1 on any unexpected result."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a5_eval as ev  # noqa: E402
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
    for n, d in docs.items():
        rw.validate_raw(n, d, ref)
    check("validate: the 12 stand-in files are valid", True)

    def bad(name, fn, exc=ev.Stop):
        d = fx.clone(docs[name])
        fn(d)
        return raises(exc, rw.validate_raw, name, d, ref)
    run0 = lambda d: d["prompts"][0]["runs"][0]  # noqa: E731
    step_ = lambda d, i=40: run0(d)["steps"][i]  # noqa: E731
    cases = [
        ("old schema /1", "C1", lambda d: d.__setitem__("schema", "cuda-rd-raw/1")),
        ("mode swapped (forced file says free)", "A1", lambda d: d.__setitem__("mode", "free")),
        ("mode swapped (free file says forced)", "C1", lambda d: d.__setitem__("mode", "forced")),
        ("wrong ids_sha256", "C1", lambda d: d["prompts"][0].__setitem__("ids_sha256", "0" * 64)),
        ("n_prompt 3 (short context)", "C1", lambda d: d["prompts"][0].__setitem__("n_prompt", 3)),
        ("wrong prompt set (P1 smuggled into B1)", "B1", lambda d: d["prompts"].append(fx.clone(d["prompts"][0]))),
        ("two runs where one is expected", "C1", lambda d: d["prompts"][0]["runs"].append(fx.clone(run0(d)))),
        ("100 steps", "C1", lambda d: run0(d).__setitem__("steps", run0(d)["steps"][:100])),
        ("host_writes 256 on a run (lane C Q-C6: one start-block write per token)", "A1", lambda d: run0(d).__setitem__("host_writes", 256)),
        ("host_writes missing", "C1", lambda d: run0(d).pop("host_writes")),
        ("syncs 256 on a run", "A1", lambda d: run0(d).__setitem__("syncs", 256)),
        ("decode_ns zero", "C1", lambda d: run0(d).__setitem__("decode_ns", 0)),
        ("phases 20 percent short of decode_ns", "C1", lambda d: run0(d)["phase_ns"].__setitem__("replay", int(run0(d)["phase_ns"]["replay"] * .7))),
        ("null lse at an EXCLUDED step", "A1", lambda d: step_(d, ref.block("P1")["excluded"][0]["step"]).__setitem__("lse", None)),
        ("NaN in top32 at any step", "C1", lambda d: step_(d)["top32"][3].__setitem__(1, float("nan"))),
        ("null argmax_logit", "C1", lambda d: step_(d).__setitem__("argmax_logit", None)),
        ("argmax != top32[0]", "C1", lambda d: step_(d).__setitem__("argmax", step_(d)["argmax"] + 1)),
        ("top32 shorter than 5", "C1", lambda d: step_(d).__setitem__("top32", step_(d)["top32"][:3])),
        ("top32 not descending", "C1", lambda d: step_(d)["top32"].reverse()),
        ("teacher forcing broken (forced input differs)", "A1", lambda d: step_(d, 41).__setitem__("in_token", 5)),
        ("free-run input is not the own argmax", "C1", lambda d: step_(d, 41).__setitem__("in_token", 5)),
        ("step 0 not fed the last prompt token", "C1", lambda d: step_(d, 0).__setitem__("in_token", 1)),
        ("forced_logit in a free run", "C1", lambda d: step_(d).__setitem__("forced_logit", 1.0)),
        ("forced stream not the frozen cpu/clean stream", "A1", lambda d: d["prompts"][0].__setitem__("forced_stream_sha256", "2" * 64)),
        ("kv_mode int8", "C1", lambda d: d["identities"].__setitem__("kv_mode", "int8")),
        ("debug profile in identities", "C1", lambda d: d["identities"].__setitem__("profile", "debug")),
        ("wrong GGUF sha256", "C1", lambda d: d["identities"].__setitem__("gguf_sha256", "1" * 64)),
        ("identity missing (empty identities)", "C1", lambda d: d.__setitem__("identities", {})),
        ("eager comparison file claims graph", "E1", lambda d: d["identities"].__setitem__("capture_requested", "graph")),
        ("captured=true in the eager file", "E1", lambda d: run0(d).__setitem__("captured", True)),
        ("captured flag missing", "C1", lambda d: run0(d).pop("captured")),
    ]
    for label, name, fn in cases:
        check(f"validate refuses: {label}", bad(name, fn))
    # a tiered run (it carries the `tiered` member) also waits on the host in every layer
    tiered = fx.clone(docs["A1"])
    run0(tiered)["tiered"] = {"bank_hits": 1}
    run0(tiered)["syncs"] = ev.N_STEPS * 121
    run0(tiered)["host_writes"] = ev.N_STEPS * 81
    check("validate: a tiered run with its in-layer waits is accepted", not raises(ev.Stop, rw.validate_raw, "A1", tiered, ref))
    run0(tiered)["syncs"] = ev.N_STEPS - 1
    check("validate refuses: a tiered run with fewer syncs than tokens", raises(ev.Stop, rw.validate_raw, "A1", tiered, ref))
    run0(tiered)["syncs"] = ev.N_STEPS * 121
    run0(tiered)["host_writes"] = ev.N_STEPS - 1
    check("validate refuses: a tiered run with fewer host writes than tokens", raises(ev.Stop, rw.validate_raw, "A1", tiered, ref))
    run0(tiered)["host_writes"] = ev.N_STEPS * 81
    run0(tiered)["tiered"] = "yes"
    check("validate refuses: a `tiered` key that is not a dict does not select the lower bound (exact syncs apply)",
          raises(ev.Stop, rw.validate_raw, "A1", tiered, ref))
    check("speed: counts_ok is exact on a plain run, a lower bound on a tiered one",
          ev.counts_ok({"syncs": ev.N_STEPS}, "syncs") and not ev.counts_ok({"syncs": ev.N_STEPS + 1}, "syncs")
          and ev.counts_ok({"syncs": ev.N_STEPS + 1, "tiered": {}}, "syncs") and not ev.counts_ok({"syncs": None, "tiered": {}}, "syncs")
          and not ev.counts_ok({"syncs": ev.N_STEPS + 1, "tiered": True}, "syncs"))
    check("validate: 500 MB free after the load -> REFUSED (admission, not STOP)", bad("C1", lambda d: d["identities"].__setitem__("mem_free_after_load", 500 * 2**20), ev.Refused))
    check("validate: a file with refused set -> REFUSED", bad("C1", lambda d: d.__setitem__("refused", {"kind": "memory"}), ev.Refused))
    # cross-file checks
    check("identities: consistent set accepted", rw.check_identities(docs)["exe_sha256"] == fx.EXE)
    d2 = fx.clone(docs); d2["C2"]["identities"]["exe_sha256"] = "f" * 64
    check("identities: mixed binaries refused", raises(ev.Stop, rw.check_identities, d2))
    d2 = fx.clone(docs); d2["B3"]["identities"]["engine_commit"] = "old"
    check("identities: stale file from another commit refused", raises(ev.Stop, rw.check_identities, d2))
    d2 = fx.clone(docs); d2["C2"] = fx.clone(docs["C1"])
    check("identities: C2 a copy of C1 refused (same pid and start)", raises(ev.Stop, rw.check_identities, d2))
    d2 = fx.clone(docs); d2["C2"]["identities"]["pid"] = 9; d2["C2"]["identities"]["started_unix_ns"] = 9
    d2["C2"]["prompts"][0]["runs"][0]["decode_ns"] = d2["C1"]["prompts"][0]["runs"][0]["decode_ns"]
    check("identities: a run copied with new pid refused (equal decode_ns)", raises(ev.Stop, rw.check_identities, d2))
    # capture state
    check("capture: graph data, all captured -> not a pre-run", rw.capture_state(docs, False) == (False, []))
    d2 = fx.clone(docs); d2["C3"]["prompts"][0]["runs"][0]["captured"] = False
    pre, why = rw.capture_state(d2, False)
    check("capture: graph requested but a run ran uncaptured -> FAIL reason (silent fallback)", not pre and len(why) == 1)
    ed = fx.make_all(ref, capture="eager")
    check("capture: all eager without --prerun -> refused", raises(ev.Stop, rw.capture_state, ed, False))
    check("capture: all eager with --prerun -> pre-run", rw.capture_state(ed, True) == (True, []))
    check("capture: --prerun on graph data refused", raises(ev.Stop, rw.capture_state, docs, True))
    d2 = fx.clone(docs); d2["B1"]["identities"]["capture_requested"] = "eager"
    check("capture: gate files disagree on the requested mode -> refused", raises(ev.Stop, rw.capture_state, d2, False))
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
