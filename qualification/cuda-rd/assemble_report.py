#!/usr/bin/env python3
"""RD-10: read the raw files of one stage-1 run, apply a5_eval, write qualification/cuda-rd/stage1-<date>.md.

Usage: assemble_report.py --head FULL_SHA --raw DIR --release-check JSON --wrapper-log LOG --selftest-result JSON --date D --out FILE [--prerun]
       assemble_report.py --check-raw FILE   (CPU check of one raw file right after its GPU call; STOP if it fails or the
       measured rate is below 3.6 tok/s = one fifth of the gate)
Exit: 0 PASS, 1 FAIL, 2 STOP (typed, no verdict; every exception ends here), 3 REFUSED (admission),
      4 PRE-RUN complete (an eager pre-run whose tables pass; never a stage gate),
      5 VOIDED (a clock rule voided a run: repeat the call; never a FAIL), 7 BLOCKED (three consecutive voided runs).
See a5_eval.py. The card samples of each call (NAME.watch.jsonl, one every 0.5 s) are judged against the timed region of
each run only: any sample below 1395 MHz voids it, a busy (>= 50 percent) median below 1600 MHz voids it.
The pre-run label comes from the DATA (capture_requested and the per-run captured flags); --prerun must agree with it.
"""
import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a5_eval as ev  # noqa: E402
import clockcheck as ck  # noqa: E402
import rawcheck as rw  # noqa: E402
import wrapperlog as wl  # noqa: E402
from report_text import LABEL, hold_records, render  # noqa: E402

GATE_STAGE = "1"


def script_hashes():
    """Every verdict-relevant script, hashed and compared with the committed MANIFEST-SCRIPTS.txt."""
    out = {}
    for ln in (HERE / "MANIFEST-SCRIPTS.txt").read_text().splitlines():
        want, name = ln.split(None, 1)
        have = ev.sha_file(HERE / name.strip())
        if have != want:
            raise ev.Stop(f"{name.strip()} differs from MANIFEST-SCRIPTS.txt ({have} != {want})")
        out[name.strip()] = have
    return out


def load_raw(raw):
    docs, problems, partial = {}, [], {}
    blocked = sorted(p.name for p in raw.glob("*.blocked"))
    if blocked:
        raise ev.Blocked(f"{len(blocked)} call(s) blocked after {ck.MAX_VOIDS} consecutive voided runs: {blocked}")
    for name in ev.FILES:                       # a refused call is reported as REFUSED even when the other calls never ran (M3)
        refused = raw / f"{name}.json"
        if refused.exists() and ev.read_json(refused).get("refused"):
            raise ev.Refused(f"{name}: refused {ev.read_json(refused)['refused']}")
    for name in ev.FILES:
        if (raw / f"{name}.killed").exists():
            problems.append(f"too slow or hung: call {name} was killed at the hold limit (no raw file)")
            continue
        if (raw / f"{name}.skipped").exists():
            problems.append(f"not run: the run stopped early after a too-slow call (call {name} skipped)")
            continue
        p = raw / f"{name}.json"
        if not p.exists():
            raise ev.Stop(f"missing raw file {name}.json")
        doc = ev.read_json(p)
        if doc.get("incomplete"):
            partial[name] = doc.get("partial", {})
            problems.append(f"too slow: call {name} hit its deadline after {doc.get('elapsed_s')} s ({doc.get('partial')})")
        docs[name] = doc
    return docs, problems, partial


def merged(docs, names):
    """prompt id -> entry, over the given files."""
    out = {}
    for n in names:
        for p in docs[n]["prompts"]:
            out[p["id"]] = p
    return out


def evaluate(docs, ref):
    res = {"a5": {}, "free": {}, "det": {}, "tf_free": {}, "cap_eq": {}}
    forced, eager = merged(docs, ("A1", "A2")), merged(docs, ("E1", "E2"))
    free = {p: [] for p in ev.PROMPTS}
    for n in sorted(k for k in ev.FILES if k[0] in "BC"):
        for p in docs[n]["prompts"]:
            free[p["id"]].extend(p["runs"])
    for pid in ev.PROMPTS:
        res["a5"][pid] = ev.eval_a5(forced[pid]["runs"][0], forced[pid]["ids_sha256"], ref, pid)
        res["free"][pid] = ev.eval_free(free[pid], ref, pid)
        need = 5 if pid == "P1" else 3
        res["det"][pid] = ev.eval_determinism(free[pid], need)
        res["tf_free"][pid] = ev.eval_tf_vs_free(free[pid][0], forced[pid]["runs"][0], res["free"][pid][0]["first_divergence"])
        res["cap_eq"][pid] = [s["hash"] for s in forced[pid]["runs"][0]["steps"]] == \
                             [s["hash"] for s in eager[pid]["runs"][0]["steps"]]
    res["speed"] = ev.eval_speed(free["P1"], GATE_STAGE)
    res["speed_others"] = {p: ev.eval_speed(free[p], GATE_STAGE)["median"] for p in ev.PROMPTS[1:]}
    return res


def release_problems(rel, exe_sha):
    bi = rel.get("build_info")
    bad = list(rel.get("misses", [])) if isinstance(rel.get("misses"), list) else ["misses missing"]
    if not isinstance(bi, dict) or bi.get("profile") != "release" or bi.get("opt_level") != "3" or bi.get("debug") is not False \
            or bi.get("debug_assertions") is not False:
        bad.append("build_info is not {profile: release, opt_level: 3, debug: false, debug_assertions: false}")
    if not re.match(r"^Cuda compilation tools, release \d+\.\d+", str(rel.get("cuobjdump_version", ""))):
        bad.append("cuobjdump version not recorded (empty or malformed)")
    if rel.get("exe_sha256") != exe_sha:
        bad.append("the executable that ran is not the one the release check examined")
    return bad


def verdict(res, why_pre, release, exe_sha, wrapper_bad, selftest_why, warm_why=()):
    why = list(why_pre) + list(selftest_why)
    if res is not None:
        why += [f"A5 {p}: {len(r['fails'])} gated steps fail" for p, r in res["a5"].items() if not r["ok"]]
        why += [f"determinism {p}" for p, r in res["det"].items() if not r["ok"]]
        why += [f"free/forced row hashes differ {p}" for p, r in res["tf_free"].items() if not r["ok"]]
        why += [f"captured != uncaptured {p}" for p, ok in res["cap_eq"].items() if not ok]
        sp = res["speed"]
        if not sp["pass"]:
            why.append(f"speed {sp['median']:.2f} tok/s < {sp['gate']:.0f}")
        if not sp["breakdown_ok"]:
            why.append(f"ms/token breakdown off by {sp['breakdown_err'] * 100:.1f} percent")
        if not sp["syncs_ok"]:
            why.append("a P1 run has syncs != 128 (>= 128 for a tiered run)")
        if not sp["host_writes_ok"]:
            why.append("a P1 run has host_writes != 128 (>= 128 for a tiered run)")
    why += [f"release-build check: {m}" for m in release_problems(release, exe_sha)]
    why += warm_why
    why += [f"wrapper test (RD-2e): {b}" for b in wrapper_bad]
    return why


def selftest_problems(path, hashes):
    r = ev.read_json(path)
    why = []
    if r.get("rc") != 0:
        why.append(f"evaluator self-test failed (rc {r.get('rc')})")
    if r.get("scripts") != hashes:
        why.append("the self-test result belongs to different script versions than the ones evaluating")
    return why


def run(a):
    ref = ev.Reference()
    hashes = script_hashes()
    raw = Path(a.raw)
    docs, why_pre, partial = load_raw(raw)
    release = ev.read_json(a.release_check)
    selftest_why = selftest_problems(a.selftest_result, hashes)
    for n, d in docs.items():
        rw.validate_raw(n, d, ref)
    ident = rw.check_identities(docs)
    rw.check_head(ident, a.head)
    wb = []
    if a.wrapper_log and Path(a.wrapper_log).exists():
        shapes = ev.read_json(HERE / "rd2e_shapes.json")
        wtext = Path(a.wrapper_log).read_text()
        wb = wl.parse_wrapper_log(wtext, shapes, ident["engine_commit"])
        hashes["data wrapper-log"] = ev.sha_file(a.wrapper_log)
    else:
        wb = ["no wrapper log (row 2 open)"]
    for n in sorted(docs):
        hashes[f"data {n}.json"] = ev.sha_file(raw / f"{n}.json")
    hashes["gated set a5-gated-steps.json"] = ref.gated_sha256
    for rel, h in sorted(ref.verified.items()):
        hashes[f"input {rel}"] = h
    hashes["binary cuda-rd-run"] = ident["exe_sha256"]
    hashes["release-check json"] = ev.sha_file(a.release_check)
    clocks = {}
    for n in sorted(docs):
        if not docs[n].get("incomplete"):
            clocks.update(ck.clock_stats(n, docs[n], raw / f"{n}.watch.jsonl"))
            hashes[f"data {n}.watch.jsonl"] = ev.sha_file(raw / f"{n}.watch.jsonl")
    voided = [f"{k[0]} {k[1]}: {'; '.join(v['why'])}" for k, v in sorted(clocks.items()) if v["voided"]]
    if voided:
        raise ev.Voided("; ".join(voided))
    incomplete = bool(why_pre)
    prerun = False
    res = None
    if not incomplete:
        prerun, cap_why = rw.capture_state(docs, a.prerun)
        why_pre += cap_why
        res = evaluate(docs, ref)
    elif a.prerun:
        raise ev.Stop("--prerun with incomplete data")
    why = verdict(res, why_pre, release, ident["exe_sha256"], wb, selftest_why, ck.warmup_problems(docs))
    note = f"\nPartial figures: {json.dumps(partial, sort_keys=True)}\n" if partial else ""
    Path(a.out).write_text(render(res, ident, hashes, why, a.date, prerun, ref, note, hold_records(a.logs), clocks))
    if why:
        print((LABEL + ": " if prerun else "") + "FAIL: " + "; ".join(why))
        return 1
    print("PRE-RUN complete (not a stage gate)" if prerun else "PASS")
    return 4 if prerun else 0


def check_raw(path):
    """Right after a GPU call: schema, identities, per-step checks, A5 for forced files, rate guard."""
    ref = ev.Reference()
    path = Path(path)
    name = path.stem
    if name not in ev.FILES:
        raise ev.Stop(f"{path.name} is not one of {sorted(ev.FILES)}")
    doc = ev.read_json(path)
    rw.validate_raw(name, doc, ref)
    watch = path.with_name(f"{name}.watch.jsonl")
    if not doc.get("incomplete"):
        voided = {k: v for k, v in ck.clock_stats(name, doc, watch).items() if v["voided"]}
        if voided:
            raise ev.Voided("; ".join(f"{k[1]}: {'; '.join(v['why'])}" for k, v in sorted(voided.items())))
    if doc.get("incomplete"):
        print(f"INCOMPLETE: {name} hit its deadline after {doc.get('elapsed_s')} s: {doc.get('partial')} (too slow: FAIL path)")
        return 1
    rates = [ev.N_STEPS / (r["decode_ns"] / 1e9) for p in doc["prompts"] for r in p["runs"]]
    if min(rates) < 3.6:
        raise ev.Stop(f"{name}: rate {min(rates):.2f} tok/s is below 3.6 (a fifth of the gate): debug build or broken lock?")
    if doc["mode"] == "forced":
        for p in doc["prompts"]:
            r = ev.eval_a5(p["runs"][0], p["ids_sha256"], ref, p["id"])
            print(p["id"], "A5", "ok" if r["ok"] else f"fails {len(r['fails'])} gated steps", f"max dev {r['max_logit_dev']:.3f}")
    print(f"{name}: valid, rates {', '.join(f'{x:.2f}' for x in rates)} tok/s")
    return 0


def write_closed(out, text, label, error, code):
    """Writes the REFUSED or BLOCKED record; an unwritable output path is a typed STOP, never a traceback or a FAIL (lane L's N5)."""
    try:
        Path(out).write_text(text)
    except OSError as e:
        print(f"STOP: cannot write the {label} record {out}: {e.strerror}; the condition was: {error}")
        return 2
    print(f"{label}:", error)
    return code


def main(argv):
    if len(argv) == 3 and argv[1] == "--check-raw":
        try:
            return check_raw(argv[2])
        except ev.Stop as e:
            print("STOP:", e)
            return 2
        except ev.Refused as e:
            print("REFUSED:", e)
            return 3
        except ev.Voided as e:
            print("VOIDED:", e, "(environmental: repeat the call; never a FAIL)")
            return 5
        except Exception as e:  # noqa: BLE001 - every unexpected input problem is a typed STOP, never a traceback
            print(f"STOP: internal error {type(e).__name__}: {e}")
            return 2
    ap = argparse.ArgumentParser()
    for k in ("--raw", "--release-check", "--selftest-result", "--out", "--date"):
        ap.add_argument(k, required=True)
    ap.add_argument("--wrapper-log", default="")
    ap.add_argument("--logs", default="")
    ap.add_argument("--head", required=True)
    ap.add_argument("--prerun", action="store_true")
    a = ap.parse_args(argv[1:])
    try:
        return run(a)
    except ev.Stop as e:
        print("STOP:", e)
        return 2
    except ev.Voided as e:
        print("VOIDED:", e, "(environmental: repeat the calls; never a FAIL)")
        return 5
    except ev.Blocked as e:
        return write_closed(a.out, f"# CUDA-RD stage 1 {a.date}: BLOCKED (a clock or thermal problem; not a FAIL, not a pass)\n\n{e}\n\nThe card samples (`*.watch.jsonl`) are attached in the raw directory.\n", "BLOCKED", e, 7)
    except ev.Refused as e:
        return write_closed(a.out, f"# CUDA-RD stage 1 {a.date}: REFUSED (not a FAIL)\n\n{e}\n", "REFUSED", e, 3)
    except Exception as e:  # noqa: BLE001 - malformed input must end as a typed STOP, not as a traceback or a FAIL
        print(f"STOP: internal error {type(e).__name__}: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
