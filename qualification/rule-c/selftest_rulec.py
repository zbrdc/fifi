#!/usr/bin/env python3
"""Self-test of the rule C grading tools (qualification/rule-c/): a5_retol.py, grade.py, determ.py.
Stdlib tests on small synthetic raw/oracle fixtures written to a temp dir; exit 0 only if every test passes.
The tools under test are the ones next to this script, unless RULEC_TOOLS_DIR names another copy (used to prove each
test fails when its rule is broken: point it at a one-line mutant).  python3 -I -B, stdlib only, no GPU.
"""
import contextlib
import copy
import hashlib
import shutil
import importlib.util
import io
import json
import math
import os
import sys
import tempfile
import traceback
from pathlib import Path

TOOLS = Path(os.environ.get("RULEC_TOOLS_DIR") or Path(__file__).resolve().parent)


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


grade = load_module("grade")  # grade first: alone it must wire its own ../cuda-rd import path
retol = load_module("a5_retol")
determ = load_module("determ")
rawcheck = load_module("rawcheck_rulec")


def write_json(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))


def cpu_step(margin, tok=100):
    """A cpu12 reference step: top-1 `tok` at logit 10.0, runner-up at 10.0 - margin."""
    return {"token": tok, "top": [[tok, 10.0], [200, 10.0 - margin]]}


def dev_step(d, tok=100):
    """An agreeing independent-source step: `tok` still top-1, its logit off by d."""
    return {"token": tok, "top": [[tok, 10.0 + d], [200, 9.0]]}


def dev_step_flip(d, tok=100):
    """A disagreeing source step: another token is top-1, `tok` is runner-up, off by d."""
    return {"token": 999, "top": [[999, 12.0], [tok, 10.0 + d]]}


def write_sources(root, p1_steps):
    """Write the four source run.json files of prompt P1: (cpu12, cpu6, cudaOld, cudaNew) steps."""
    cpu12, cpu6, cuda_old, cuda_new = p1_steps
    write_json(root / "oracle" / "cpu/clean/P1/run.json", {"steps": cpu12})
    write_json(root / "oracle" / "cpu/threads6/P1/run.json", {"steps": cpu6})
    write_json(root / "oracle" / "cuda/follow/P1/run.json", {"steps": cuda_old})
    write_json(root / "lcpp-new" / "P1/run.json", {"steps": cuda_new})


def build_rows(root):
    src = retol.make_sources(root / "oracle", root / "lcpp-new")
    return retol.build("P1", src), src


TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


# ---- grade: nearest-rank percentile on known lists, including the empty list and a single value
@test
def grade_percentile_known_lists(td):
    assert math.isnan(grade.percentile([], 0.95)), "empty list must be nan"
    assert grade.percentile([7.0], 0.99) == 7.0, "single value must be that value"
    hundred = list(range(1, 101))
    assert grade.percentile(hundred, 0.95) == 95, "nearest rank of 100 values at q=0.95 is the 95th"
    assert grade.percentile([10, 20, 30, 40], 0.25) == 10, "rank 1.4999999 rounds to 1"
    assert grade.percentile([10, 20, 30, 40, 50, 60], 0.75) == 50, "rank 4.5 must round UP to 5, not to even 4"


# ---- grade: the S3c bounds verdict on each side of every one of the three bounds
@test
def grade_bounds_verdict_each_side(td):
    b = grade.BOUNDS
    assert b == (0.165, 0.651, 1.131), "frozen S3c bounds"
    assert grade.bounds_verdict(0.100, 0.500, 1.000) and grade.bounds_verdict(b[0], b[1], b[2])
    assert grade.bounds_verdict(b[0], 0.500, 1.000) and not grade.bounds_verdict(0.166, 0.500, 1.000), "median edge"
    assert grade.bounds_verdict(0.100, b[1], 1.000) and not grade.bounds_verdict(0.100, 0.652, 1.000), "p95 edge"
    assert grade.bounds_verdict(0.100, 0.500, b[2]) and not grade.bounds_verdict(0.100, 0.500, 1.132), "p99 edge"


# ---- grade: mismatched steps are excluded from n and counted as mismatches
class FixRef:
    """Reference stand-in: only the streams; block() refuses so grade_file's A5 arm stays inert."""

    def __init__(self, streams):
        self.streams = streams

    def block(self, pid):
        raise grade.ev.Stop("fixture: no gated set")


@test
def grade_mismatched_steps_excluded(td):
    ref = FixRef({"P1": [
        {"token": 5, "top": [[5, 2.0], [6, 1.0]]},
        {"token": 5, "top": [[5, 2.0], [6, 1.0]]},
        {"token": 7, "top": [[7, 3.0], [8, 2.0]]},
    ]})
    doc = {"prompts": [{"id": "P1", "ids_sha256": "x", "runs": [{
        "decode_ns": 2_000_000_000, "phase_ns": {"replay": 30_000_000},
        "steps": [{"argmax": 5, "argmax_logit": 2.5},    # agrees -> dev 0.5
                  {"argmax": 6, "argmax_logit": 9.9},    # token mismatch -> no dev
                  {"argmax": 7, "argmax_logit": 2.7}]}]}]}  # agrees -> dev 0.3
    path = td / "R.json"
    write_json(path, doc)
    _, devs, miss, a5, rates, replays, counters = grade.grade_file(path, doc, ref)
    assert devs == [abs(2.5 - 2.0), abs(2.7 - 3.0)] and miss == [("P1", 1)] and len(devs) + len(miss) == 3, f"{devs} {miss}"
    assert rates == [1.5] and replays == [10.0] and a5[0][1]["ok"] is None, "decode/replay rates; A5 stays information-only"


# ---- retol: S_t = max(floor, max source deviation), with the floor case
@test
def retol_spread_floor_and_max(td):
    row = {"dev": {"cpu6": 0.2, "cudaOld": 0.07, "cudaNew": 0.06}}
    assert retol.spread(row, retol.ALL) == 0.2, "S_t is the largest source deviation"
    floor_row = {"dev": {"cpu6": 0.001, "cudaOld": 0.002, "cudaNew": 0.003}}
    assert retol.spread(floor_row, retol.ALL) == retol.FLOOR == 0.05, "the floor applies when every deviation is below it"
    missing = {"dev": {"cpu6": None, "cudaOld": 0.2, "cudaNew": 0.1}}
    assert retol.spread(missing, retol.ALL) is None, "an undefined deviation leaves S_t undefined"


# ---- retol: the L set (all sources agree and margin > 2 S_t) and the T set (margin > S_t), each side of its edge
@test
def retol_L_and_T_edges(td):
    write_sources(td, (
        [cpu_step(1.0), cpu_step(1.001), cpu_step(0.5), cpu_step(2.0)],   # cpu12 margins
        [dev_step(0.5), dev_step(0.5), dev_step(0.5), dev_step_flip(0.4)],  # cpu6 (step 3 disagrees)
        [dev_step(0.1)] * 4,                                               # cudaOld
        [dev_step(0.1)] * 4))                                              # cudaNew
    rows, _ = build_rows(td)
    assert [retol.spread(r, retol.ALL) for r in rows] == [0.5, 0.5, 0.5, abs(10.4 - 10.0)]
    T, L = retol.classes(rows, retol.ALL)
    assert [r["t"] for r, _ in T] == [0, 1, 3], f"T needs margin > S_t: step 2 (margin == S_t) stays out: {T}"
    assert [r["t"] for r, _ in L] == [1], f"L needs agree and margin > 2 S_t strictly: {[r['t'] for r, _ in L]}"
    assert rows[3]["agree"] is False, "a disagreeing source breaks agreement even when its deviation is defined"


# ---- retol: leave-one-out drops exactly one source from the S_t derivation
@test
def retol_leave_one_out_drops_one(td):
    write_sources(td, ([cpu_step(100.0)], [dev_step(1.0)], [dev_step(3.0)], [dev_step(0.3)]))
    rows, src = build_rows(td)
    # held-out graded against the other two only: S_t is 1.0 for cudaOld (its dev 3.0 fails at tol 2.0), 3.0 for
    # cpu6/cudaNew (devs 1.0/0.3 pass at tol 6.0); values are (nT, nL, token fails, logit fails).
    got = {h: retol.grade_source("P1", rows, h, rest, src) for h, rest in
           (("cudaOld", ("cpu6", "cudaNew")), ("cpu6", ("cudaOld", "cudaNew")), ("cudaNew", ("cpu6", "cudaOld")))}
    assert got == {"cudaOld": (1, 1, 0, 1), "cpu6": (1, 1, 0, 0), "cudaNew": (1, 1, 0, 0)}, got


# ---- retol (rule C token gate): threshold = 1.25 x the largest flip margin; a mismatch above it fails, below passes
@test
def retol_token_gate_0p511(td):
    flips = {"cpu6": [0.020, 0.4088], "cudaOld": [0.100], "cudaNew": [0.200]}
    mstar = retol.flip_threshold(flips)
    assert f"{mstar:.3f}" == "0.511", f"1.25 x 0.4088 = 0.511, got {mstar}"
    assert retol.flip_threshold({"cpu6": [], "cudaOld": [], "cudaNew": []}) == 0.0, "no flips anywhere -> threshold 0"
    data = {"P1": [{"t": 0, "tok": 100, "margin": 0.512}, {"t": 1, "tok": 100, "margin": 0.510}]}
    nT, tf = retol.token_gate({"P1": [{"argmax": 999}, {"argmax": 999}]}, data, mstar)
    assert (nT, tf) == (1, 1), f"only the step with margin > 0.511 is gated; its mismatch fails: {(nT, tf)}"
    nT, tf = retol.token_gate({"P1": [{"argmax": 100}, {"argmax": 100}]}, data, mstar)
    assert (nT, tf) == (1, 0), "agreement on the gated step passes"
    loo = retol.flip_threshold({"cudaOld": flips["cudaOld"], "cudaNew": flips["cudaNew"]})
    assert f"{loo:.3f}" == "0.250", "the leave-one-out threshold drops the largest flip"


# ---- determ: equal files pass; a single changed hash, argmax or in_token fails; refused input; length mismatch
def raw_doc(steps, mode="free", incomplete=False, refused=None):
    return {"schema": "cuda-rd-raw/2", "mode": mode, "incomplete": incomplete, "refused": refused,
            "prompts": [{"id": "P1", "runs": [{"steps": steps}]}]}


def dstep(hash_="h1", argmax=5, in_token=9):
    return {"hash": hash_, "argmax": argmax, "in_token": in_token}


def run_determ(td, a, b):
    pa, pb = td / "A.json", td / "B.json"
    write_json(pa, a)
    write_json(pb, b)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = determ.main(pa, pb)
    return rc, buf.getvalue()


@test
def determ_equal_files_pass(td):
    steps = [dstep("h1"), dstep("h2", argmax=6, in_token=10)]
    rc, out = run_determ(td, raw_doc(steps), raw_doc(steps))
    assert rc == 0 and "DETERMINISTIC" in out and "2/2" in out, f"identical files: rc={rc} out={out!r}"


@test
def determ_hash_change_fails(td):
    rc, out = run_determ(td, raw_doc([dstep("h1"), dstep("h2")]), raw_doc([dstep("hX"), dstep("h2")]))
    assert rc == 1 and "NOT IDENTICAL" in out and "1/2" in out, f"one changed hash: rc={rc} out={out!r}"


@test
def determ_argmax_change_fails(td):
    rc, out = run_determ(td, raw_doc([dstep("h1"), dstep("h2", argmax=6)]),
                         raw_doc([dstep("h1"), dstep("h2", argmax=7)]))
    assert rc == 1 and "NOT IDENTICAL" in out and "1/2" in out, f"one changed argmax: rc={rc} out={out!r}"


@test
def determ_in_token_change_fails(td):
    rc, out = run_determ(td, raw_doc([dstep("h1", in_token=9), dstep("h2")]),
                         raw_doc([dstep("h1", in_token=11), dstep("h2")]))
    assert rc == 1 and "NOT IDENTICAL" in out and "1/2" in out, f"one changed in_token: rc={rc} out={out!r}"


@test
def determ_incomplete_or_refused_refused(td):
    steps = [dstep()]
    pa, pb = td / "A.json", td / "B.json"
    write_json(pb, raw_doc(steps))
    for side, path in (("A", pa), ("B", pb)):
        for bad in (raw_doc(steps, incomplete=True), raw_doc(steps, refused={"kind": "memory"})):
            write_json(pa if side == "A" else pb, bad)
            try:
                determ.main(pa, pb)
            except determ.rc.Refused as e:
                assert ("incomplete" in str(e) or "refused" in str(e)) and path.name in str(e), \
                    f"the refusal must name {side} and the reason: {e}"
            else:
                raise AssertionError("an incomplete or refused raw must be refused, never graded")
            write_json(pa, raw_doc(steps))  # restore A for the next round
            write_json(pb, raw_doc(steps))


@test
def determ_length_mismatch_fails(td):
    steps = [dstep("h1"), dstep("h2")]
    rc, out = run_determ(td, raw_doc(steps), raw_doc(steps + [dstep("h3")]))
    assert rc == 1 and "NOT IDENTICAL" in out, f"a longer second file must fail even though the shared prefix matches: rc={rc}"


# ---- the two percentile definitions (lane C ruling 2026-10-14), pinned on a list where they differ
@test
def percentile_definitions_pinned_where_they_differ(td):
    hundred = list(range(1, 101))
    assert grade.percentile(sorted(hundred), 0.95) == 95, "grade's nearest rank of 100 values at q=0.95 is the 95th"
    assert retol.q(hundred, 0.95) == 96, "a5_retol's q picks index int(0.95*100) = 95, i.e. the 96th value"
    assert grade.percentile(sorted(hundred), 0.99) == 99 and retol.q(hundred, 0.99) == 100, \
        "0.99*100 == 99.0 exactly, so q takes the 100th while the nearest rank takes the 99th"
    assert math.isnan(grade.percentile([], 0.95)), "grade's empty list is nan (retol's q has no empty-list contract)"


# ---- the shared raw gate (rawcheck_rulec.check_raw): every refusal, and what it does NOT check
def valid_doc(name_fields=True):
    """A acceptable cuda-rd-raw/2 document: run name A1, prompts P1+P3, one 3-step run each, finite floats."""
    step = {"argmax": 5, "argmax_logit": 2.5, "lse": 1.25, "top32": [[5, 2.5], [6, 1.0]],
            "hash": "h", "in_token": 9, "forced_logit": 2.5}
    return {"schema": "cuda-rd-raw/2", "mode": "forced", "incomplete": False, "refused": None,
            "prompts": [{"id": "P1", "runs": [{"steps": [dict(step) for _ in range(3)]}]},
                        {"id": "P3", "runs": [{"steps": [dict(step) for _ in range(3)]}]}]}


def refused_reason(doc, name, **kw):
    try:
        rawcheck.check_raw(doc, name, **kw)
    except rawcheck.Refused as e:
        return str(e)
    raise AssertionError("expected the raw gate to refuse")


@test
def rawcheck_refuses_bad_schema_flags_and_duplicate_ids(td):
    assert rawcheck.check_raw(valid_doc(), "A1.json", n_steps=3) is None, "the clean document passes silently"
    doc = valid_doc(); doc["schema"] = "cuda-rd-raw/1"
    assert "cuda-rd-raw/1" in refused_reason(doc, "A1.json", n_steps=3), "names the found schema"
    doc = valid_doc(); doc["incomplete"] = True
    assert "incomplete" in refused_reason(doc, "A1.json", n_steps=3)
    doc = valid_doc(); doc["refused"] = {"kind": "memory"}
    assert "refused" in refused_reason(doc, "A1.json", n_steps=3)
    doc = valid_doc(); doc["prompts"][1]["id"] = "P1"
    assert "duplicate" in refused_reason(doc, "A1.json", n_steps=3) and "P1" in refused_reason(doc, "A1.json", n_steps=3)


@test
def rawcheck_refuses_no_runs_bad_mode_and_malformed_containers(td):
    doc = valid_doc(); doc["prompts"][0]["runs"] = []
    assert "no runs" in refused_reason(doc, "A1.json", n_steps=3), "runs: [] is a coverage bypass (MAJOR-1)"
    doc = valid_doc(); doc["mode"] = "free"
    assert "mode 'free' is not 'forced'" in refused_reason(doc, "A1.json", n_steps=3, mode="forced")
    assert rawcheck.check_raw(doc, "D1.json") is None, "determ compares modes instead of requiring one"
    doc = valid_doc(); doc["prompts"] = "nope"
    assert "no prompts list" in refused_reason(doc, "A1.json", n_steps=3)
    doc = valid_doc(); doc["prompts"][0] = "nope"
    assert "not an object" in refused_reason(doc, "A1.json", n_steps=3)
    doc = valid_doc(); del doc["prompts"][0]["id"]
    assert "without an id" in refused_reason(doc, "A1.json", n_steps=3)
    doc = valid_doc(); doc["prompts"][0]["runs"] = "nope"
    assert "no runs list" in refused_reason(doc, "A1.json", n_steps=3)
    doc = valid_doc(); doc["prompts"][0]["runs"][0] = "nope"
    assert "no steps list" in refused_reason(doc, "A1.json", n_steps=3)
    doc = valid_doc(); doc["prompts"][0]["runs"][0]["steps"] = "nope"
    assert "no steps list" in refused_reason(doc, "A1.json", n_steps=3)
    doc = valid_doc(); doc["prompts"][0]["runs"][0]["steps"][1]["top32"] = [[5, 2.5, 9.9]]
    assert "not a finite" in refused_reason(doc, "A1.json", fields=("top32",)), "pair length checked"
    doc = valid_doc(); doc["prompts"][0]["runs"][0]["steps"][1]["argmax_logit"] = True
    assert "not a finite" in refused_reason(doc, "A1.json", fields=("argmax_logit",)), "a bool is not a number"


@test
def check_coverage_refuses_dup_unknown_and_partial(td):
    oracle = ["P1", "P2", "P3", "P4", "P5"]
    assert rawcheck.check_coverage([("A1.json", ["P1", "P3"]), ("A2.json", ["P2", "P4", "P5"])], oracle) is False
    try:
        rawcheck.check_coverage([("A1.json", ["P1", "P3", "P2"]), ("A2.json", ["P2", "P4", "P5"])], oracle)
    except rawcheck.Refused as e:
        assert "P2 also comes from" in str(e), str(e)
    else:
        raise AssertionError("a prompt id shared across files must be refused")
    try:
        rawcheck.check_coverage([("A1.json", ["P1", "P3", "P9"]), ("A2.json", ["P2", "P4", "P5"])], oracle)
    except rawcheck.Refused as e:
        assert "P9" in str(e) and "gated set" in str(e), str(e)
    else:
        raise AssertionError("an unknown prompt id must be refused")
    try:
        rawcheck.check_coverage([("NAT-A1.json", ["P1"]), ("A2.json", ["P2", "P4", "P5"])], oracle)
    except rawcheck.Refused as e:
        assert "not covered" in str(e) and "P3" in str(e), str(e)
    else:
        raise AssertionError("a missing prompt must be refused")
    assert rawcheck.check_coverage([("NAT-A1.json", ["P1"]), ("A2.json", ["P2", "P4", "P5"])],
                                   oracle, allow_partial=True) is True, "the explicit partial opt-in"


@test
def check_steps_refuses_nonfinite_and_short_reference_trees(td):
    steps = [{"top": [[5, 2.5], [6, 1.0]]} for _ in range(3)]
    assert rawcheck.check_steps(steps, "cpu6 P1", 3) is None
    bad = copy.deepcopy(steps)
    bad[1]["top"][0][1] = float("nan")
    try:
        rawcheck.check_steps(bad, "cudaNew P2", 3)
    except rawcheck.Refused as e:
        assert "cudaNew P2" in str(e) and "non-finite" in str(e), str(e)
    else:
        raise AssertionError("a NaN top logit in a reference tree must be refused")
    try:
        rawcheck.check_steps(steps[:2], "cudaNew P2", 3)
    except rawcheck.Refused as e:
        assert "2 steps" in str(e) and "3" in str(e), str(e)
    else:
        raise AssertionError("a truncated reference tree must be refused")
    bad = copy.deepcopy(steps)
    bad[1]["top"][0] = [5, 2.5, 9.9]  # a malformed (id, logit) pair
    try:
        rawcheck.check_steps(bad, "cudaNew P2", 3)
    except rawcheck.Refused as e:
        assert "non-finite" in str(e), str(e)
    else:
        raise AssertionError("a malformed top pair must be refused")
    try:
        rawcheck.check_steps("nope", "cudaNew P2", 3)
    except rawcheck.Refused as e:
        assert "no steps list" in str(e), str(e)
    else:
        raise AssertionError("a non-list steps member must be refused")


@test
def rawcheck_refuses_step_counts_not_the_oracles(td):
    doc = valid_doc(); doc["prompts"][0]["runs"][0]["steps"] = doc["prompts"][0]["runs"][0]["steps"][:2]
    reason = refused_reason(doc, "A1.json", n_steps=3)
    assert "2 steps" in reason and "3" in reason, "names both counts; positional steps make a count miss a gap"
    assert rawcheck.check_raw(doc, "A1.json", n_steps={"P1": 2, "P3": 3}) is None, "a per-prompt oracle count works too"
    assert rawcheck.check_raw(doc, "A1.json") is None, "no oracle known (determ) skips the count check"


@test
def rawcheck_refuses_nonfinite_only_in_fields_the_tool_reads(td):
    doc = valid_doc(); doc["prompts"][0]["runs"][0]["steps"][1]["argmax_logit"] = float("nan")
    reason = refused_reason(doc, "A1.json", fields=("argmax_logit",))
    assert "non-finite" in reason or "not a finite" in reason, reason
    assert rawcheck.check_raw(doc, "A1.json", fields=()) is None, "determ reads no floats, so it does not finite-check"
    doc["prompts"][0]["runs"][0]["steps"][1]["lse"] = float("inf")
    doc["prompts"][0]["runs"][0]["steps"][1]["top32"] = [[5, 2.5], [6, float("nan")]]
    reason = refused_reason(doc, "A1.json", fields=("argmax_logit", "lse", "top32"))
    assert "lse" in reason and "top32" in reason, reason
    doc = valid_doc(); doc["prompts"][0]["runs"][0]["steps"][1]["argmax_logit"] = None
    assert "not a finite" in refused_reason(doc, "A1.json", fields=("argmax_logit",)), "null is refused too"

# ---- determ: the structural refusals and comparisons of item 2
def run_determ_full(td, a, b, name_a="A.json", name_b="B.json"):
    """Returns (code, stdout, stderr); code is main's return, or 2 when it raises rc.Refused."""
    pa, pb = td / name_a, td / name_b
    write_json(pa, a)
    if pb != pa:
        write_json(pb, b)
    out, err = io.StringIO(), io.StringIO()
    code = None
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = determ.main(pa, pb)
        except determ.rc.Refused as refused:  # determ's own imported module, not the self-test's copy
            code, _ = 2, err.write(f"refused: {refused}\n")
        except SystemExit as e:
            code = e.code
    return code, out.getvalue(), err.getvalue()


@test
def determ_same_file_twice_refused(td):
    doc = raw_doc([dstep("h1")])
    code, out, err = run_determ_full(td, doc, None, name_b=td / "A.json")
    assert code == 2 and "same file" in err and "A.json" in err, f"{code} {err!r}"
    assert out == "", "a refusal prints nothing on stdout"


@test
def determ_same_file_by_spelling_refused(td):
    (td / "sub").mkdir()
    doc = raw_doc([dstep("h1")])
    code, out, err = run_determ_full(td, doc, None, name_b="sub/../A.json")
    assert code == 2 and "same file" in err, f"{code} {err!r}"


@test
def determ_empty_prompts_runs_or_steps_refused(td):
    doc = raw_doc([dstep("h1")])
    empty_prompts = raw_doc([dstep("h1")]); empty_prompts["prompts"] = []
    code, out, err = run_determ_full(td, empty_prompts, doc)
    assert code == 2 and "no prompts" in err and out == "", f"{code} {err!r}"
    empty_runs = raw_doc([dstep("h1")]); empty_runs["prompts"][0]["runs"] = []
    code, out, err = run_determ_full(td, empty_runs, doc)
    assert code == 2 and "no runs" in err and out == "", f"{code} {err!r}"
    empty_steps = raw_doc([])
    code, out, err = run_determ_full(td, empty_steps, doc)
    assert code == 2 and "empty step list" in err and out == "", f"{code} {err!r}"
    code, out, err = run_determ_full(td, doc, empty_steps)
    assert code == 2 and "B.json" in err and "empty step list" in err, "the B side is checked too"


@test
def determ_unequal_run_counts_refused(td):
    two = raw_doc([dstep("h1")]); two["prompts"][0]["runs"] = two["prompts"][0]["runs"] * 2
    code, out, err = run_determ_full(td, two, raw_doc([dstep("h1")]))
    assert code == 2 and "has 1 runs" in err and "has 2 runs" in err and out == "", f"{code} {err!r}"


@test
def determ_ids_sha256_mismatch_fails(td):
    a, b = raw_doc([dstep("h1")]), raw_doc([dstep("h1")])
    a["prompts"][0]["ids_sha256"] = "x"
    b["prompts"][0]["ids_sha256"] = "y"
    code, out, _ = run_determ_full(td, a, b)
    assert code == 1 and "NOT IDENTICAL" in out, f"{code} {out!r}"
    b["prompts"][0]["ids_sha256"] = "x"
    code, out, _ = run_determ_full(td, a, b)
    assert code == 0 and "DETERMINISTIC" in out, "equal ids_sha256 passes"


@test
def determ_mode_or_prompt_list_mismatch_fails(td):
    a, b = raw_doc([dstep("h1")]), raw_doc([dstep("h1")])
    b["mode"] = "forced"
    code, out, _ = run_determ_full(td, a, b)
    assert code == 1 and "NOT IDENTICAL" in out, f"mode mismatch: {code} {out!r}"
    a, b = raw_doc([dstep("h1")]), raw_doc([dstep("h1")])
    a["prompts"].append({"id": "P9", "runs": a["prompts"][0]["runs"]})
    code, out, _ = run_determ_full(td, a, b)
    assert code == 1 and "NOT IDENTICAL" in out, f"prompt list mismatch: {code} {out!r}"


@test
def determ_a_longer_step_list_fails(td):
    code, out, _ = run_determ_full(td, raw_doc([dstep("h1"), dstep("h2"), dstep("h3")]),
                                   raw_doc([dstep("h1"), dstep("h2")]))
    assert code == 1 and "NOT IDENTICAL" in out and "(other file 2 steps)" in out, \
        f"a longer first file must fail: {code} {out!r}"


@test
def determ_second_run_differs_fails(td):
    """Both runs of every prompt are compared, not just the first pair."""
    run_a = {"steps": [dstep("h1"), dstep("h2")]}
    run_b = {"steps": [dstep("h1"), dstep("hX")]}
    a = raw_doc([dstep("h1")]); a["prompts"][0]["runs"] = [run_a, run_a]
    b = raw_doc([dstep("h1")]); b["prompts"][0]["runs"] = [run_a, run_b]
    code, out, _ = run_determ_full(td, a, b)
    assert code == 1 and "NOT IDENTICAL" in out and "1/2" in out, f"{code} {out!r}"
    b["prompts"][0]["runs"] = [run_a, run_a]
    code, out, _ = run_determ_full(td, a, b)
    assert code == 0 and "DETERMINISTIC" in out and "2/2" in out, f"{code} {out!r}"


@test
def determ_cli_end_to_end(td):
    steps = [dstep(f"h{i}") for i in range(4)]
    pa, pb = td / "D1.json", td / "D2.json"
    write_json(pa, raw_doc(steps))
    write_json(pb, raw_doc(steps))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            determ.cli([str(pa), str(pb)])
        except SystemExit as e:
            code = e.code
    assert code == 0 and "DETERMINISTIC" in buf.getvalue() and "4/4" in buf.getvalue(), f"{code} {buf.getvalue()!r}"
    bad = raw_doc(steps)
    bad["prompts"][0]["runs"][0]["steps"][1] = dstep("hX")
    write_json(pb, bad)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            determ.cli([str(pa), str(pb)])
        except SystemExit as e:
            code = e.code
    assert code == 1 and "NOT IDENTICAL" in buf.getvalue() and "3/4" in buf.getvalue(), f"{code} {buf.getvalue()!r}"
    buf = io.StringIO()
    err = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
        try:
            determ.cli([str(pa), str(td / "missing.json")])
        except SystemExit as e:
            code = e.code
    assert code == 2 and buf.getvalue() == "" and "missing.json" in err.getvalue(), f"{code} {err.getvalue()!r}"
    bad = raw_doc(steps)
    bad["incomplete"] = True
    write_json(pb, bad)
    buf = io.StringIO()
    err = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
        try:
            determ.cli([str(pa), str(pb)])
        except SystemExit as e:
            code = e.code
    assert code == 2 and buf.getvalue() == "" and "incomplete" in err.getvalue(), f"{code} {err.getvalue()!r}"


# ---- end to end: a5_retol.main and grade.cli on one synthetic 5-prompt fixture (3 steps per prompt)

TOK = 100


def fx_cpu12(margin):
    return {"token": TOK, "top": [[TOK, 10.0], [200, 10.0 - margin]], "lse": 10.9}


def fx_source(dev, flip=False, below=False):
    if flip:
        return {"token": 999, "top": [[999, 12.0], [TOK, 10.0 + dev]]}
    logit = 10.0 - dev if below else 10.0 + dev  # cudaOld sits BELOW cpu12: its raw deviation is negative
    return {"token": TOK, "top": [[TOK, logit], [200, 9.0]]}


def fx_fifi(dev, match=True):
    return {"argmax": TOK if match else 999, "argmax_logit": 10.0 + dev,
            "top32": [[TOK if match else 999, 10.0 + dev], [200, 9.0]],
            "lse": 10.9, "hash": "h", "in_token": 5, "forced_logit": 10.0 + dev}


# per prompt: cpu12 margins; cpu6 / cudaOld / cudaNew deviations ("f" = flip); fifi deviations ("m" = a fifi
# token mismatch). Exact binary fractions keep every engineered edge (==) exact in floating point.
FX_PLAN = {
    "P1": ([0.5, 0.25, 1.5], [("f", 0.25), 0.25, 0.25], [0.5, ("f", 0.5), 0.5], [0.625] * 3,
           [0.25, 0.25, 0.78125]),
    "P2": ([1.25, 1.5, 1.5], [0.25] * 3, [0.5, 0.5, 0.25], [0.625, 0.625, 0.25],
           [("m", 0.78125), 0.25, 0.5]),
    "P3": ([1.5] * 3, [0.25] * 3, [0.5] * 3, [0.25, 0.25, 0.625], [0.25, 0.78125, 0.78125]),
    "P4": ([1.0, 1.5, 1.5], [0.25] * 3, [("f", 0.5), 0.5, 0.5], [0.625, 0.625, 0.625], [0.25, 0.78125, 0.5]),
    "P5": ([0.625, 1.5, 1.5], [0.25] * 3, [0.5] * 3, [("f", 0.625), 0.625, 0.625], [0.25, 0.78125, 0.78125]),
}


def write_rulec_fixture(root, plan=None):
    """Oracle (cpu/clean, cpu/threads6, cuda/follow, prompts/), lcpp-new tree, a matching gated-steps manifest and
    the A1/A2 raws, all small (3 steps) and engineered so the rule C edges sit exactly ON the bounds."""
    plan = copy.deepcopy(plan or FX_PLAN)

    def src_step(s, below=False):
        return fx_source(s[1], flip=True) if isinstance(s, tuple) else fx_source(s, below=below)

    def fifi_step(s):
        return fx_fifi(s[1], match=False) if isinstance(s, tuple) else fx_fifi(s)

    inputs = {}
    for p, (margins, c6, old, new, _) in plan.items():
        (root / "oracle" / "prompts").mkdir(parents=True, exist_ok=True)
        write_json(root / "oracle" / "cpu/clean" / p / "run.json", {"steps": [fx_cpu12(m) for m in margins]})
        write_json(root / "oracle" / "cpu/threads6" / p / "run.json", {"steps": [src_step(s) for s in c6]})
        write_json(root / "oracle" / "cuda/follow" / p / "run.json", {"steps": [src_step(s, below=True) for s in old]})
        write_json(root / "lcpp-new" / p / "run.json", {"steps": [src_step(s) for s in new]})
        (root / "oracle" / "prompts" / f"{p}.ids").write_text("10 20 30\n")
        write_json(root / "oracle" / "prompts" / f"{p}.ids.json", {"ids": [10, 20, 30]})
        for rel in (f"cpu/clean/{p}/run.json", f"prompts/{p}.ids.json",
                    f"cpu/threads6/{p}/run.json", f"cuda/follow/{p}/run.json"):
            inputs[rel] = hashlib.sha256((root / "oracle" / rel).read_bytes()).hexdigest()
    write_json(root / "gated.json", {
        "schema": "a5-gated-steps/1", "inputs_sha256": inputs,
        "prompts": {p: {"n_gated": 2, "n_steps": 3, "gated": [], "prompt_ids_sha256": "fx",
                        "n_excluded": 0, "excluded_share_percent": 0.0} for p in plan}})
    for name, ps in (("A1", ["P1", "P3"]), ("A2", ["P2", "P4", "P5"])):
        write_json(root / f"{name}.json", {
            "schema": "cuda-rd-raw/2", "mode": "forced", "incomplete": False, "refused": None,
            "prompts": [{"id": p, "ids_sha256": "fx",
                         "runs": [{"decode_ns": 3_000_000_000,
                                   "phase_ns": {"replay": 30_000_000, "readback": 0, "host": 0},
                                   "steps": [fifi_step(s) for s in plan[p][4]]}]} for p in ps]})
    return root


def write_grade_fixture(root, n=40):
    """A five-prompt forced fixture whose pooled TOTAL line has distinct p95 (0.2) and p99 (0.3); all PASS."""
    devs = [0.1] * 189 + [0.2] * 8 + [0.3] * 2 + [0.4]  # sorted; rank 190 -> 0.2, rank 198 -> 0.3
    all_prompts = sorted(["P1", "P3"] + ["P2", "P4", "P5"])
    inputs = {}
    for pi, p in enumerate(all_prompts):
        steps = [{"token": TOK, "top": [[TOK, 10.0], [200, 9.0]], "lse": 10.9} for _ in range(n)]
        write_json(root / "oracle" / "cpu/clean" / p / "run.json", {"steps": steps})
        (root / "oracle" / "prompts").mkdir(parents=True, exist_ok=True)
        (root / "oracle" / "prompts" / f"{p}.ids").write_text("10 20 30\n")
        write_json(root / "oracle" / "prompts" / f"{p}.ids.json", {"ids": [10, 20, 30]})
        for rel in (f"cpu/clean/{p}/run.json", f"prompts/{p}.ids.json"):
            inputs[rel] = hashlib.sha256((root / "oracle" / rel).read_bytes()).hexdigest()
        run = {"decode_ns": n * 1_000_000_000, "phase_ns": {"replay": n * 10_000_000, "readback": 0, "host": 0},
               "steps": [fx_fifi(devs[(pi * n + j) % len(devs)]) for j in range(n)]}
        for name, ps in (("A1", ["P1", "P3"]), ("A2", ["P2", "P4", "P5"])):
            if p not in ps:
                continue
            doc_path = root / f"{name}.json"
            doc = json.loads(doc_path.read_text()) if doc_path.exists() else {
                "schema": "cuda-rd-raw/2", "mode": "forced", "incomplete": False, "refused": None,
                "prompts": []}
            doc["prompts"].append({"id": p, "ids_sha256": "fx", "runs": [run]})
            write_json(doc_path, doc)
    write_json(root / "gated.json", {
        "schema": "a5-gated-steps/1", "inputs_sha256": inputs,
        "prompts": {p: {"n_gated": 2, "n_steps": n, "gated": [], "prompt_ids_sha256": "fx",
                        "n_excluded": 0, "excluded_share_percent": 0.0} for p in all_prompts}})
    return root


def run_fx(tool, td, extra=(), build=True, raws=None, plan=None):
    """Runs one tool's CLI entry on the fixture; returns (code, stdout, stderr). `raws` replaces grade's raw list."""
    root = td / "fx"
    if build:
        write_rulec_fixture(root, plan=plan)
    if tool is grade:
        argv = [str(root / "oracle")] + [str(root / f"{n}.json") for n in (raws or ("A1", "A2"))] + list(extra)
    else:
        argv = ["--oracle-dir", str(root / "oracle"), "--lcpp-new-dir", str(root / "lcpp-new"),
                "--raw-dir", str(root), "--gated-steps", str(root / "gated.json"),
                "--out", str(td / "retol.md")] + list(extra)
    out, err = io.StringIO(), io.StringIO()
    code = None
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            if tool is grade:
                grade.cli(argv)
            else:
                code = retol.main(argv)
        except SystemExit as e:
            code = e.code
    return code, out.getvalue(), err.getvalue()


@test
def retol_end_to_end_verdict_lines(td):
    code, out, err = run_fx(retol, td)
    assert code == 1, "the fixture's leave-one-out token check fails (a flip above its LOO threshold), so rc is 1"
    for line in (
        "| P1 | 3 | 2 | 1 | 1 | 0.62 | 0.62 | 0.62 |",           # section 1 (n_gated from the manifest)
        "| P3 | 3 | 2 | 3 | 3 | 0.50 | 0.62 | 0.62 |",
        "| P2 | 3 | 1 [0] | 2 | 0  | 1.00 | FAIL |",              # section 2 (superseded rule, informational)
        "| P3 | 3 | 0  | 3 | 0  | 0.78 | PASS |",
        "| all | 12 | 1 | 10 | 0 | | FAIL |",
        "| **cpu6 total** | cudaOld+cudaNew | all | 12 | 0 | 10 | 0 |",   # section 3 leave-one-out
        "| **cudaNew total** | cpu6+cudaOld | all | 13 | 1 | 11 | 0 |",
        "| cudaNew | cpu6+cudaOld | P2 | 3 | 0 | 3 | 0 |",
        "| cudaOld | cpu6+cudaNew | P3 | 3 | 0 | 3 | 0 |",        # its dev 0.5 == 2*S_t is NOT a logit failure
        "| cudaNew | cpu6+cudaOld | P4 | 3 | 0 | 2 | 0 |",
        "- cpu6 only: T=14 L=11 token fails=1 logit fails=6",
        "| cpu6 (held out) | cudaOld+cudaNew | 10 | 0.40 | 1.00 | 1.00 | 1.00 |",   # section 4
        "| cudaNew (held out) | cpu6+cudaOld | 11 | 1.25 | 1.25 | 1.25 | 1.25 |",
        "| fifi | cpu6+cudaOld+cudaNew | 10 | 1.25 | 2.00 | 2.00 | 2.00 |",
        "- cpu6: n=11 0.25 / 0.25 / 0.25 / 0.25",
        "- cudaOld: n=11 0.50 / 0.50 / 0.50 / 0.50",              # negative raw deviations come out positive
        "- cpu6: median 0.250 p95 0.250 p99 0.250 max 0.25",      # section 5: the rule C verdict lines
        "- cudaNew: median 0.625 p95 0.625 p99 0.625 max 0.62",
        "- derived bounds (3 sources): median <= 0.781, p95 <= 0.781, p99 <= 0.781",
        "- fifi row 1: median 0.500 p95 0.781 p99 0.781 max 0.78 -> PASS",   # p95/p99 EXACTLY at their bounds
        "- cudaNew: bounds 0.625/0.625/0.625 vs 0.625/0.625/0.625 -> PASS",  # every statistic EQUAL to its LOO bound
        "- threshold = 1.25 x the largest flip margin among independent sources = 1.250",
        "- fifi top-1 token mismatches on the 10 steps with margin > 1.250: 0",  # the 1.25-margin step sits ON it
        "- LOO cudaOld: threshold from cpu6+cudaNew = 0.781; flips of the held-out source above it: 1",
        "- LOO cudaNew: threshold from cpu6+cudaOld = 1.250; flips of the held-out source above it: 0",
    ):
        assert line in out, f"missing line: {line}"
    # the --out file is the byte-identity artifact: all printed lines, one extra blank line before sections 4 and 5
    sol = out.splitlines()
    for header in ("## 4. Calibration of the factor k (logit set fixed at k=2 step selection; dev/S_t over those steps)",
                   "## 5. Candidate rule C: distribution bounds + margin-based token agreement (derived from independent sources only)"):
        sol.insert(sol.index(header), "")
    assert (td / "retol.md").read_text().splitlines() == sol, "the --out file must hold the whole report"


@test
def retol_stats_p95_and_p99_slots_distinct(td):
    assert retol.stats(list(range(1, 41))) == (20.5, 39, 40, 40), \
        "q(, .95) and q(, .99) hit different ranks at n = 40, so the stats slots cannot be swapped unseen"


def pass_variant():
    """The fixture with every section 5 verdict term passing (rc 0)."""
    plan = copy.deepcopy(FX_PLAN)
    for p in plan:
        plan[p] = [list(g) for g in plan[p]]  # mutable per-prompt fields for the variants below
    plan["P4"][2][0] = 0.5   # the PASS variant has no cudaOld flip at the 1.0-margin step
    plan["P2"][4][0] = 0.78125
    return plan


@test
def retol_pass_variant_exits_zero(td):
    code, out, err = run_fx(retol, td, plan=pass_variant())
    assert code == 0, f"every section 5 term passes here: {code} {err!r}"
    assert "- fifi row 1: median 0.500 p95 0.781 p99 0.781 max 0.78 -> PASS" in out
    assert "- cudaNew: bounds 0.625/0.625/0.625 vs 0.625/0.625/0.625 -> PASS" in out
    assert "- fifi top-1 token mismatches on the 12 steps with margin > 0.781: 0" in out
    assert "- LOO cudaNew: threshold from cpu6+cudaOld = 0.625; flips of the held-out source above it: 0" in out


@test
def retol_each_section5_term_fails_alone(td):
    plan = pass_variant()  # fifi bounds alone: shift every fifi deviation by +3
    for p in plan:
        plan[p][4] = [(d + 3.0 if not isinstance(d, tuple) else d) for d in plan[p][4]]
    code, out, _ = run_fx(retol, td, plan=plan)
    assert code == 1 and "- fifi row 1: median 3.500 p95 3.781 p99 3.781 max 3.78 -> FAIL" in out, f"{code}"
    assert "- cpu6: bounds 0.781/0.781/0.781 vs 0.250/0.250/0.250 -> PASS" in out, "the LOO bounds still pass"
    assert "- cudaNew: bounds 0.625/0.625/0.625 vs 0.625/0.625/0.625 -> PASS" in out
    plan = pass_variant(); plan["P2"][1][1] = 1.0  # leave-one-out bounds alone: cpu6's p95 exceeds its bound
    code, out, _ = run_fx(retol, td, plan=plan)
    assert code == 1 and "- cpu6: bounds 0.781/0.781/0.781 vs 0.250/1.000/1.000 -> FAIL" in out, f"{code}"
    assert "- fifi row 1: median 0.500 p95 0.781 p99 0.781 max 0.78 -> PASS" in out, "the bounds term still passes"
    assert "- LOO cudaNew: threshold from cpu6+cudaOld = 0.625; flips of the held-out source above it: 0" in out
    plan = pass_variant(); plan["P3"][4][0] = ("m", 0.78125)  # token gate alone: a high-margin fifi flip
    code, out, _ = run_fx(retol, td, plan=plan)
    assert code == 1 and "- fifi top-1 token mismatches on the 12 steps with margin > 0.781: 1" in out, f"{code}"
    assert "- fifi row 1: median 0.641 p95 0.781 p99 0.781 max 0.78 -> PASS" in out


@test
def retol_refuses_bad_raw_and_reference_trees(td):
    root = write_rulec_fixture(td / "fx")
    (td / "retol.md").write_text("stale report")  # NIT-7: a refusal must not leave an older report behind
    doc = json.loads((root / "A1.json").read_text())
    doc["prompts"][0]["runs"][0]["steps"][1]["argmax_logit"] = float("nan")
    write_json(root / "A1.json", doc)
    code, out, err = run_fx(retol, td, build=False)
    assert code == 2 and out == "" and "A1.json" in err and "finite" in err, f"{code} {out!r} {err!r}"
    assert not (td / "retol.md").exists(), "the stale --out report is removed on refusal"
    write_rulec_fixture(root)  # restore, then drop P3: the union no longer covers the oracle's five prompts
    doc = json.loads((root / "A1.json").read_text())
    doc["prompts"] = doc["prompts"][:1]
    write_json(root / "A1.json", doc)
    code, out, err = run_fx(retol, td, build=False)
    assert code == 2 and out == "" and "not covered" in err, f"{code} {err!r}"
    doc = json.loads((root / "A1.json").read_text())
    doc["prompts"][0]["runs"][0]["steps"] = doc["prompts"][0]["runs"][0]["steps"][:2]
    write_json(root / "A1.json", doc)
    code, out, err = run_fx(retol, td, build=False)
    assert code == 2 and "steps" in err, f"the oracle step count is wired into the gate: {code} {err!r}"


@test
def retol_refuses_shared_prompt_custom_names(td):
    root = write_rulec_fixture(td / "fx")
    plan = copy.deepcopy(FX_PLAN)

    def raw_file(prompts):
        return {"schema": "cuda-rd-raw/2", "mode": "forced", "incomplete": False, "refused": None,
                "prompts": [{"id": p, "ids_sha256": "fx", "runs": [{"decode_ns": 3_000_000_000,
                             "phase_ns": {"replay": 30_000_000, "readback": 0, "host": 0},
                             "steps": [fx_fifi(s[1], match=False) if isinstance(s, tuple) else fx_fifi(s)
                                       for s in plan[p][4]]}]} for p in prompts]}

    # B1 carries P1 plus a corrupted P2; B2 carries the good P2 plus P4 and P5: last-wins would silently
    # grade the good copy, so the shared id must be refused before anything is printed
    b1 = raw_file(["P1", "P2"])
    b1["prompts"][1]["runs"][0]["steps"][0]["argmax_logit"] = 25.0
    write_json(root / "B1.json", b1)
    write_json(root / "B2.json", raw_file(["P2", "P4", "P5"]))
    out, err = io.StringIO(), io.StringIO()
    code = None
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = retol.main(["--oracle-dir", str(root / "oracle"), "--lcpp-new-dir", str(root / "lcpp-new"),
                               "--raw-dir", str(root), "--run-names", "B1", "B2",
                               "--gated-steps", str(root / "gated.json"), "--out", str(td / "retol.md")])
        except SystemExit as e:
            code = e.code
    assert code == 2 and out.getvalue() == "" and "P2 also comes from" in err.getvalue(), f"{code} {err.getvalue()!r}"


@test
def retol_refuses_bad_reference_trees_and_tampered_oracle(td):
    root = write_rulec_fixture(td / "fx")
    lcpp = json.loads((root / "lcpp-new" / "P2" / "run.json").read_text())
    lcpp["steps"][0]["top"][0][1] = float("nan")
    write_json(root / "lcpp-new" / "P2" / "run.json", lcpp)
    code, out, err = run_fx(retol, td, build=False)
    assert code == 2 and out == "" and "cudaNew P2" in err and "non-finite" in err, \
        f"a NaN cudaNew logit must refuse, not move a bound: {code} {err!r}"
    lcpp["steps"][0]["top"][0][1] = 10.625
    lcpp["steps"] = lcpp["steps"][:2]  # a cut cudaNew tree
    write_json(root / "lcpp-new" / "P2" / "run.json", lcpp)
    code, out, err = run_fx(retol, td, build=False)
    assert code == 2 and "cudaNew P2" in err and "2 steps" in err, f"{code} {err!r}"
    # a tampered oracle file behind the fixture manifest stops before any number (rc 2, was a traceback rc 1)
    clean = json.loads((root / "oracle" / "cpu/threads6" / "P1" / "run.json").read_text())
    tampered = copy.deepcopy(clean)
    tampered["steps"][0]["top"][0][1] += 0.5
    write_json(root / "oracle" / "cpu/threads6" / "P1" / "run.json", tampered)
    out, err = io.StringIO(), io.StringIO()
    code = None
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = retol.cli(["--oracle-dir", str(root / "oracle"), "--lcpp-new-dir", str(root / "lcpp-new"),
                          "--raw-dir", str(root), "--gated-steps", str(root / "gated.json"),
                          "--out", str(td / "retol.md")])
    assert code == 2 and out.getvalue() == "" and "sha256" in err.getvalue(), f"{code} {err.getvalue()!r}"


@test
def retol_default_manifest_keeps_the_frozen_pin(td):
    root = write_rulec_fixture(td / "fx")
    defaults = retol.ev.Reference.__init__.__defaults__
    retol.ev.Reference.__init__.__defaults__ = defaults[:-1] + ("f" * 64,)
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = retol.cli(["--oracle-dir", str(root / "oracle"), "--lcpp-new-dir", str(root / "lcpp-new"),
                              "--raw-dir", str(root), "--out", str(td / "retol.md")])
    finally:
        retol.ev.Reference.__init__.__defaults__ = defaults
    assert code == 2 and "a5-gated-steps.json sha256" in err.getvalue() and "!= frozen" in err.getvalue(), \
        f"the default branch pins the frozen hash: {code} {err.getvalue()!r}"


@test
def grade_end_to_end_fail_pass_and_refuse(td):
    root = td / "fx"
    gated = ["--gated-steps", str(root / "gated.json")]
    code, out, err = run_fx(grade, td, extra=gated)
    assert code == 1, "the fixture's deviations exceed the frozen S3c bounds: VERDICT FAIL must exit 1"
    for line in (
        "== A1.json: prompts ['P1', 'P3'], tiered None, steps graded 6, token mismatches 0",
        "== A2.json: prompts ['P2', 'P4', 'P5'], tiered None, steps graded 8, token mismatches 1",
        "   A5 P1: ok=True fails=0 max_dev=0.0",
        "TOTAL n=14 (token mismatches 1): median 0.500 / p95 0.781 / p99 0.781 / max 0.78",
        "bounds (S3c): <= 0.165 / 0.651 / 1.131; 3090 native control 0.107 / 0.532 / 1.03",
        "VERDICT FAIL",
    ):
        assert line in out, f"missing line: {line}"
    # a NaN in the SECOND file is gated too (rc 2, nothing printed for the first)
    doc = json.loads((root / "A2.json").read_text())
    doc["prompts"][0]["runs"][0]["steps"][1]["argmax_logit"] = float("nan")
    write_json(root / "A2.json", doc)
    code, out, err = run_fx(grade, td, extra=gated, build=False)
    assert code == 2 and out == "" and "finite" in err, f"{code} {out!r} {err!r}"
    write_rulec_fixture(root)  # restore
    # a partial prompt set under a foreign file name (the old VERDICT PASS trap), a repeated file, runs: []
    doc = json.loads((root / "A1.json").read_text())
    doc["prompts"] = doc["prompts"][:1]
    write_json(root / "NAT-A1.json", doc)
    code, out, err = run_fx(grade, td, extra=gated, raws=("NAT-A1", "A2"), build=False)
    assert code == 2 and out == "" and "not covered" in err, f"{code} {out!r} {err!r}"
    code, out, err = run_fx(grade, td, extra=gated + ["--allow-partial"], raws=("NAT-A1", "A2"), build=False)
    assert code == 1 and "VERDICT PARTIAL" in out, f"{code} {out!r}"
    code, out, err = run_fx(grade, td, extra=gated, raws=("A1", "A1"), build=False)
    assert code == 2 and "also comes from" in err, f"{code} {err!r}"
    doc = json.loads((root / "A2.json").read_text())
    doc["prompts"][0]["runs"] = []
    write_json(root / "noruns.json", doc)
    code, out, err = run_fx(grade, td, extra=gated, raws=("A1", "noruns"), build=False)
    assert code == 2 and "no runs" in err, f"{code} {err!r}"
    doc = json.loads((root / "A1.json").read_text())
    doc["prompts"][0]["runs"][0]["steps"] = doc["prompts"][0]["runs"][0]["steps"][:2]  # 2 of the oracle's 3
    write_json(root / "trunc.json", doc)
    code, out, err = run_fx(grade, td, extra=gated, raws=("trunc",), build=False)
    assert code == 2 and out == "" and "trunc.json" in err and "steps" in err, f"{code} {err!r}"
    # the distinct p95/p99 fixture: full coverage, everything passes, the TOTAL ranks cannot be swapped unseen
    groot = write_grade_fixture(td / "gx")
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            grade.cli([str(groot / "oracle"), str(groot / "A1.json"), str(groot / "A2.json"),
                       "--gated-steps", str(groot / "gated.json")])
        except SystemExit as e:
            code = e.code
    assert code == 0 and "TOTAL n=200 (token mismatches 0): median 0.100 / p95 0.200 / p99 0.300 / max 0.40" \
        in out.getvalue() and "VERDICT PASS" in out.getvalue(), f"{code} {out.getvalue()!r}"
    # a tampered oracle file behind the fixture manifest stops before any number (rc 2)
    tampered = json.loads((groot / "oracle" / "cpu/clean" / "P2" / "run.json").read_text())
    tampered["steps"][0]["top"][0][1] += 0.5
    write_json(groot / "oracle" / "cpu/clean" / "P2" / "run.json", tampered)
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            grade.cli([str(groot / "oracle"), str(groot / "A1.json"), str(groot / "A2.json"),
                       "--gated-steps", str(groot / "gated.json")])
        except SystemExit as e:
            code = e.code
    assert code == 2 and out.getvalue() == "" and "sha256" in err.getvalue(), f"{code} {err.getvalue()!r}"
    # the default manifest keeps the frozen pin (the fixture oracle cannot be graded under it)
    defaults = grade.ev.Reference.__init__.__defaults__
    grade.ev.Reference.__init__.__defaults__ = defaults[:-1] + ("f" * 64,)
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                grade.cli([str(groot / "oracle"), str(groot / "A1.json"), str(groot / "A2.json")])
            except SystemExit as e:
                code = e.code
    finally:
        grade.ev.Reference.__init__.__defaults__ = defaults
    assert code == 2 and "a5-gated-steps.json sha256" in err.getvalue() and "!= frozen" in err.getvalue(), \
        f"{code} {err.getvalue()!r}"
    # an internal error exits 3 (restore the untampered fixture first)
    shutil.rmtree(groot)
    write_grade_fixture(groot)
    real_gf = grade.grade_file
    def boom(*a):
        raise RuntimeError("boom")
    grade.grade_file = boom
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                grade.cli([str(groot / "oracle"), str(groot / "A1.json"), str(groot / "A2.json"),
                           "--gated-steps", str(groot / "gated.json")])
            except SystemExit as e:
                code = e.code
    finally:
        grade.grade_file = real_gf
    assert code == 3 and "boom" in err.getvalue(), f"{code} {err.getvalue()!r}"


def main():
    print(f"tools under test: {TOOLS}")
    failures = []
    for fn in TESTS:
        try:
            with tempfile.TemporaryDirectory(prefix="rulec-selftest-") as td:
                fn(Path(td))
        except Exception:
            failures.append(fn.__name__)
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
        else:
            print(f"PASS {fn.__name__}")
    print(f"{len(TESTS) - len(failures)}/{len(TESTS)} passed")
    if failures:
        print("FAILED:", ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
