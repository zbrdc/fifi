"""RD-10 parser of the RD-2e wrapper-test log (release, --nocapture), lane C's line format (head 40d670d).
CPU only, stdlib only, `python3 -I`. Design guidance only, no code copied.
"""
import re

WRAPPER_FILTER = "projection_repeat_"
WRAPPER_TESTS = 5


def parse_wrapper_log(text, shapes, commit):
    """RD-2e log (release, --nocapture), lane C's format (head 40d670d), one line per shape name: a dense or lm_head line
    carries the shape name and `equal`; an expert line carries the shape name and `repeats=1000`. All lines and
    `5 passed` are required; `repeats=1000` is NOT required on the dense lines. Returns a list of problems."""
    bad = []
    lines = text.splitlines()
    if f"engine_commit {commit}" not in lines:
        bad.append("log header does not name the engine commit under test")
    if "profile release" not in lines:
        bad.append("log header does not say profile release")
    names = {m.group(1) for ln in lines if (m := re.match(rf"^test (\S*{WRAPPER_FILTER}\S*) \.\.\.", ln))}
    if len(names) != WRAPPER_TESTS:
        bad.append(f"{len(names)} distinct `{WRAPPER_FILTER}` tests named, expected {WRAPPER_TESTS}")
    if any(re.match(rf"^test \S*{WRAPPER_FILTER}\S* \.\.\. ignored", ln) for ln in lines):
        bad.append("a `projection_repeat_` test is #[ignore]d: it must run explicitly (`--ignored`, with its env), never silently skip")
    passed = sum(int(ln.split("ok. ")[1].split(" passed")[0]) for ln in lines if ln.startswith("test result: ok. "))
    failed = sum(int(ln.split("; ")[1].split(" failed")[0]) for ln in lines if ln.startswith("test result:"))
    if passed != WRAPPER_TESTS or failed:
        bad.append(f"test result totals: {passed} passed, {failed} failed (expected {WRAPPER_TESTS} and 0)")
    for sh in shapes:
        # with --nocapture the output may follow `test <name> ... ` on the same line: match the shape name anywhere as a word
        hit = [ln for ln in lines if re.search(rf"(^|\s){sh['name']}\s", ln) and sh["shape"] in ln and sh["match"] in ln]
        if not hit:
            bad.append(f"no passing line for {sh['name']} {sh['shape']} (needs `{sh['match']}`)")
    return bad
