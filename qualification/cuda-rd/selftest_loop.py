#!/usr/bin/env python3
"""CPU self-test of the `call` loop and the state handling of run_stage1.sh, with a stub for gpu_run.sh and for hold.sh (a COPY of the
qualification tree is used, with the baton path replaced; the production script is not modified). Covers: reset (a rerun starts clean),
a void then a good run, three consecutive voids (BLOCKED, exit 7), REFUSED (exit 3), STOP (exit 2), a stale `.skipped` removed, the budget
checked before EVERY attempt, and the clean-tree check. Usage: python3 -I selftest_loop.py SCRATCH_DIR"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a5_eval as ev  # noqa: E402
import fixtures_files as fx  # noqa: E402

# A caller's RD10_* variables (a real eager pre-run sets RD10_CAPTURE, RD10_DATE, RD10_HEAD) must not leak into the stubbed runs.
BASE_ENV = {k: v for k, v in os.environ.items() if not k.startswith("RD10_")}
T = Path(sys.argv[1]) / "loop"
results = []


def check(name, cond):
    results.append(cond)
    print(("ok   " if cond else "FAIL ") + name)


def setup():
    shutil.rmtree(T, ignore_errors=True)
    (T / "tree").mkdir(parents=True)
    shutil.copytree(HERE, T / "tree", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    script = (T / "tree/run_stage1.sh").read_text()
    script = script.replace("BATON=/media/dan/nvme/fifi/scrch/c62fd02/gpu-baton/gpu_run.sh", f"BATON={T}/gpu_run.sh")
    script = script.replace("BDIR=/media/dan/nvme/fifi/scrch/c62fd02/gpu-baton", f"BDIR={T}/baton")
    (T / "tree/run_stage1.sh").write_text(script)
    (T / "baton").mkdir()
    (T / "gpu_run.sh").write_text('#!/usr/bin/env bash\nshift\nexec "$@"\n')
    (T / "gpu_run.sh").chmod(0o755)
    (T / "tree/hold.sh").write_text(f'''#!/usr/bin/env bash
watch="$3"; shift 3; out=
while [ $# -gt 0 ]; do [ "$1" = --out ] && out="$2"; shift; done
n=$(cat {T}/count 2>/dev/null || echo 0); n=$((n+1)); echo $n > {T}/count
outcome=$(sed -n "${{n}}p" {T}/outcomes)
case "$outcome" in
  good) cp {T}/good.json "$out"; cp {T}/good.watch "$watch" ;;
  void) cp {T}/good.json "$out"; cp {T}/void.watch "$watch" ;;
  refused) exit 3 ;;
  stop) exit 2 ;;
esac
echo "HOLD seconds 100"
''')
    ref = ev.Reference()
    doc = fx.make_file(ref, "A1")
    (T / "good.json").write_text(json.dumps(doc))
    for name, kw in (("good", {}), ("void", {"floor_dip": 1394})):
        (T / f"{name}.watch").write_text("\n".join(json.dumps(s) for s in fx.make_watch(doc, **kw)) + "\n")
    repo = T / "wt"
    (repo / "qualification").mkdir(parents=True)
    (repo / "qualification/cuda-rd").symlink_to(T / "tree")
    g = lambda *a: subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *a], capture_output=True, text=True, check=True)  # noqa: E731
    g("init", "-q")
    (repo / "tracked.txt").write_text("x")
    g("add", "-A"); g("commit", "-q", "-m", "c")      # the symlink to the tool tree is committed too: the tree must be clean


def sdir(name):
    d = T / f"S-{name}"
    shutil.rmtree(d, ignore_errors=True)
    (d / "raw").mkdir(parents=True)
    (d / "logs").mkdir()
    (d / "release.json").write_text(json.dumps({"exe_sha256": "e" * 64}))
    (d / "gguf.json").write_text(json.dumps({"sha256": "a" * 64}))
    return d


def run(d, outcomes, *args, state=None):
    (T / "outcomes").write_text("\n".join(outcomes) + "\n")
    (T / "count").write_text("0")
    for k, v in (state or {}).items():
        (d / k).write_text(str(v))
    env = {**BASE_ENV, "RD10_S": str(d), "RD10_WT": str(T / "wt"), "RD10_HEAD": "HEAD", "RD10_DATE": "2026-10-10"}
    return subprocess.run(["bash", str(T / "tree/run_stage1.sh"), *args], capture_output=True, text=True, env=env)


def read(d, name):
    p = d / name
    return p.read_text().strip() if p.exists() else None


def run_idle(d, fake_smi):
    env = {**BASE_ENV, "RD10_S": str(d), "RD10_WT": str(T / "wt"), "PATH": f"{fake_smi}:{os.environ['PATH']}"}
    return subprocess.run(["bash", str(T / "tree/run_stage1.sh"), "idle"], capture_output=True, text=True, env=env)


def run_check_build(d, head):
    """16c F1: check-build must STOP unless the binary's build-info commit is a full 40-hex commit equal to RD10_HEAD.
    A fake cuda-rd-run at the scratch target path stands in for the built binary; a fake docker answers the cuobjdump
    calls of check_release_build.py (the release profile itself is selftest_release_check.sh's subject)."""
    return subprocess.run(["bash", str(T / "tree/run_stage1.sh"), "check-build"], capture_output=True, text=True,
                          env={**BASE_ENV, "RD10_S": str(d), "RD10_WT": str(T / "wt"), "RD10_HEAD": head,
                               "PATH": f"{T / 'fake-docker'}:{os.environ['PATH']}"})


def two_rows():
    """The eager pre-run row first, committed, then the captured gate row in the same worktree with its own scratch dir."""
    from report_case import release, wrapper_log
    ref = ev.Reference()
    wt = T / "wt3"
    shutil.rmtree(wt, ignore_errors=True)
    (wt / "qualification/cuda-rd").mkdir(parents=True)
    (wt / "qualification/cuda-rd/placeholder.txt").write_text("x")
    g = lambda *a: subprocess.run(["git", "-C", str(wt), "-c", "user.name=t", "-c", "user.email=t@t", *a], capture_output=True, text=True, check=True).stdout.strip()  # noqa: E731
    g("init", "-q"); g("add", "-A"); g("commit", "-q", "-m", "base")
    # the copy of the tool tree has its own manifest (its run_stage1.sh was edited for the stub baton)
    subprocess.run(["bash", str(T / "tree/make_manifest.sh")], capture_output=True, check=True)

    def row(name, capture, head):
        d = sdir(name)
        docs = fx.make_all(ref, capture=capture)
        for doc in docs.values():
            doc["identities"]["engine_commit"] = head
        fx.write_all(docs, d / "raw")
        (d / "release.json").write_text(json.dumps(release()))
        (d / "wrapper.log").write_text(wrapper_log(head))
        hashes = subprocess.run([sys.executable, "-I", "-c", "import json,sys; sys.path.insert(0, sys.argv[1]); import assemble_report as a; print(json.dumps({'rc': 0, 'scripts': a.script_hashes()}))", str(T / "tree")],
                                capture_output=True, text=True, check=True).stdout
        (d / "selftest-result.json").write_text(hashes)
        return d

    def report(d, capture, head):
        env = {**BASE_ENV, "RD10_S": str(d), "RD10_WT": str(wt), "RD10_HEAD": head, "RD10_DATE": "2026-10-10"}
        if capture:
            env["RD10_CAPTURE"] = capture
        return subprocess.run(["bash", str(T / "tree/run_stage1.sh"), "report"], capture_output=True, text=True, env=env)

    h1 = g("rev-parse", "HEAD")
    d1 = row("row-eager", "eager", h1)
    r = report(d1, "eager", h1)
    eager_file = wt / "qualification/cuda-rd/stage1-eager-prerun-2026-10-10.md"
    check("EAGER-FIRST row (RD-11a absent): the report is accepted as a PRE-RUN (exit 4, `PRE-RUN complete (not a stage gate)`), file stage1-eager-prerun-<date>.md", r.returncode == 4 and "PRE-RUN complete (not a stage gate)" in r.stdout and "PASS" not in r.stdout and eager_file.exists())
    text = eager_file.read_text()
    t3 = text.split("## Table 3")[1].split("## Hold records")[0]
    check("the eager-first report is labelled EAGER PRE-RUN, not a stage gate, never prints PASS for stage 1, and states CaptureOutcome Eager", "EAGER PRE-RUN, not a stage gate" in text and "PASS" not in t3 and "tables pass (information only" in t3 and "CaptureOutcome" in text)
    r = subprocess.run(["bash", str(T / "tree/run_stage1.sh"), "report"], capture_output=True, text=True, env={**BASE_ENV, "RD10_S": str(d1), "RD10_WT": str(wt), "RD10_HEAD": h1, "RD10_DATE": "eager-prerun-2026-10-10", "RD10_CAPTURE": "eager"})
    check("RD10_DATE must be exactly YYYY-MM-DD: `eager-prerun-2026-10-10` -> STOP, nothing written (lane L's N4)", r.returncode == 2 and "YYYY-MM-DD" in r.stdout)
    for bad_date in ("2026-1-10", "20261010", "2026-10-10x"):
        r = subprocess.run(["bash", str(T / "tree/run_stage1.sh"), "report"], capture_output=True, text=True, env={**BASE_ENV, "RD10_S": str(d1), "RD10_WT": str(wt), "RD10_HEAD": h1, "RD10_DATE": bad_date, "RD10_CAPTURE": "eager"})
        check(f"RD10_DATE {bad_date!r} -> STOP", r.returncode == 2 and "YYYY-MM-DD" in r.stdout)
    aside = T / "eager-report.aside"
    eager_file.rename(aside)                       # so that clean-tree passes and the EVALUATOR is what refuses
    r = report(d1, None, h1)
    aside.rename(eager_file)
    check("the same eager data reported WITHOUT RD10_CAPTURE=eager (a gate attempt) is refused (STOP exit 2): eager data can never become a stage 1 report", r.returncode == 2 and "every run is eager" in r.stdout and not (wt / "qualification/cuda-rd/stage1-2026-10-10.md").exists())
    g("add", "-A"); g("commit", "-q", "-m", "stage 1 eager pre-run record")
    before = eager_file.read_text()
    r = report(d1, "eager", g("rev-parse", "HEAD"))
    check("a COMMITTED report for the same date is never overwritten: the second eager report STOPs and the file is unchanged (lane L's N3)", r.returncode == 2 and "already exists" in r.stdout and eager_file.read_text() == before)
    h2 = g("rev-parse", "HEAD")
    d2 = row("row-captured", "graph", h2)
    env = {**BASE_ENV, "RD10_S": str(d2), "RD10_WT": str(wt), "RD10_HEAD": h2, "RD10_DATE": "2026-10-10"}
    sh = lambda *a: subprocess.run(["bash", str(T / "tree/run_stage1.sh"), *a], capture_output=True, text=True, env=env)  # noqa: E731
    check("the later captured row: `reset` leaves the COMMITTED eager report alone and clean-tree passes", sh("reset").returncode == 0 and eager_file.exists() and sh("clean-tree").returncode == 0)
    # reset emptied the raw dir of the (fresh) captured scratch: write it again, as the calls would
    fx.write_all(fx.make_all(ref), d2 / "raw")
    for doc in (json.loads(p.read_text()) for p in sorted((d2 / "raw").glob("*.json"))):
        pass
    docs = fx.make_all(ref)
    for doc in docs.values():
        doc["identities"]["engine_commit"] = h2
    fx.write_all(docs, d2 / "raw")
    r = sh("report")
    gate_file = wt / "qualification/cuda-rd/stage1-2026-10-10.md"
    check("the captured gate row: exit 0 PASS, a DISTINCT file name stage1-<date>.md, and the first row's report is untouched", r.returncode == 0 and gate_file.exists() and eager_file.exists() and gate_file != eager_file)
    check("only the captured row's report says PASS for stage 1", "| PASS |" in gate_file.read_text() and "EAGER PRE-RUN" not in gate_file.read_text())


def main():
    setup()
    d = sdir("reset")
    r = run(d, ["good"], "reset", state={"void-count": 2, "budget-seconds": 2300, "last-hold-seconds": 300})
    check("reset removes the void count, the budget and the last hold length", r.returncode == 0 and all(read(d, f) is None for f in ("void-count", "budget-seconds", "last-hold-seconds")))
    (d / "raw/stale.json").write_text("{}")
    run(d, [], "reset")
    check("reset also empties the raw directory", not list((d / "raw").glob("*")))
    r = run(d, ["good"], "call", "A1")
    check("a call after reset starts clean: exit 0, budget equals this call only (100 s)", r.returncode == 0 and read(d, "budget-seconds") == "100" and read(d, "void-count") == "0")
    d = sdir("stale")
    r = run(d, ["void"], "call", "A1", state={"void-count": 2})
    check("WITHOUT a reset a stale void count of 2 makes one void BLOCKED at once (why Step 1 resets)", r.returncode == 7)
    d = sdir("void1")
    r = run(d, ["void", "good"], "call", "A1")
    check("one void, then a good run: exit 0 and the void count is back at 0", r.returncode == 0 and read(d, "void-count") == "0" and read(d, "raw/A1.json") is not None)
    d = sdir("void3")
    r = run(d, ["void", "void", "void"], "call", "A1")
    check("three consecutive voids -> exit 7 BLOCKED, A1.blocked written, the clock logs kept", r.returncode == 7 and read(d, "raw/A1.blocked") is not None and len(list((d / "logs").glob("clock-A1-void*.jsonl"))) == 3)
    d = sdir("ref")
    r = run(d, ["refused"], "call", "A1")
    check("the binary's REFUSED (exit 3) stays exit 3, not STOP", r.returncode == 3)
    d = sdir("stop")
    r = run(d, ["stop"], "call", "A1")
    check("the binary's STOP (exit 2) stays exit 2", r.returncode == 2)
    d = sdir("skipped")
    (d / "raw/A1.skipped").write_text("old")
    r = run(d, ["good"], "call", "A1")
    check("a stale A1.skipped is removed at the start of a call", r.returncode == 0 and read(d, "raw/A1.skipped") is None)
    d = sdir("budget")
    r = run(d, ["good"], "call", "A1", state={"budget-seconds": 2300, "last-hold-seconds": 100})
    check("the budget is checked before the first hold: 2300 s used -> skipped (exit 1, A1.skipped)", r.returncode == 1 and read(d, "raw/A1.skipped") == "budget" and read(d, "count") is None)
    d = sdir("budget-retry")
    r = run(d, ["void", "good"], "call", "A1", state={"budget-seconds": 2200, "last-hold-seconds": 100})
    check("the budget is checked before EVERY attempt: after the voided attempt (2300 s used) the retry is skipped", r.returncode == 1 and read(d, "raw/A1.skipped") == "budget" and read(d, "raw/A1.json") is None)
    d = sdir("logs")
    (d / "logs/call-A1.log").write_text("HOLD seconds 77\n")
    r = run(d, [], "reset")
    moved = list((d / "logs").glob("prev-*/call-A1.log"))
    check("reset MOVES the old call logs into logs/prev-<time>/ (kept as evidence), so a new report lists only its own holds", r.returncode == 0 and not (d / "logs/call-A1.log").exists() and len(moved) == 1 and "HOLD seconds 77" in moved[0].read_text())
    d = sdir("leftover")
    wt2 = T / "wt2"                                   # a worktree where qualification/cuda-rd is a real directory
    shutil.rmtree(wt2, ignore_errors=True)
    (wt2 / "qualification/cuda-rd").mkdir(parents=True)
    (wt2 / "qualification/cuda-rd/placeholder.txt").write_text("x")
    g2 = lambda *a: subprocess.run(["git", "-C", str(wt2), "-c", "user.name=t", "-c", "user.email=t@t", *a], capture_output=True, text=True, check=True)  # noqa: E731
    g2("init", "-q"); g2("add", "-A"); g2("commit", "-q", "-m", "c")
    (wt2 / "qualification/cuda-rd/stage1-2026-10-10.md").write_text("# an earlier report\n")
    env2 = {**BASE_ENV, "RD10_S": str(d), "RD10_WT": str(wt2), "RD10_HEAD": "HEAD", "RD10_DATE": "2026-10-10"}
    sh = lambda *a: subprocess.run(["bash", str(T / "tree/run_stage1.sh"), *a], capture_output=True, text=True, env=env2)  # noqa: E731
    check("a leftover stage1-<date>.md is untracked: clean-tree (so a second `report`) would STOP", sh("clean-tree").returncode == 2)
    r = sh("reset")
    moved = list((d / "logs").glob("prev-*/stage1-2026-10-10.md"))
    check("reset MOVES the leftover report into logs/prev-<time>/ (lane L's N10)", r.returncode == 0 and len(moved) == 1 and not (wt2 / "qualification/cuda-rd/stage1-2026-10-10.md").exists())
    check("after the reset clean-tree passes again, so the second `report` runs", sh("clean-tree").returncode == 0)
    two_rows()
    d = sdir("tree")
    check("clean-tree: a clean worktree -> exit 0", run(d, [], "clean-tree").returncode == 0)
    (T / "wt/tracked.txt").write_text("changed")
    r = run(d, [], "clean-tree")
    check("clean-tree: an uncommitted change to a tracked file -> STOP exit 2", r.returncode == 2 and "uncommitted" in r.stdout)
    (T / "wt/tracked.txt").write_text("x")
    (T / "wt/.cargo").mkdir(exist_ok=True)
    (T / "wt/.cargo/config.toml").write_text("[build]\n")
    r = run(d, [], "clean-tree")
    check("clean-tree: an UNTRACKED .cargo/config.toml -> STOP exit 2 (lane L's N4)", r.returncode == 2 and "untracked" in r.stdout)
    (T / "wt/.cargo/config.toml").unlink()
    (T / "wt/.cargo").rmdir()
    (T / "wt/.gitignore").write_text("ignored.txt\n")
    subprocess.run(["git", "-C", str(T / "wt"), "add", ".gitignore"], check=True)
    subprocess.run(["git", "-C", str(T / "wt"), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "ign"], check=True)
    (T / "wt/ignored.txt").write_text("x")
    check("clean-tree: a file that .gitignore lists is fine", run(d, [], "clean-tree").returncode == 0)
    busy, idle_smi = T / "smi-busy", T / "smi-idle"
    for dir_, apps in ((busy, "4242, 900 MiB\n"), (idle_smi, "")):
        dir_.mkdir(exist_ok=True)
        (dir_ / "nvidia-smi").write_text(f'#!/usr/bin/env bash\ncase "$*" in\n  *query-compute-apps*) printf \'%s\' "{apps}" ;;\n  *) echo "0, 414, 210, 405, 37, 24.26, 580.159.03, P8" ;;\nesac\n')
        (dir_ / "nvidia-smi").chmod(0o755)
    d = sdir("idle")
    check("run_stage1 idle: a busy card is a typed STOP (exit 2), never the FAIL code 1", run_idle(d, busy).returncode == 2)
    check("run_stage1 idle: an idle card exits 0", run_idle(d, idle_smi).returncode == 0)
    dock = T / "fake-docker"
    dock.mkdir(exist_ok=True)
    (dock / "docker").write_text('#!/usr/bin/env bash\ncase "$*" in\n  *-elf*) printf \'Fatbin elf code (SM86)\\n\\tarch = sm_86\\n\' ;;\n  *--version*) echo "Cuda compilation tools, release 12.2, V12.2.140" ;;\n  *) exit 3 ;;\nesac\n')
    (dock / "docker").chmod(0o755)
    d = sdir("check-build")
    fakebin = d / "target/release/cuda-rd-run"
    fakebin.parent.mkdir(parents=True)
    info = lambda commit: json.dumps({"profile": "release", "opt_level": "3", "debug": False, "debug_assertions": False, "commit": commit})  # noqa: E731
    good, wrong = "a" * 40, "b" * 40
    for commit_value in ("unknown", good, wrong):
        fakebin.write_text(f'#!/usr/bin/env bash\ncat <<\'EOF\'\n{info(commit_value)}\nEOF\n')
        fakebin.chmod(0o755)
        if commit_value == "unknown":
            r = run_check_build(d, good)
            check("run_stage1 check-build: a binary that records commit 'unknown' (the git-less image) -> STOP exit 2",
                  r.returncode == 2 and "not a full 40-hex commit" in r.stdout)
        elif commit_value == good:
            r = run_check_build(d, good)
            check("run_stage1 check-build: a binary that records RD10_HEAD -> exit 0 (release.json written)",
                  r.returncode == 0 and json.loads((d / "release.json").read_text())["build_info"]["commit"] == good)
        else:
            r = run_check_build(d, good)
            check("run_stage1 check-build: a binary that records a different 40-hex commit -> STOP exit 2 naming both",
                  r.returncode == 2 and f"records commit {wrong}, RD10_HEAD is {good}" in r.stdout)
    print("LOOP SELFTEST", "OK" if all(results) else "FAILED", f"({len(results)} checks)")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
