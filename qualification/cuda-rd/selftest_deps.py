#!/usr/bin/env python3
"""CPU self-test of check_deps.py on a throw-away git repository under SCRATCH (never a fifi repository).
Usage: python3 -I selftest_deps.py SCRATCH_DIR"""
import os
import stat
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check_deps as cd  # noqa: E402

root = Path(sys.argv[1]) / "depsrepo"
SUBJECTS = ["chore(cuda): merge RD-9b1 resident state book and device owner",
            "chore(cuda): merge RD-9b2 router scratch, layer table and mixer geometry",
            "chore(merge): RD-9b3a DeltaNet block executor",
            "chore(merge): RD-9b3b attention block ops, shapes and executor arms"]
FILES = {"engine/src/a.rs": "struct QwenResidentStep; // INFERENCE_CUDA_SYNC_EACH\n",
         "engine/src/kernels/cuda/context/launch/tests/repeat.rs": "fn projection_repeat_dense() {}\n",
         "engine/src/kernels/cuda/resident/matmul_expert/tests/repeat.rs": "fn projection_repeat_expert() {}\n"}


def mkrepo(r, merges, plains=()):
    """A throw-away repository with the base files; every subject in `merges` lands as a REAL merge commit (two
    parents), every subject in `plains` as a plain commit carrying the same subject (16c F5: only a merge satisfies
    a split). The splits land in order, so each merge is an ancestor of the next. Returns (head, base, {subject: sha})."""
    subprocess.run(["rm", "-rf", str(r)], check=True)
    r.mkdir(parents=True)

    def gg(*a):
        return subprocess.run(["git", "-C", str(r), "-c", "user.name=t", "-c", "user.email=t@t", *a], capture_output=True, text=True, check=True).stdout.strip()

    gg("init", "-q")
    for f, c in FILES.items():
        (r / f).parent.mkdir(parents=True, exist_ok=True)
        (r / f).write_text(c)
    gg("add", "-A"); gg("commit", "-q", "-m", "base")
    base = gg("rev-parse", "HEAD")
    main = gg("rev-parse", "--abbrev-ref", "HEAD")
    shas = {}
    for i, subj in enumerate(SUBJECTS):
        if subj not in merges and subj not in plains:
            continue
        gg("checkout", "-q", "-b", f"split{i}")
        (r / f"split{i}.txt").write_text(subj + "\n"); gg("add", "-A"); gg("commit", "-q", "-m", f"{subj} (work)")
        gg("checkout", "-q", main)
        if subj in merges:
            gg("merge", "--no-ff", "-q", "--no-edit", "-m", subj, f"split{i}")
        else:
            gg("merge", "--squash", "-q", "--no-edit", f"split{i}")
            gg("commit", "-q", "-m", subj)
        shas[subj] = gg("rev-parse", "HEAD")
    (r / "head.txt").write_text("x"); gg("add", "-A"); gg("commit", "-q", "-m", "head")
    return gg("rev-parse", "HEAD"), base, shas


def g(*a):
    return subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", *a], capture_output=True, text=True, check=True).stdout.strip()


head, base, split_shas = mkrepo(root, merges=set(SUBJECTS))
pre3b = split_shas[SUBJECTS[2]]            # the RD-9b1, RD-9b2 and RD-9b3a merges are ancestors here; RD-9b3b exists but is not
g("checkout", "-q", "-b", "other", base)
(root / "other.txt").write_text("y"); g("add", "-A"); g("commit", "-q", "-m", "other")
other = g("rev-parse", "HEAD")
deps = Path(sys.argv[1]) / "deps.txt"


def run(lines, h=head):
    deps.write_text("\n".join(lines) + "\n")
    return cd.check(str(root), h, str(deps))


allgood = [f"{n} {base}" for n in cd.REQUIRED]
res = []
res.append(run(allgood) == [])
res.append(any("RD-9d" in b for b in run([x for x in allgood if not x.startswith("RD-9d")])))
res.append(any("not an ancestor" in b for b in run([x if not x.startswith("RD-11a") else f"RD-11a {other}" for x in allgood])))
def run_eager(lines):
    deps.write_text("\n".join(lines) + "\n")
    return cd.check(str(root), head, str(deps), eager=True)


no11a = [x for x in allgood if not x.startswith("RD-11a")]
res.append(run_eager(no11a) == [])
res.append(run_eager([x if not x.startswith("RD-11a") else "RD-11a pending merge" for x in allgood]) == [])
res.append(any("RD-9d" in b for b in run_eager([x for x in no11a if not x.startswith("RD-9d")])))
res.append(any("RD-8b" in b for b in run_eager([x if not x.startswith("RD-8b") else "RD-8b pending merge" for x in no11a])))
res.append(any("RD-11a" in b for b in run(no11a)))
res.append(run([x for x in allgood if not x.startswith("RD-9b ")]) == [])       # all four split merges present -> the name RD-9b is satisfied
res.append(any("RD-9b3b" in b for b in run(allgood, h=pre3b)))                 # one split merge missing -> STOP naming it
# 16c F4: each of RD-9b1, RD-9b2 and RD-9b3a missing on its own (the other three merged) -> STOP naming exactly that split
for idx, name in ((0, "RD-9b1"), (1, "RD-9b2"), (2, "RD-9b3a")):
    r = Path(sys.argv[1]) / f"depsrepo-no{name}"
    h2, b2, _ = mkrepo(r, merges=set(SUBJECTS) - {SUBJECTS[idx]})
    deps2 = Path(sys.argv[1]) / f"deps-{name}.txt"
    deps2.write_text("\n".join([f"{n} {b2}" for n in cd.REQUIRED]) + "\n")
    res.append(any(f"split merge {name}: " in b for b in cd.check(str(r), h2, str(deps2))))
# 16c F5: a plain (non-merge) commit whose subject names the split — and is an ancestor — never satisfies it
r4 = Path(sys.argv[1]) / "depsrepo-plain9b1"
h4, b4, _ = mkrepo(r4, merges=set(SUBJECTS[1:]), plains=(SUBJECTS[0],))
deps4 = Path(sys.argv[1]) / "deps-plain9b1.txt"
deps4.write_text("\n".join([f"{n} {b4}" for n in cd.REQUIRED]) + "\n")
res.append(any("RD-9b1" in b and "not a merge commit" in b for b in cd.check(str(r4), h4, str(deps4))))
import contextlib
import io
with contextlib.redirect_stdout(io.StringIO()) as out1:
    run_eager(allgood)
with contextlib.redirect_stdout(io.StringIO()) as out2:
    run_eager([x if not x.startswith("RD-11a") else f"RD-11a {other}" for x in allgood])
with contextlib.redirect_stdout(io.StringIO()) as out3:
    run_eager(no11a)
res.append("present, an ancestor of HEAD" in out1.getvalue())
res.append("NOT an ancestor of HEAD" in out2.getvalue() and run_eager([x if not x.startswith("RD-11a") else f"RD-11a {other}" for x in allgood]) == [])
res.append("absent" in out3.getvalue())
hung = Path(sys.argv[1]) / "hung-bin"; hung.mkdir(exist_ok=True)
(hung / "git").write_text("#!/usr/bin/env bash\nsleep 30\n"); (hung / "git").chmod(0o755)
real_run = cd.subprocess.run
cd.subprocess.run = lambda cmd, **k: real_run(cmd, **{**k, "timeout": 0.2})   # shrink the 120 s; same raise site
saved = os.environ["PATH"]; os.environ["PATH"] = f"{hung}:{saved}"
code = None
try:
    cd.check(str(root), head, str(deps))
except SystemExit as e:
    code = e.code
finally:
    cd.subprocess.run = real_run; os.environ["PATH"] = saved
res.append(code == 2)
print("deps: good empty, missing RD-9d named, non-ancestor RD-11a named; eager mode accepts RD-11a absent or pending, still names RD-9d and a pending RD-8b; captured mode names RD-11a; the name RD-9b is satisfied by its four split MERGE commits, one missing split is named, each of RD-9b1/RD-9b2/RD-9b3a missing on its own is named, and a plain commit with a split's subject is refused; a hung git is a typed STOP (exit 2) ->", "OK" if all(res) else f"FAILED {res}")
sys.exit(0 if all(res) else 1)
