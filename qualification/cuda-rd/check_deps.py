#!/usr/bin/env python3
"""RD-10 Step 1: every dependency commit is an ancestor of the commit under test, and the symbols and tests exist there.
Usage: check_deps.py REPO HEAD DEPS_FILE [--eager]   (DEPS_FILE: one `NAME SHA` per line, written from lane C's posts)
--eager (RD10_CAPTURE=eager, the labelled EAGER PRE-RUN): the 17 names OTHER than RD-11a are required; RD-11a is reported as present or
absent and is never required. Any other pending name still STOPs. Without --eager all 18 are required.
Exit 0 = all present, 2 = STOP. Reads git only (cat-file, merge-base, grep, ls-tree); changes nothing.
"""
import subprocess
import sys
from pathlib import Path

REQUIRED = ["RD-0b", "RD-5", "RD-9a", "RD-9b", "RD-9c", "RD-9d", "RD-4a", "RD-4b", "RD-4c", "RD-4d", "RD-4e",
            "RD-7a", "RD-7b", "RD-8a", "RD-8b", "RD-2d", "RD-11a", "RD-2e"]
SPLITS = ["RD-9b1", "RD-9b2", "RD-9b3a", "RD-9b3b"]      # RD-9b has no merge commit titled that: it landed as four split merges
SYMBOLS = [("QwenResidentStep", "engine/src"), ("INFERENCE_CUDA_SYNC_EACH", "engine/src")]
FILTER = "projection_repeat_"


def git(repo, *args):
    try:
        p = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as e:
        print("STOP: git failed:", e)
        sys.exit(2)
    return p.returncode, p.stdout


def split_merges(repo, head, subjects):
    """RD-9b has no merge commit titled that; the work landed as four split merges whose subjects name the split
    (the exact subjects come from `git log --first-parent` on task/cuda-p5). The name RD-9b is satisfied when every
    split lands as a MERGE commit (two parents: 16c F5 — a plain commit whose subject mentions the split does not
    count) that is an ancestor of HEAD; a missing, unmerged or non-merge split is a STOP that names it."""
    bad = []
    for split in SPLITS:
        hits = [h for h, s in subjects if split in s]
        merges = []
        sha = ""
        for cand in hits:
            if len(git(repo, "rev-list", "--parents", "-n", "1", cand)[1].split()) == 3:
                merges.append(cand)
                if git(repo, "merge-base", "--is-ancestor", cand, head)[0] == 0:
                    sha = cand
                    break
        print(f"{'RD-9b':7} {split:8} {sha[:12]:12} {'ancestor' if sha else 'NOT AN ANCESTOR'}")
        if not sha:
            if not hits:
                why = "no commit with that name in its subject exists"
            elif not merges:
                why = f"{hits[0][:12]} is not a merge commit: a split lands as a merge (two parents)"
            else:
                why = f"{merges[0][:12]} is not an ancestor of {head}"
            bad.append(f"RD-9b: split merge {split}: {why}")
    return bad


def check(repo, head, deps_path, eager=False):
    bad = []
    rc, _ = git(repo, "cat-file", "-e", f"{head}^{{commit}}")
    if rc != 0:
        return [f"HEAD {head} is not a commit"]
    try:
        deps = dict(ln.split(None, 1) for ln in Path(deps_path).read_text().splitlines() if ln.strip())      # `RD-11a pending merge` is a name and a note
    except (OSError, ValueError) as e:
        return [f"cannot read deps file: {e}"]
    required = [n for n in REQUIRED if not (eager and n == "RD-11a")]
    if eager:
        first = deps.get("RD-11a", "").split()[:1]
        if not first or first[0] == "pending":
            print("RD-11a (not required in the eager pre-run): absent")
        else:
            rc, _ = git(repo, "merge-base", "--is-ancestor", first[0], head)
            print("RD-11a (not required in the eager pre-run):", "present, an ancestor of HEAD" if rc == 0 else f"listed as {first[0]} but NOT an ancestor of HEAD (not required here; the captured run will STOP on it)")
    for name in required:
        if name == "RD-9b":
            rc, out = git(repo, "log", "--all", "--format=%H%x09%s")
            bad += split_merges(repo, head, [ln.split("\t", 1) for ln in out.splitlines() if "\t" in ln])
            continue
        sha = deps.get(name, "").split()[0] if deps.get(name, "").split() else ""
        if not sha:
            bad.append(f"{name}: no commit named in the deps file")
            continue
        rc, _ = git(repo, "merge-base", "--is-ancestor", sha, head)
        print(f"{name:7} {sha[:12]:12} {'ancestor' if rc == 0 else 'NOT AN ANCESTOR'}")
        if rc != 0:
            bad.append(f"{name}: {sha} is not an ancestor of {head}")
    for sym, path in SYMBOLS:
        rc, out = git(repo, "grep", "-l", sym, head, "--", path)
        if rc != 0:
            bad.append(f"symbol {sym} not found under {path} at {head}")
    rc, out = git(repo, "ls-tree", "-r", "--name-only", head)
    repeats = [ln for ln in out.splitlines() if ln.endswith("tests/repeat.rs") and "kernels/cuda" in ln]
    print("RD-2e files:", repeats)
    if len(repeats) != 2:
        bad.append(f"{len(repeats)} tests/repeat.rs files under kernels/cuda, expected 2 (matmul and matmul_expert)")
    for f in repeats:
        rc, _ = git(repo, "grep", "-q", FILTER, head, "--", f)
        if rc != 0:
            bad.append(f"{f}: no test named {FILTER}*")
    return bad


def main(argv):
    if len([a for a in argv if a != "--eager"]) != 4:
        print("STOP: usage: check_deps.py REPO HEAD DEPS_FILE")
        return 2
    bad = check(*[a for a in argv[1:] if a != "--eager"], eager="--eager" in argv)
    for b in bad:
        print("STOP:", b)
    return 2 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
