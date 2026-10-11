#!/usr/bin/env python3
"""Replays the patch chain of provenance.json over the vendored xavier-qwen files.

Usage: python3 -I engine/vendor/xavier-qwen/check_patches.py

Needs GNU patch on PATH (Linux). Checks, in this order:
1. Every file in "files" lists a "patched_sha256" equal to the last "after_sha256" in "patches"
   and to the SHA-256 of the file on disk.
2. The chain links: the "before_sha256" of a patch equals the "after_sha256" of the patch
   before it, and the first "before_sha256" equals "vendored_sha256". A file that a patch
   creates has "before_sha256": null in that patch and no "vendored_sha256" in "files".
3. Every patch with a "diff" file reverses cleanly from the on-disk files to its
   "before_sha256" and replays forward to its "after_sha256", byte for byte. A patch with
   "diff": null has no stored diff; its end points are checked by hash only.
"""
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_or_none(path):
    return sha256(path) if path.exists() else None


def fail(message):
    print("FAIL " + message)
    return 1


def check_hashes(provenance):
    patches = provenance["patches"]
    for path, entry in provenance["files"].items():
        steps = [p["files"][path] for p in patches if path in p["files"]]
        if not steps:
            continue
        if steps[0]["before_sha256"] != entry.get("vendored_sha256"):
            return fail(f"{path}: first before_sha256 is not vendored_sha256")
        for earlier, later in zip(steps, steps[1:]):
            if earlier["after_sha256"] != later["before_sha256"]:
                return fail(f"{path}: patch chain is broken")
        if steps[-1]["after_sha256"] != entry.get("patched_sha256"):
            return fail(f"{path}: patched_sha256 is not the last after_sha256")
        if sha256(HERE / path) != entry["patched_sha256"]:
            return fail(f"{path}: file on disk does not match patched_sha256")
    print("ok hashes")
    return 0


def run_patch(work, diff, reverse):
    command = ["patch", "-p1", "--no-backup-if-mismatch", "-s"]
    if reverse:
        command.extend(["-R", "-E"])  # -E removes a file that the reversed patch created
    with open(diff, "rb") as stream:
        return subprocess.run(command, cwd=work, stdin=stream, capture_output=True, text=True)


def check_replay(provenance):
    patches = provenance["patches"]
    with tempfile.TemporaryDirectory() as tmp:
        work = pathlib.Path(tmp)
        for path in provenance["files"]:
            (work / path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(HERE / path, work / path)
        stored = [p for p in patches if p.get("diff")]
        for patch in reversed(stored):
            diff = HERE / patch["diff"]
            done = run_patch(work, diff, reverse=True)
            if done.returncode != 0:
                return fail(f"{patch['id']}: reverse apply failed: {done.stderr.strip()}")
            for path, step in patch["files"].items():
                if sha256_or_none(work / path) != step["before_sha256"]:
                    return fail(f"{patch['id']}: {path} after reverse is not before_sha256")
        for patch in stored:
            diff = HERE / patch["diff"]
            done = run_patch(work, diff, reverse=False)
            if done.returncode != 0:
                return fail(f"{patch['id']}: forward apply failed: {done.stderr.strip()}")
            for path, step in patch["files"].items():
                if sha256_or_none(work / path) != step["after_sha256"]:
                    return fail(f"{patch['id']}: {path} after replay is not after_sha256")
        for path in provenance["files"]:
            if (work / path).read_bytes() != (HERE / path).read_bytes():
                return fail(f"{path}: replayed bytes differ from the vendored file")
    print("ok replay: " + ", ".join(p["id"] for p in stored))
    return 0


def main():
    provenance = json.loads((HERE / "provenance.json").read_text())
    return check_hashes(provenance) or check_replay(provenance)


if __name__ == "__main__":
    raise SystemExit(main())
