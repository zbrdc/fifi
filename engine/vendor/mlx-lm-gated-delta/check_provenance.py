#!/usr/bin/env python3
"""Checks the vendored mlx-lm gated delta files against provenance.json.

Usage: python3 -I engine/vendor/mlx-lm-gated-delta/check_provenance.py

Checks, in this order:
1. Every file in "files" has equal "upstream_sha256" and "vendored_sha256" (the import is
   byte-identical), and the file on disk has that hash.
2. "patches" is empty. A patch needs its own chain check, so adding one means extending this script
   in the same commit (see engine/vendor/xavier-qwen/check_patches.py for the pattern).
3. Every file in "derived" has the recorded SHA-256 on disk and equals what extract_kernel.py
   extracts from the vendored Python file.
"""
import hashlib
import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fail(message):
    print("FAIL " + message)
    return 1


def main():
    provenance = json.loads((HERE / "provenance.json").read_text())
    for name, entry in provenance["files"].items():
        if entry["upstream_sha256"] != entry["vendored_sha256"]:
            return fail(f"{name}: the import is not byte-identical")
        if sha256(HERE / name) != entry["vendored_sha256"]:
            return fail(f"{name}: file on disk does not match vendored_sha256")
    if provenance["patches"]:
        return fail("patches are not supported by this checker")
    print("ok hashes")
    for name, entry in provenance["derived"].items():
        if sha256(HERE / name) != entry["sha256"]:
            return fail(f"{name}: file on disk does not match the recorded sha256")
    done = subprocess.run(
        [sys.executable, "-I", str(HERE / "extract_kernel.py"), "--check"],
        capture_output=True, text=True,
    )
    print(done.stdout.strip())
    return done.returncode


if __name__ == "__main__":
    raise SystemExit(main())
