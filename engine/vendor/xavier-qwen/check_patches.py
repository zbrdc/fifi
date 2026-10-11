#!/usr/bin/env python3
"""Replays the patch chain of provenance.json over a vendored tree and checks the tree against it.

Usage: python3 -I engine/vendor/xavier-qwen/check_patches.py [<vendor-dir>]

No argument: the directory of this script (xavier-qwen). With an argument, that vendor directory
(for example engine/vendor/ggml-cuda-mmq); the script is never copied. Needs GNU patch on PATH.
provenance.json keys read: "files" {path: {upstream_path, upstream_sha256, vendored_sha256,
patched_sha256}}, "patches" [{id, diff, files {path: {before_sha256, after_sha256}}}], and, both
optional, "include_roots" (default ["src"]) and "shims" [{include, stands_in_for, path, sha256}].
Hashes are lowercase hex. Listed paths are plain ("src/a.h": no "./", no "//", no "..", not absolute).
"patches" is required (it may be empty). A repeated JSON key, or a malformed file, is a FAIL line.
Checks, in this order (3b first):
1. "LICENSE" is listed. Every file in "files" exists and its SHA-256 on disk equals "patched_sha256", or
   "vendored_sha256" when no patch touches it (patched or not). A "patched_sha256" needs a patch.
   The chain links: the "before_sha256" of a patch equals the "after_sha256" of the patch
   before it, and the first "before_sha256" equals "vendored_sha256". A file that a patch
   creates has "before_sha256": null in that patch and no "vendored_sha256" in "files".
   Two listed paths that differ only in case fail.
2. The tree equals the list: the files on disk are the listed files, provenance.json, this script
   (only when it sits in the vendor directory) and the stored diffs; none extra, none missing.
3. Every quoted #include in a listed source file resolves (its own directory, then "include_roots")
   to a listed file, or is the "include" name of a shim. The scan is textual and sees only lines that
   start with #include (after blanks), in files with the suffixes of SOURCE_SUFFIXES. It does not see
   `/* c */ #include "x"`, a backslash-continued include, `#include MACRO`, `<x>`, or a file with another
   suffix (.inl, .hh, .tpp); it ignores #if and comments. CPF-V1 shows the 28 imported files have none of
   these (closure.py prints it). Every listed shim must be included by at least one file.
4. Every shim stands in for a file that is not vendored. "sha256" of its file at "path" (relative
   to the vendor directory) is the file's SHA-256 on disk, or null while the file is not written.
3b. "trim" (only a block that has "removed_files"; xavier-qwen's differently shaped "trim" is ignored): every removed file is
   listed as {upstream_path, upstream_sha256 (64 lowercase hex)}, is not in "files" and is absent from the tree; upstream_path
   is a plain path that ends with the removed path minus its first component (src/a/b.h <- ggml/src/a/b.h); "entries" is a list
   of {id, files} with at least one file each; every path an entry names is in "removed_files" and every removed path is named
   by exactly one entry. "import_file_count" (optional top-level key, written by CPF-V2) must equal the number of listed files
   that have an upstream_sha256 plus the number of removed files, so a deleted trim block or a dropped entry fails.
   LIMIT, accepted by lane C (KUBDAN-MSG-43 item 3): an offline check cannot prove that a well-formed hash equals the pin's, or that
   a plain upstream_path with the right tail names the right file; CPF-V1's import-identity evidence and the packet-side
   verify_trim.py (which reads the pin) cover that. Deleting both "import_file_count" and the trim block is also not seen.
5. Every patch with a "diff" file reverses cleanly from the on-disk files to its
   "before_sha256" and replays forward to its "after_sha256", byte for byte. After the reverse pass every
   listed file is in the state its first stored patch records (so a stored diff cannot edit a file that
   its patch entry does not declare). A patch with
   "diff": null has no stored diff; its end points are checked by hash only.
"""
import hashlib
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

SCRIPT = pathlib.Path(__file__).resolve()
INCLUDE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.M)
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp"}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_or_none(path):
    return sha256(path) if path.is_file() else None


def fail(message):
    print("FAIL " + message)
    return 1


def check_chain(path, entry, steps):
    if not steps:
        if entry.get("patched_sha256", entry.get("vendored_sha256")) != entry.get("vendored_sha256"):
            return fail(f"{path}: patched_sha256 but no patch lists the file")
        return 0
    if steps[0]["before_sha256"] != entry.get("vendored_sha256"):
        return fail(f"{path}: first before_sha256 is not vendored_sha256")
    for earlier, later in zip(steps, steps[1:]):
        if earlier["after_sha256"] != later["before_sha256"]:
            return fail(f"{path}: patch chain is broken")
    if steps[-1]["after_sha256"] != entry.get("patched_sha256"):
        return fail(f"{path}: patched_sha256 is not the last after_sha256")
    return 0


def check_listing(provenance):
    """The list itself: LICENSE, case collisions, patches that list unlisted files, paths that escape."""
    if "LICENSE" not in provenance["files"]:
        return fail("LICENSE is not listed in files")
    lowered = [path.lower() for path in provenance["files"]]
    if len(set(lowered)) != len(lowered):
        return fail("two listed paths differ only in case")
    for patch in provenance["patches"]:
        unlisted = [p for p in patch["files"] if p not in provenance["files"]]
        if unlisted:
            return fail(f"{patch['id']}: patches {unlisted[0]}, which \"files\" does not list")
    for path in provenance["files"]:
        if pathlib.PurePosixPath(path).is_absolute() or ".." in pathlib.PurePosixPath(path).parts:
            return fail(f"{path}: listed path leaves the vendor directory")
    return 0


def check_file(root, path, entry, patches):
    steps = [p["files"][path] for p in patches if path in p["files"]]
    problem = check_chain(path, entry, steps)
    if problem:
        return problem
    key = "patched_sha256" if "patched_sha256" in entry else "vendored_sha256"
    if key not in entry:
        return fail(f"{path}: neither patched_sha256 nor vendored_sha256")
    if not (root / path).is_file():
        return fail(f"{path}: listed file is missing")
    if sha256(root / path) != entry[key]:
        return fail(f"{path}: file on disk does not match {key}")
    return 0


def check_hashes(root, provenance):
    problem = check_listing(provenance)
    for path, entry in provenance["files"].items():
        problem = problem or check_file(root, path, entry, provenance["patches"])
    if not problem:
        print("ok hashes")
    return problem


def check_tree(root, provenance):
    allowed = set(provenance["files"]) | {"provenance.json"}
    allowed |= {p["diff"] for p in provenance["patches"] if p.get("diff")}
    if SCRIPT.parent == root:
        allowed.add(SCRIPT.name)
    paths = [p for p in root.rglob("*") if p.is_file() or p.is_symlink()]
    if any(p.is_symlink() for p in paths):
        return fail("the vendor tree holds a symbolic link")
    found = {p.relative_to(root).as_posix() for p in paths}
    if found != allowed:
        parts = [f"{label} {', '.join(sorted(names))}"
                 for label, names in (("extra", found - allowed), ("missing", allowed - found)) if names]
        return fail("tree differs from provenance.json: " + "; ".join(parts))
    print(f"ok tree: {len(found)} files")
    return 0


HEX64 = re.compile(r"[0-9a-f]{64}")


def plain_path(path):
    if not isinstance(path, str) or path == "":
        return False
    pure = pathlib.PurePosixPath(path)
    return not pure.is_absolute() and ".." not in pure.parts


def check_removed(root, provenance, path, info):
    if not plain_path(path):
        return fail(f"trim: {path}: removed path leaves the vendor directory")
    if path in provenance["files"]:
        return fail(f"trim: {path} is removed and also listed in files")
    if (root / path).exists() or (root / path).is_symlink():
        return fail(f"trim: {path} is removed but still in the tree")
    if not isinstance(info, dict) or not plain_path(info.get("upstream_path")):
        return fail(f"trim: {path}: upstream_path is missing or not a plain path")
    if not info["upstream_path"].endswith(path.split("/", 1)[-1]):
        return fail(f"trim: {path}: upstream_path {info['upstream_path']} does not end with the removed path")
    if not isinstance(info.get("upstream_sha256"), str) or not HEX64.fullmatch(info["upstream_sha256"]):
        return fail(f"trim: {path}: upstream_sha256 is not 64 lowercase hex")
    return 0


def check_trim_entries(trim):
    removed, entries, seen = trim["removed_files"], trim.get("entries"), []
    if not isinstance(entries, list) or not entries:
        return fail("trim: entries is missing or empty")
    for entry in entries:
        if not isinstance(entry.get("id"), str) or not isinstance(entry.get("files"), list):
            return fail("trim: an entry has no id or no files list")
        if not entry["files"]:
            return fail(f"trim: entry {entry['id']} lists no file")
        for path in entry["files"]:
            if path not in removed:
                return fail(f"trim: entry {entry['id']} names {path}, which is not in removed_files")
        seen.extend(entry["files"])
    uncovered = sorted(set(removed) - set(seen))
    if uncovered or len(seen) != len(set(seen)):
        return fail("trim: " + (f"no entry lists {uncovered[0]}" if uncovered else "a path is named by two entries"))
    return 0


def check_import_count(provenance, removed):
    expected = provenance.get("import_file_count")
    if expected is None:
        return 0
    pinned = sum(1 for e in provenance["files"].values() if "upstream_sha256" in e)
    if not isinstance(expected, int) or pinned + len(removed) != expected:
        return fail(f"trim: import_file_count is {expected} but {pinned} listed files with an upstream plus "
                    f"{len(removed)} removed files make {pinned + len(removed)}")
    return 0


def check_trim(root, provenance):
    trim = provenance.get("trim")
    if not isinstance(trim, dict) or "removed_files" not in trim:
        return check_import_count(provenance, {})
    if not isinstance(trim["removed_files"], dict):
        return fail("trim: removed_files is not an object")
    upstream = [e.get("upstream_path") for e in provenance["files"].values()]
    for path, info in trim["removed_files"].items():
        problem = check_removed(root, provenance, path, info)
        if problem:
            return problem
        if isinstance(info, dict) and info["upstream_path"] in upstream:
            return fail(f"trim: {path}: upstream_path is also the upstream of a listed file")
    problem = check_trim_entries(trim) or check_import_count(provenance, trim["removed_files"])
    if not problem:
        print(f"ok trim: {len(trim['removed_files'])} removed files")
    return problem


def resolve_include(root, includer, name, provenance):
    bases = [includer.parent] + [root / r for r in provenance.get("include_roots", ["src"])]
    for base in bases:
        candidate = (base / name).resolve()
        if candidate.is_file() and root in candidate.parents:
            return candidate.relative_to(root).as_posix()
    return None


def check_includes(root, provenance):
    names = {s["include"] for s in provenance.get("shims", [])}
    count = shimmed = 0
    used = set()
    for path in provenance["files"]:
        if pathlib.PurePosixPath(path).suffix not in SOURCE_SUFFIXES:
            continue
        for name in INCLUDE.findall((root / path).read_text(errors="replace")):
            target = resolve_include(root, root / path, name, provenance)
            count += 1
            if target in provenance["files"] and name in names:
                return fail(f"{path}: shim \"{name}\" shadows the vendored file {target}")
            if target in provenance["files"]:
                continue
            if name not in names:
                return fail(f"{path}: #include \"{name}\" is neither a vendored file nor a listed shim")
            shimmed += 1
            used.add(name)
    if names - used:
        return fail("shim " + ", ".join(sorted(names - used)) + " is included by no listed file")
    print(f"ok includes: {count} quoted includes, {shimmed} through shims")
    return 0


def check_shims(root, provenance):
    vendored = {e["upstream_path"] for e in provenance["files"].values() if "upstream_path" in e}
    shims = provenance.get("shims", [])
    if len({s["include"] for s in shims}) != len(shims):
        return fail("two shims have the same include name")
    for shim in shims:
        if shim["stands_in_for"] in vendored:
            return fail(f"shim {shim['include']}: {shim['stands_in_for']} is vendored, not shimmed")
        if sha256_or_none(root / shim["path"]) != shim["sha256"]:
            return fail(f"shim {shim['include']}: file at {shim['path']} does not match sha256")
    unwritten = sum(1 for s in shims if s["sha256"] is None)
    print(f"ok shims: {len(shims)} listed, {unwritten} not written yet")
    return 0


def run_patch(work, diff, reverse):
    command = ["patch", "-p1", "--no-backup-if-mismatch", "-s"]
    if reverse:
        command.extend(["-R", "-E"])  # -E removes a file that the reversed patch created
    with open(diff, "rb") as stream:
        return subprocess.run(command, cwd=work, stdin=stream, capture_output=True, text=True)


def check_reversed(work, provenance):
    """After the reverse pass each file is in the state its first stored patch records."""
    for path, entry in provenance["files"].items():
        expected = entry.get("patched_sha256", entry.get("vendored_sha256"))
        for patch in provenance["patches"]:
            if patch.get("diff") and path in patch["files"]:
                expected = patch["files"][path]["before_sha256"]
                break
        if sha256_or_none(work / path) != expected:
            return fail(f"{path}: after the reverse pass the file is not in the recorded state; "
                        "a stored diff edits a file that its patch entry does not declare")
    return 0


def check_replay(root, provenance):
    patches = provenance["patches"]
    with tempfile.TemporaryDirectory() as tmp:
        work = pathlib.Path(tmp)
        for path in provenance["files"]:
            (work / path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / path, work / path)
        stored = [p for p in patches if p.get("diff")]
        for patch in reversed(stored):
            diff = root / patch["diff"]
            done = run_patch(work, diff, reverse=True)
            if done.returncode != 0:
                return fail(f"{patch['id']}: reverse apply failed: {done.stderr.strip()}")
            for path, step in patch["files"].items():
                if sha256_or_none(work / path) != step["before_sha256"]:
                    return fail(f"{patch['id']}: {path} after reverse is not before_sha256")
        problem = check_reversed(work, provenance)
        if problem:
            return problem
        for patch in stored:
            diff = root / patch["diff"]
            done = run_patch(work, diff, reverse=False)
            if done.returncode != 0:
                return fail(f"{patch['id']}: forward apply failed: {done.stderr.strip()}")
            for path, step in patch["files"].items():
                if sha256_or_none(work / path) != step["after_sha256"]:
                    return fail(f"{patch['id']}: {path} after replay is not after_sha256")
        for path in provenance["files"]:
            if (work / path).read_bytes() != (root / path).read_bytes():
                return fail(f"{path}: replayed bytes differ from the vendored file")
    print("ok replay: " + (", ".join(p["id"] for p in stored) or "no stored diff"))
    return 0


def no_repeats(pairs):
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError("repeated JSON key in " + ", ".join(sorted({k for k in keys if keys.count(k) > 1})))
    return dict(pairs)


def main(argv):
    if len(argv) > 1:
        print(__doc__.split("Checks,")[0])
        return 2
    root = (pathlib.Path(argv[0]) if argv else SCRIPT.parent).resolve()
    if not (root / "provenance.json").is_file():
        return fail(f"{root}: no provenance.json")
    try:
        provenance = json.loads((root / "provenance.json").read_text(), object_pairs_hook=no_repeats)
        for check in (check_trim, check_hashes, check_tree, check_includes, check_shims, check_replay):
            problem = check(root, provenance)
            if problem:
                return problem
    except (KeyError, TypeError, AttributeError, ValueError) as error:
        return fail(f"provenance.json is malformed: {error!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
