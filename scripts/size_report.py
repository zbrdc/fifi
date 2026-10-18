#!/usr/bin/env python3
"""Report nonblank line changes for staged or committed Git content."""

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile


CATEGORY_NAMES = (
    "production", "tests", "qualification", "tooling", "docs", "generated",
    "vendor", "fixtures", "lookup_tables", "unclassified",
)
CLASSIFICATION_LIMITS = [
    "Classification uses repository paths and file names. It does not parse source code.",
    "Inline tests stay with their containing file, so inline Rust tests under engine/src count as production.",
    "Renamed files use the destination path for category assignment.",
    "Binary files and submodules are reported but excluded from nonblank line totals.",
    "Binary classification and patch counts use attributes from the selected comparison snapshot.",
]


class ReportError(Exception):
    """A Git or input error that the command can report without a traceback."""


def git(args, check=True, cwd=None, input_data=None, env=None):
    try:
        result = subprocess.run(
            ["git", *args], cwd=cwd, input=input_data, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except OSError as error:
        raise ReportError("cannot run Git: %s" % error) from error
    if check and result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise ReportError(detail or "Git exited with status %d" % result.returncode)
    return result


def resolve_commit(revision):
    result = git(["rev-parse", "--verify", "--end-of-options",
                  revision + "^{commit}"])
    return result.stdout.decode("ascii").strip()


def tree_revision(revision):
    result = git(["rev-parse", "--verify", revision + "^{tree}"])
    return result.stdout.decode("ascii").strip()


def empty_tree_revision():
    result = git(["hash-object", "-w", "-t", "tree", "--stdin"],
                 input_data=b"")
    return result.stdout.decode("ascii").strip()


def staged_tree_revision():
    result = git(["write-tree"])
    return result.stdout.decode("ascii").strip()


def snapshot_diff_context(temp_root, object_dir, object_format, attribute_source):
    template = temp_root / "empty-template"
    isolated_repository = temp_root / "repository"
    worktree = temp_root / "worktree"
    template.mkdir()
    worktree.mkdir()

    init_env = os.environ.copy()
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR",
                 "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES",
                 "GIT_PREFIX", "GIT_ATTR_SOURCE"):
        init_env.pop(name, None)
    git(["init", "--quiet", "--object-format=" + object_format,
         "--template=" + str(template), str(isolated_repository)],
        cwd=temp_root, env=init_env)

    isolated_git_dir = isolated_repository / ".git"
    alternates = isolated_git_dir / "objects" / "info" / "alternates"
    alternates.parent.mkdir(parents=True, exist_ok=True)
    alternates.write_bytes(os.fsencode(object_dir) + b"\n")

    diff_env = os.environ.copy()
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR",
                 "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES",
                 "GIT_PREFIX", "GIT_ATTR_SOURCE"):
        diff_env.pop(name, None)
    diff_env["GIT_DIR"] = str(isolated_git_dir)
    diff_env["GIT_WORK_TREE"] = str(worktree)
    diff_env["GIT_INDEX_FILE"] = str(isolated_git_dir / "index")
    diff_env["GIT_ATTR_NOSYSTEM"] = "1"
    git(["read-tree", attribute_source], cwd=worktree, env=diff_env)
    return worktree, diff_env


def staged_base_revision():
    result = git(["rev-parse", "--verify", "--quiet", "--end-of-options",
                  "HEAD^{commit}"], check=False)
    if result.returncode == 0:
        return result.stdout.decode("ascii").strip()

    symbolic = git(["symbolic-ref", "--quiet", "HEAD"], check=False)
    status = git(["status", "--porcelain", "--untracked-files=no"], check=False)
    if symbolic.returncode == 0 and status.returncode == 0:
        return None
    detail = result.stderr.decode("utf-8", "replace").strip()
    raise ReportError(detail or "cannot resolve HEAD")


def parse_raw_diff(data):
    tokens = data.split(b"\0")
    entries = []
    index = 0
    while index < len(tokens) - 1:
        header = tokens[index]
        index += 1
        if not header:
            continue
        fields = header[1:].split()
        if len(fields) != 5:
            raise ReportError("Git returned an invalid raw diff record")
        old_mode, new_mode, _, _, raw_status = fields
        status_code = raw_status[:1].decode("ascii")
        if status_code in ("R", "C"):
            if index + 1 >= len(tokens):
                raise ReportError("Git returned an incomplete rename record")
            old_path, new_path = tokens[index:index + 2]
            index += 2
        else:
            if index >= len(tokens) - 1:
                raise ReportError("Git returned an incomplete path record")
            path = tokens[index]
            index += 1
            old_path = None if status_code == "A" else path
            new_path = None if status_code == "D" else path
        entries.append({
            "old_mode": old_mode,
            "new_mode": new_mode,
            "status_code": status_code,
            "old_path": os.fsdecode(old_path) if old_path is not None else None,
            "new_path": os.fsdecode(new_path) if new_path is not None else None,
            "added": 0,
            "removed": 0,
            "binary": False,
        })
    return entries


def apply_snapshot_binary_attributes(entries, source, cwd, env):
    paths = sorted({entry["new_path"] or entry["old_path"] for entry in entries})
    if not paths:
        return
    input_data = b"\0".join(os.fsencode(path) for path in paths) + b"\0"
    result = git(
        ["-c", "core.attributesFile=" + os.devnull, "check-attr",
         "--source=" + source, "--stdin", "-z", "diff"],
        cwd=cwd, input_data=input_data, env=env)
    fields = result.stdout.split(b"\0")
    if (len(fields) - 1) % 3:
        raise ReportError("Git returned invalid attribute records")
    binary_paths = set()
    for index in range(0, len(fields) - 1, 3):
        path, attribute, value = fields[index:index + 3]
        if attribute == b"diff" and value == b"unset":
            binary_paths.add(os.fsdecode(path))
    for entry in entries:
        path = entry["new_path"] or entry["old_path"]
        if path in binary_paths:
            entry["binary"] = True


def count_patch_lines(args, entries, cwd, env, attribute_source):
    command = ["git", "--attr-source=" + attribute_source,
               "-c", "core.attributesFile=" + os.devnull, *args,
               "--no-ext-diff", "--no-textconv", "--no-color", "--no-relative",
               "--submodule=short", "--unified=0", "--find-renames", "--"]
    try:
        process = subprocess.Popen(command, cwd=cwd, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError as error:
        raise ReportError("cannot run Git diff: %s" % error) from error

    current = -1
    sections_remaining = 0
    in_hunk = False
    unexpected_section = False
    for line in process.stdout:
        if line.startswith(b"diff --git "):
            if sections_remaining == 0:
                current += 1
                if current >= len(entries):
                    unexpected_section = True
                else:
                    sections_remaining = 2 if entries[current]["status_code"] == "T" else 1
            if current < len(entries):
                sections_remaining -= 1
            in_hunk = False
            continue
        if current < 0 or current >= len(entries):
            continue
        entry = entries[current]
        if line.startswith((b"Binary files ", b"GIT binary patch")):
            entry["binary"] = True
        elif line.startswith(b"@@"):
            in_hunk = True
        elif line.startswith(b"\\ No newline at end of file"):
            continue
        elif in_hunk and (line.startswith(b"+") or line.startswith(b"-")):
            try:
                text = line[1:].decode("utf-8")
            except UnicodeDecodeError:
                entry["binary"] = True
                continue
            if text.strip():
                key = "added" if line.startswith(b"+") else "removed"
                entry[key] += 1

    stderr = process.stderr.read()
    return_code = process.wait()
    if return_code != 0:
        detail = stderr.decode("utf-8", "replace").strip()
        raise ReportError(detail or "Git diff exited with status %d" % return_code)
    if unexpected_section or sections_remaining != 0 or current + 1 != len(entries):
        raise ReportError("Git returned inconsistent raw and patch diffs")


def classify(path):
    lowered = path.lower()
    parts = lowered.split("/")
    name = parts[-1]

    if any(part in ("vendor", "vendored") for part in parts):
        return "vendor"
    if "generated" in parts or re.search(r"\.generated\.[^.]+(?:\.[^.]+)*$", name):
        return "generated"
    if any(part in ("fixture", "fixtures", "testdata", "test_data")
           for part in parts):
        return "fixtures"
    if parts[0] in ("docs", "api") or name.endswith((".md", ".rst", ".adoc")):
        return "docs"
    if any(part in ("test", "tests") for part in parts):
        return "tests"
    if name.startswith("test_") or re.search(r"(?:^|[._])tests?\.[^.]+$", name):
        return "tests"
    if re.search(r"(?:^|[_-])(?:lookup|tables?)(?:[_-]|\.)", name):
        return "lookup_tables"
    if parts[0] == "qualification":
        return "qualification"
    if parts[0] in ("scripts", ".githooks"):
        return "tooling"
    if parts[0] == ".github" and len(parts) > 1 and parts[1] == "workflows":
        return "tooling"
    if lowered.startswith(("engine/src/", "src/")):
        return "production"
    return "unclassified"


def public_status(code):
    return {
        "A": "added", "D": "deleted", "M": "modified", "R": "renamed",
        "C": "copied", "T": "type_changed",
    }.get(code, "changed")


def empty_counts():
    return {
        "added_nonblank": 0,
        "removed_nonblank": 0,
        "changed_files": 0,
        "binary_files": 0,
        "symlink_files": 0,
        "submodule_files": 0,
    }


def build_report(basis, entries):
    files = []
    categories = {name: empty_counts() for name in CATEGORY_NAMES}
    totals = empty_counts()
    unclassified = []

    for entry in entries:
        path = entry["new_path"] or entry["old_path"]
        category = classify(path)
        if category == "unclassified":
            unclassified.append(path)

        if entry["old_mode"] == b"160000" or entry["new_mode"] == b"160000":
            kind = "submodule"
        elif entry["old_mode"] == b"120000" or entry["new_mode"] == b"120000":
            kind = "symlink"
        elif entry["binary"]:
            kind = "binary"
        else:
            kind = "text"

        countable = kind != "binary" and kind != "submodule"
        added = entry["added"] if countable else None
        removed = entry["removed"] if countable else None
        file_record = {
            "path": path,
            "previous_path": entry["old_path"] if entry["status_code"] == "R" else None,
            "status": public_status(entry["status_code"]),
            "category": category,
            "kind": kind,
            "added_nonblank": added,
            "removed_nonblank": removed,
        }
        files.append(file_record)

        for counts in (totals, categories[category]):
            counts["changed_files"] += 1
            if added is not None:
                counts["added_nonblank"] += added
                counts["removed_nonblank"] += removed
            if kind == "binary":
                counts["binary_files"] += 1
            elif kind == "symlink":
                counts["symlink_files"] += 1
            elif kind == "submodule":
                counts["submodule_files"] += 1

    files.sort(key=lambda item: (item["path"], item["previous_path"] or ""))
    return {
        "schema_version": 1,
        "basis": basis,
        "totals": totals,
        "categories": categories,
        "files": files,
        "unclassified_paths": sorted(unclassified),
        "classification_limits": CLASSIFICATION_LIMITS,
    }


def make_parser():
    parser = argparse.ArgumentParser(
        description="Report nonblank line changes for staged or committed Git content.")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--staged", action="store_true",
                       help="compare HEAD with the index")
    modes.add_argument("--base", metavar="REV", help="base commit for a comparison")
    parser.add_argument("--head", metavar="REV", help="head commit for a comparison")
    parser.add_argument("--json", action="store_true", help="write schema-versioned JSON")
    return parser


def main(argv=None):
    parser = make_parser()
    options = parser.parse_args(argv)
    if not options.json:
        parser.error("--json is required")
    if options.staged:
        if options.head is not None:
            parser.error("--head applies only with --base")
        base_revision = staged_base_revision()
        base_tree = (empty_tree_revision() if base_revision is None
                     else tree_revision(base_revision))
        attribute_source = staged_tree_revision()
        diff_args = ["diff", base_tree, attribute_source]
        basis = {
            "type": "staged",
            "base_revision": base_revision,
            "head_revision": "index",
        }
    else:
        if options.base is None or options.head is None:
            parser.error("choose --staged or provide both --base and --head")
        base_revision = resolve_commit(options.base)
        head_revision = resolve_commit(options.head)
        attribute_source = tree_revision(head_revision)
        diff_args = ["diff", base_revision, head_revision]
        basis = {
            "type": "commits",
            "base_revision": base_revision,
            "head_revision": head_revision,
        }

    object_dir = os.fsdecode(git([
        "rev-parse", "--path-format=absolute", "--git-path", "objects",
    ]).stdout.rstrip(b"\n"))
    object_format = git(["rev-parse", "--show-object-format"]).stdout.decode("ascii").strip()
    with tempfile.TemporaryDirectory(prefix="size-report-attributes-") as temp_root:
        clean_worktree, diff_env = snapshot_diff_context(
            pathlib.Path(temp_root), object_dir, object_format, attribute_source)
        raw_args = ["--attr-source=" + attribute_source,
                    "-c", "core.attributesFile=" + os.devnull, *diff_args,
                    "--no-relative", "--submodule=short", "--raw",
                    "--no-abbrev", "-z", "--find-renames"]
        raw = git(raw_args, cwd=clean_worktree, env=diff_env)
        entries = parse_raw_diff(raw.stdout)
        apply_snapshot_binary_attributes(entries, attribute_source,
                                         clean_worktree, diff_env)
        count_patch_lines(diff_args, entries, clean_worktree, diff_env,
                          attribute_source)
    print(json.dumps(build_report(basis, entries), indent=2, sort_keys=True))
    return 0


def entry_point():
    try:
        return main()
    except ReportError as error:
        print("size_report.py: error: %s" % error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(entry_point())
