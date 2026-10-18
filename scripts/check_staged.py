#!/usr/bin/env python3
"""Run fast offline checks against the staged snapshot, commit subject, and pushed refs."""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


COMMIT_SUBJECT = re.compile(
    r"(?P<type>feat|fix|perf|test|docs|style|chore|refactor)"
    r"\((?P<scope>[^()\r\n]*)\)(?P<breaking>!)?: (?P<subject>.*)"
)
GROWTH_FIELDS = {
    "schema_version": int,
    "basis": dict,
    "totals": dict,
    "categories": dict,
    "files": list,
    "unclassified_paths": list,
}


SECRET_PATTERNS = [
    re.compile(r"ghp_[A-Za-z0-9]{36}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{36,}"),
    re.compile(r"sk-[A-Za-z0-9]{32,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY(?: BLOCK)?-----"),
]

# Attribution lines are banned in commit messages (user rule): no line may credit a
# model, agent, harness, tool, or session.
ATTRIBUTION_LINE_START = re.compile(
    r"^\s*(?:co-authored-by:|claude-session:|worker:|generated-by:)",
    re.IGNORECASE,
)
ATTRIBUTION_INLINE = re.compile(r"\bgenerated with\b|\bvia pi\b", re.IGNORECASE)
ATTRIBUTION_TRAILER_KEY = re.compile(r"^\s*[A-Za-z][A-Za-z0-9-]*\s*:\s")
ATTRIBUTION_NAME = re.compile(r"\b(?:claude|anthropic|noreply@anthropic\.com)\b", re.IGNORECASE)


class CheckError(Exception):
    """A required fast check could not validate its input."""


def display_path(path: bytes) -> str:
    """Render Git path bytes without allowing control characters in diagnostics."""
    return os.fsdecode(path).encode("unicode_escape").decode("ascii")


def run_git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[bytes]:
    if shutil.which("git") is None:
        raise CheckError("required tool git was not found on PATH")
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as error:
        raise CheckError(f"could not run git {args[0]}: {error}") from error
    if result.returncode != 0:
        details = (result.stderr or result.stdout).decode("utf-8", errors="backslashreplace").strip()
        raise CheckError(
            f"git {' '.join(args)} failed with exit status {result.returncode}"
            + (f": {details}" if details else "")
        )
    return result


def repository_root() -> Path:
    result = run_git(["rev-parse", "--show-toplevel"], Path.cwd())
    raw_root = result.stdout.rstrip(b"\r\n")
    if not raw_root:
        raise CheckError("git rev-parse returned an empty repository path")
    return Path(os.fsdecode(raw_root))


def staged_python_blobs(root: Path) -> tuple[int, int]:
    """Parse changed regular Python blobs as recorded in the index."""
    whitespace = run_git(["diff", "--cached", "--check"], root)
    if whitespace.stdout or whitespace.stderr:
        details = (whitespace.stdout + whitespace.stderr).decode(
            "utf-8", errors="backslashreplace"
        ).strip()
        raise CheckError("staged whitespace check failed (git diff --cached --check):\n" + details)
    print("Staged whitespace check passed.")

    unmerged = run_git(["ls-files", "--unmerged", "-z"], root).stdout
    if unmerged:
        raise CheckError("the index contains unmerged paths; resolve them before committing")

    changed_output = run_git(
        [
            "diff",
            "--cached",
            "--name-only",
            "--find-renames",
            "--diff-filter=ACMRT",
            "-z",
        ],
        root,
    ).stdout
    changed_paths = [path for path in changed_output.split(b"\0") if path]
    if not changed_paths:
        return 0, 0

    index_output = run_git(["ls-files", "--stage", "-z"], root).stdout
    index_entries: dict[bytes, tuple[int, bytes]] = {}
    for record in index_output.split(b"\0"):
        if not record:
            continue
        try:
            metadata, path = record.split(b"\t", 1)
            mode_text, object_id, stage_text = metadata.split()
            mode = int(mode_text, 8)
            stage = int(stage_text)
        except (ValueError, TypeError) as error:
            raise CheckError("could not parse a Git index entry") from error
        if stage != 0:
            raise CheckError("the index contains unmerged paths; resolve them before committing")
        index_entries[path] = (mode, object_id)

    checked = 0
    symlinks = 0
    for path in changed_paths:
        if not path.endswith(b".py"):
            continue
        entry = index_entries.get(path)
        if entry is None:
            # Deleted paths have no stage-0 index entry and are excluded by the diff filter.
            continue
        mode, object_id = entry
        if mode == 0o120000:
            symlinks += 1
            print(f"Skipping Python-named symlink without following it: {display_path(path)}")
            continue
        if mode not in (0o100644, 0o100755):
            continue

        blob = run_git(["cat-file", "blob", object_id.decode("ascii")], root).stdout
        filename = display_path(path)
        try:
            ast.parse(blob, filename=filename)
        except (SyntaxError, UnicodeError, ValueError) as error:
            line = getattr(error, "lineno", None)
            location = f":{line}" if line is not None else ""
            message = getattr(error, "msg", str(error))
            raise CheckError(
                f"invalid Python syntax in staged file {filename}{location}: {message}"
            ) from error
        checked += 1

    print(f"Staged Python syntax check passed ({checked} regular file(s)).")
    return checked, symlinks


def validate_growth_report(raw: bytes) -> dict[str, object]:
    try:
        report = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CheckError(f"growth report schema failure: output is not valid JSON: {error}") from error
    if not isinstance(report, dict):
        raise CheckError("growth report schema failure: top-level JSON value must be an object")

    for field, expected_type in GROWTH_FIELDS.items():
        if field not in report:
            raise CheckError(f"growth report schema failure: missing {field!r}")
        value = report[field]
        if field == "schema_version":
            if type(value) is not int or value != 1:
                raise CheckError("growth report schema failure: schema_version must be integer 1")
        elif not isinstance(value, expected_type):
            raise CheckError(
                f"growth report schema failure: {field!r} must be {expected_type.__name__}"
            )

    basis = report["basis"]
    if (
        basis.get("type") != "staged"
        or basis.get("head_revision") != "index"
        or "base_revision" not in basis
        or (
            basis["base_revision"] is not None
            and not isinstance(basis["base_revision"], str)
        )
    ):
        raise CheckError(
            "growth report schema failure: basis must describe the staged index comparison"
        )
    return report


def run_growth_report(root: Path) -> None:
    if shutil.which("python3") is None:
        raise CheckError("required tool python3 was not found on PATH")
    counter = root / "scripts" / "size_report.py"
    if not counter.is_file():
        raise CheckError("required growth reporter scripts/size_report.py was not found")

    try:
        result = subprocess.run(
            ["python3", "scripts/size_report.py", "--staged", "--json"],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as error:
        raise CheckError(f"could not run scripts/size_report.py: {error}") from error
    if result.returncode != 0:
        details = result.stderr.decode("utf-8", errors="backslashreplace").strip()
        raise CheckError(
            "scripts/size_report.py --staged --json failed with "
            f"exit status {result.returncode}"
            + (f": {details}" if details else "")
        )

    report = validate_growth_report(result.stdout)
    print("Advisory growth report (no blocking threshold):")
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))


def staged_secret_patterns(root: Path) -> None:
    """Block staged lines that match common credential patterns."""
    diff = run_git(["diff", "--cached", "--no-color", "-U0"], root)
    text = diff.stdout.decode("utf-8", errors="backslashreplace")
    hits: list[tuple[int, str]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.startswith("+") or line.startswith("+++"):
            continue
        for pattern in SECRET_PATTERNS:
            if pattern.search(line):
                hits.append((number, pattern.pattern))
                break
    if hits:
        number, name = hits[0]
        raise CheckError(
            f"staged change matches a credential pattern ({name}) at diff line {number}; "
            f"{len(hits)} matching line(s) in total. Remove the secret before committing."
        )
    print("Staged credential-pattern check passed.")


def is_ancestor(root: Path, ancestor: str, descendant: str) -> bool:
    """Return whether ancestor is reachable from descendant; False on any git failure."""
    try:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError:
        return False
    return result.returncode == 0


def is_all_zero(value: str) -> bool:
    return bool(value) and value.count("0") == len(value)


def run_pre_push() -> None:
    """Refuse pushes that delete or non-fast-forward main (AGENTS.md forbids force-push)."""
    root = repository_root()
    violations: list[str] = []
    for line in sys.stdin.read().splitlines():
        fields = line.split()
        if len(fields) != 4:
            continue
        local_ref, local_sha, remote_ref, remote_sha = fields
        if remote_ref != "refs/heads/main":
            continue
        if is_all_zero(local_sha):
            violations.append("deleting main is not allowed")
        elif not is_all_zero(remote_sha) and not is_ancestor(root, remote_sha, local_sha):
            violations.append(
                "non-fast-forward push to main is not allowed "
                "(AGENTS.md forbids force-push to main)"
            )
    if violations:
        raise CheckError("pre-push refused: " + "; ".join(dict.fromkeys(violations)))
    print("Pre-push main protection passed.")


def run_pre_commit() -> None:
    root = repository_root()
    staged_python_blobs(root)
    staged_secret_patterns(root)
    run_growth_report(root)


def attribution_offense(message: str) -> tuple[int, str] | None:
    """Return the first attribution line number and text, or None when the message is clean."""
    for number, line in enumerate(message.splitlines(), start=1):
        if line.startswith("#"):
            continue
        if ATTRIBUTION_LINE_START.search(line):
            return number, line
        if ATTRIBUTION_INLINE.search(line):
            return number, line
        if ATTRIBUTION_TRAILER_KEY.match(line) and ATTRIBUTION_NAME.search(line):
            return number, line
    return None


def run_commit_message(path: str) -> None:
    message_path = Path(path)
    try:
        message = message_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise CheckError(f"could not read commit message file {path!r}: {error}") from error
    subject = message.splitlines()[0] if message.splitlines() else ""
    match = COMMIT_SUBJECT.fullmatch(subject)
    if (
        match is None
        or not match.group("scope").strip()
        or not match.group("subject").strip()
    ):
        raise CheckError(
            "commit subject must use an allowed Conventional Commit type, "
            "a nonempty scope, and a nonempty subject"
        )
    print("Conventional Commit subject passed.")
    offense = attribution_offense(message)
    if offense is not None:
        number, line = offense
        raise CheckError(
            f"commit message line {number} is an attribution line ({line.strip()!r}); "
            "commit messages carry no attribution lines (user rule)"
        )
    print("Commit attribution check passed.")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="check", required=True)
    subparsers.add_parser("pre-commit", help="check staged whitespace, Python syntax, and growth")
    message_parser = subparsers.add_parser(
        "commit-msg", help="check the commit subject and attribution lines"
    )
    message_parser.add_argument("message_file")
    subparsers.add_parser("pre-push", help="block deletion and non-fast-forward pushes to main")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.check == "pre-commit":
            run_pre_commit()
        elif args.check == "pre-push":
            run_pre_push()
        else:
            run_commit_message(args.message_file)
    except CheckError as error:
        print(f"staged-gates: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
