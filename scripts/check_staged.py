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
import tempfile


COMMIT_SUBJECT = re.compile(
    r"(?P<type>feat|fix|perf|test|docs|style|chore|refactor)"
    r"\((?P<scope>[^()\r\n]*)\)(?P<breaking>!)?: (?P<subject>.*)"
)
# Subjects Git writes itself for merges, reverts, and autosquash commits.
GENERATED_SUBJECT = re.compile(r"Merge |Revert \"|(?:fixup|squash|amend)! ")
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

# Attribution lines are banned in commit messages: no line may credit a model, agent,
# harness, tool, or session.
ATTRIBUTION_LINE_START = re.compile(
    r"^\s*(?:co-authored-by:|claude-session:|worker:|generated-by:)",
    re.IGNORECASE,
)
ATTRIBUTION_INLINE = re.compile(r"\bgenerated with\b", re.IGNORECASE)
ATTRIBUTION_TRAILER_KEY = re.compile(r"^\s*[A-Za-z][A-Za-z0-9-]*\s*:\s")
ATTRIBUTION_NAME = re.compile(r"\b(?:claude|anthropic|noreply@anthropic\.com)\b", re.IGNORECASE)

# Host paths and host names of the maintainers' machines are private. The patterns are built
# from pieces so that this file does not match its own rule.
_OWNER = "da" + "n"
# A path separator as written in plain text, in JSON (an escaped slash) or on Windows.
_SEP = r"(?:\\?/|\\+)"
HOST_DETAIL_PATTERNS = [
    re.compile(
        _SEP + r"(?:media|home|users)" + _SEP + _OWNER + r"(?![A-Za-z0-9_])", re.IGNORECASE
    ),
    re.compile(r"\b(?:kub" + _OWNER + "|" + _OWNER + r"lap)\b", re.IGNORECASE),
    re.compile(r"\bpi" + "nu" + r"\b", re.IGNORECASE),
    re.compile(
        r"\bfifi" + _SEP + r"(?:scr" + "ch|ledger|knowledge|wrk" + r"tr)\b", re.IGNORECASE
    ),
]
# Vendored sources keep their pinned bytes, and the archived publisher
# snapshots in pinned/ carry the live hook bytes verbatim, so the rule skips
# exactly those two paths. The developed publisher files carry no host
# details and stay fully gated. The exemption is full-path anchored.
# New pinned files require an intentional update here plus a manifest entry.
HOST_DETAIL_SKIPPED = re.compile(
    r"(?:^|/)vendor/|^qualification/publisher/pinned/(?:publish_commits|sanitize)\.py$"
)


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


def staged_host_details(root: Path) -> None:
    """Block staged lines that name a maintainer's host path or host name."""
    diff = run_git(["diff", "--cached", "--no-color", "-U0"], root)
    text = diff.stdout.decode("utf-8", errors="backslashreplace")
    path = ""
    hits: list[tuple[str, int]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if line.startswith("+++ "):
            path = line[4:].removeprefix("b/")
        elif line.startswith("+") and not HOST_DETAIL_SKIPPED.search(path):
            if any(pattern.search(line) for pattern in HOST_DETAIL_PATTERNS):
                hits.append((path, number))
    if hits:
        path, number = hits[0]
        raise CheckError(
            f"staged change names a host path or host name in {path} at diff line {number}; "
            f"{len(hits)} matching line(s) in total. Use a placeholder or an environment "
            "variable name instead."
        )
    print("Staged host-detail check passed.")


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
    """Refuse pushes that delete or non-fast-forward main (force-push to main is forbidden)."""
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
            violations.append("non-fast-forward push to main is not allowed")
    if violations:
        raise CheckError("pre-push refused: " + "; ".join(dict.fromkeys(violations)))
    print("Pre-push main protection passed.")


def staged_publisher_gate(root: Path) -> None:
    """Publisher code can carry internal vocabulary that only the
    published-text Rust suite catches (the v4.1 slice slipped exactly
    that way past Python-only gates), so a commit touching publisher
    files must pass that suite. First build in a cold worktree is the
    cost of enforcing this at commit time."""
    result = run_git(["diff", "--cached", "--name-only"], cwd=root)
    staged = result.stdout.decode("utf-8", errors="backslashreplace").splitlines()
    if not any(
        path == "qualification/publisher" or path.startswith("qualification/publisher/")
        for path in staged
    ):
        print("Staged publisher gate not required (no publisher files staged).")
        return
    engine = root / "engine"
    if shutil.which("cargo") is None or not engine.is_dir():
        raise CheckError(
            "publisher files staged but cargo/engine is unavailable to run "
            "the published-text gate"
        )
    # cargo evaluates the working tree, but the commit records the index;
    # a partially staged publisher file would let the suite test different
    # content than the commit contains. Fail closed on any difference.
    publisher_staged = [
        path
        for path in staged
        if path == "qualification/publisher" or path.startswith("qualification/publisher/")
    ]
    dirty = run_git(["diff", "--name-only", "--", *publisher_staged], cwd=root)
    differing = dirty.stdout.decode("utf-8", errors="backslashreplace").splitlines()
    if differing:
        raise CheckError(
            "publisher files staged differ from their working-tree versions, "
            "so the published-text gate would test content other than the "
            "commit; commit or restore the unstaged changes first: "
            + ", ".join(differing)
        )
    # The gate's tests spawn `git` in throwaway directories; the hook
    # context exports GIT_DIR/GIT_INDEX_FILE/GIT_WORK_TREE, which would
    # hijack those subprocesses into this worktree's git dir (observed:
    # ignored_local_files_are_not_published_files fails under pre-commit
    # but passes standalone). Scrub the git context, keep the rest.
    gate_env = {
        key: value for key, value in os.environ.items()
        if key not in ("GIT_DIR", "GIT_INDEX_FILE", "GIT_WORK_TREE",
                       "GIT_PREFIX")
    }
    try:
        gate = subprocess.run(
            ["cargo", "test", "--locked", "--test", "published_text", "--quiet"],
            cwd=engine,
            env=gate_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as error:
        raise CheckError(f"could not run the published-text gate: {error}") from error
    if gate.returncode != 0:
        details = gate.stderr.decode("utf-8", errors="backslashreplace").strip()
        raise CheckError(
            "published-text gate failed for staged publisher files"
            + (f": {details[-800:]}" if details else "")
        )
    print(
        "Staged publisher published-text gate passed (vocabulary check only; "
        "publisher integration qualification requires "
        "scripts/check_staged.py publisher-integration)."
    )


TOKEN_FORCING_DRIVERS = "qualification/token-forcing"


def git_paths(args: list[str], root: Path) -> list[str]:
    """Paths from a git command run with -z: without it git quotes a non-ASCII path, and a
    quoted path no longer starts with the directory it lies in. The flag follows the
    subcommand, ahead of any `--` pathspec."""
    output = run_git([args[0], "-z", *args[1:]], cwd=root).stdout
    return [path.decode("utf-8", errors="backslashreplace") for path in output.split(b"\0") if path]


def staged_token_forcing_gate(root: Path) -> None:
    """The token-forcing device drivers decide which tasks a GPU qualification runs and which
    files its later leases accept, and no GPU run tests them before they cost a lease. A commit
    touching them must pass their offline suite, which needs no device, weights or lease."""
    # Without rename detection a driver moved out of the directory still counts as staged.
    staged = git_paths(["diff", "--cached", "--name-only", "--no-renames"], root)
    drivers = [path for path in staged if path.startswith(TOKEN_FORCING_DRIVERS + "/")]
    if not drivers:
        print("Staged token-forcing driver tests not required (no driver files staged).")
        return
    # The suite runs on the working tree, but the commit records the index. Any unstaged or
    # untracked file in the directory, staged or not, could make the tests pass on content
    # the commit does not hold, so fail closed on all of them.
    differing = git_paths(["diff", "--name-only", "--", TOKEN_FORCING_DRIVERS], root)
    if differing:
        raise CheckError(
            "token-forcing driver files in the working tree differ from the index, so their "
            "tests would run on content other than the commit; stage or restore the unstaged "
            "changes first: " + ", ".join(differing)
        )
    untracked = git_paths(
        ["ls-files", "--others", "--exclude-standard", "--", TOKEN_FORCING_DRIVERS], root
    )
    if untracked:
        raise CheckError(
            "untracked files in the token-forcing driver directory would take part in its "
            "tests but not in the commit; add or remove them first: " + ", ".join(untracked)
        )
    tests = root / TOKEN_FORCING_DRIVERS / "tests"
    if not tests.is_dir():
        raise CheckError(f"token-forcing driver files staged but {tests} is missing")
    try:
        suite = subprocess.run(
            [sys.executable, "-I", "-m", "unittest", "discover", "-s", str(tests)],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except OSError as error:
        raise CheckError(f"could not run the token-forcing driver tests: {error}") from error
    # unittest exits with status 5 when it found no tests to run.
    if suite.returncode == 5:
        raise CheckError(f"token-forcing driver tests failed: {tests} holds no tests")
    if suite.returncode != 0:
        tail = suite.stdout.decode("utf-8", errors="backslashreplace").strip()[-800:]
        raise CheckError(
            f"token-forcing driver tests failed with exit status {suite.returncode}"
            + (f":\n{tail}" if tail else "")
        )
    print("Staged token-forcing driver tests passed.")


def run_pre_commit() -> None:
    root = repository_root()
    staged_python_blobs(root)
    staged_secret_patterns(root)
    staged_host_details(root)
    staged_publisher_gate(root)
    staged_token_forcing_gate(root)
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


def check_conventional_subject(subject: str) -> None:
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


def run_commit_message(path: str) -> None:
    message_path = Path(path)
    try:
        message = message_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise CheckError(f"could not read commit message file {path!r}: {error}") from error
    subject = message.splitlines()[0] if message.splitlines() else ""
    if GENERATED_SUBJECT.match(subject):
        print("Git-generated commit subject accepted.")
    else:
        check_conventional_subject(subject)
        print("Conventional Commit subject passed.")
    offense = attribution_offense(message)
    if offense is not None:
        number, line = offense
        raise CheckError(
            f"commit message line {number} is an attribution line ({line.strip()!r}); "
            "commit messages carry no attribution lines"
        )
    print("Commit attribution check passed.")


PUBLISHER_INTEGRATION_CHECKS: tuple[tuple[str, str, list[str]], ...] = (
    # (label, working directory relative to the repository root, argv)
    ("rust formatting", "engine", ["cargo", "fmt", "--check"]),
    (
        "strict clippy",
        "engine",
        ["cargo", "clippy", "--locked", "--all-targets", "--", "-D", "warnings"],
    ),
    (
        "full rust suite including published_text",
        "engine",
        ["cargo", "test", "--locked", "--quiet"],
    ),
    (
        "publisher python suite",
        ".",
        ["python3", "-m", "unittest", "discover", "-s", "qualification/publisher/tests"],
    ),
    (
        "scripts python suite",
        ".",
        ["python3", "-m", "unittest", "discover", "-s", "scripts/tests"],
    ),
    ("source inventory validation", ".", ["python3", "scripts/qual_inventory.py", "validate"]),
    ("pack registry validation", ".", ["python3", "scripts/qual_inventory.py", "packs"]),
)


def run_publisher_integration(root: Path) -> None:
    """Complete publisher integration qualification. The pre-commit
    published-text gate is a fast vocabulary check only; a passing hook
    never establishes this qualification. Run on a clean worktree so the
    suites evaluate exactly the tree that would be integrated."""
    status = run_git(["status", "--porcelain"], cwd=root)
    if status.stdout.strip():
        raise CheckError(
            "publisher integration gate requires a clean worktree so the "
            "qualification evaluates exactly the integrated tree; commit or "
            "stash first"
        )
    # The mlx unit suite's Scratch requires a task TMPDIR (deliberate); a
    # caller without one fails six tests on the missing variable rather than
    # on integration. Provide one for the checks' lifetime, under a
    # task-owned parent — TMP, then TEMP, then a scratch directory under
    # CARGO_TARGET_DIR — and remove it afterwards. Never /tmp: the host rule
    # forbids agent scratch there. A caller that set its own TMPDIR is
    # untouched.
    created_tmpdir = None
    if not os.environ.get("TMPDIR"):
        parent = None
        for key in ("TMP", "TEMP"):
            candidate = os.environ.get(key)
            if candidate and candidate != "/tmp":
                parent = candidate
                break
        if parent is None:
            cargo_target = os.environ.get("CARGO_TARGET_DIR")
            if cargo_target:
                parent = os.path.join(cargo_target, "fifi-gate-scratch")
                os.makedirs(parent, exist_ok=True)
        if parent is None:
            raise CheckError(
                "publisher integration gate needs a task scratch directory: "
                "set TMPDIR (or TMP, TEMP or CARGO_TARGET_DIR) to a non-/tmp "
                "directory"
            )
        created_tmpdir = tempfile.mkdtemp(prefix="publisher-gate-tmp.", dir=parent)
        os.environ["TMPDIR"] = created_tmpdir
    try:
        for label, relative, argv in PUBLISHER_INTEGRATION_CHECKS:
            if shutil.which(argv[0]) is None:
                raise CheckError(
                    f"publisher integration check \"{label}\" cannot run: "
                    f"{argv[0]} was not found on PATH"
                )
            try:
                result = subprocess.run(
                    argv,
                    cwd=str(root / relative),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            except OSError as error:
                raise CheckError(f"publisher integration check \"{label}\" failed to start: {error}") from error
            if result.returncode != 0:
                tail = result.stdout.decode("utf-8", errors="backslashreplace").strip()[-800:]
                raise CheckError(
                    f"publisher integration check \"{label}\" failed with exit status "
                    f"{result.returncode}" + (f":\n{tail}" if tail else "")
                )
            print(f"publisher integration check passed: {label}")
    finally:
        if created_tmpdir is not None:
            del os.environ["TMPDIR"]
            shutil.rmtree(created_tmpdir, ignore_errors=True)
    print(
        "Publisher integration qualification PASSED (clean tree, all mandated "
        "suites green). The pre-commit published-text gate alone does not "
        "establish this qualification."
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="check", required=True)
    subparsers.add_parser("pre-commit", help="check staged whitespace, Python syntax, secrets, host details, and growth")
    message_parser = subparsers.add_parser(
        "commit-msg", help="check the commit subject and attribution lines"
    )
    message_parser.add_argument("message_file")
    subparsers.add_parser("pre-push", help="block deletion and non-fast-forward pushes to main")
    subparsers.add_parser(
        "publisher-integration",
        help="run the complete publisher integration qualification (clean worktree required)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.check == "pre-commit":
            run_pre_commit()
        elif args.check == "pre-push":
            run_pre_push()
        elif args.check == "publisher-integration":
            run_publisher_integration(repository_root())
        else:
            run_commit_message(args.message_file)
    except CheckError as error:
        print(f"staged-gates: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
