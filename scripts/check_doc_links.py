#!/usr/bin/env python3
"""Fail when a tracked markdown or yaml file links to a path that .gitignore covers.

The check reads the staged snapshot (the Git index), so it validates what the
next commit would record. A relative link target is resolved against the
directory of the document that contains it. Targets that leave the repository,
that use a URL scheme, that are absolute, or that are pure fragments are out of
scope. Plain-text mentions of an ignored path are allowed; only link targets are
checked. Link targets cover markdown inline and reference links (an optional
quoted title, and an optional ?query or #fragment suffix, are stripped), HTML
href attributes, angle autolinks, and - in yaml documents - plain path values
under mapping keys.
"""

from __future__ import annotations

import os
from pathlib import Path
import posixpath
import re
import shutil
import subprocess
import sys
import urllib.parse


DOCUMENT_SUFFIXES = (".md", ".markdown", ".mdx", ".yaml", ".yml")
YAML_SUFFIXES = (".yaml", ".yml")
INLINE_LINK = re.compile(
    r"\[[^\]]*\]\(\s*(?P<target><[^>]*>|[^)\s]*)(?:\s+(?:\"[^\"]*\"|'[^']*'))?\s*\)"
)
REFERENCE_LINK = re.compile(r"^[ \t]*\[[^\]]+\]:[ \t]*(?P<target><[^>]*>|\S+)", re.MULTILINE)
HTML_LINK = re.compile(
    r"<a\b[^>]*?\bhref\s*=\s*[\"'](?P<target>[^\"']*)[\"']", re.IGNORECASE
)
# Angle targets must look like a path (contain "/" or "."); bare words such as
# a shell <target> placeholder stay plain text.
ANGLE_AUTOLINK = re.compile(r"(?<!\()(?<!: )<(?P<target>[^<>\s]*[/.][^<>\s]*)>")
YAML_PLAIN_VALUE = re.compile(
    r"^[ \t]*(?:-[ \t]+)?[^\s:#][^:\n]*?:[ \t]+(?P<target>[^\s#][^\s]*)[ \t]*$",
    re.MULTILINE,
)
URI_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*:")


class CheckError(Exception):
    """A required check could not validate its inputs."""


def run_git(arguments: list[str], cwd: Path) -> subprocess.CompletedProcess[bytes]:
    if shutil.which("git") is None:
        raise CheckError("required tool git was not found on PATH")
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as error:
        raise CheckError(f"could not run git {arguments[0]}: {error}") from error
    return result


def repository_root() -> Path:
    result = run_git(["rev-parse", "--show-toplevel"], Path.cwd())
    if result.returncode != 0:
        raise CheckError("git rev-parse --show-toplevel failed; run inside a Git worktree")
    raw_root = result.stdout.rstrip(b"\r\n")
    if not raw_root:
        raise CheckError("git rev-parse returned an empty repository path")
    return Path(os.fsdecode(raw_root))


def staged_documents(root: Path) -> list[tuple[str, bytes]]:
    """Return the staged blobs of tracked regular documents, skipping symlinks."""
    result = run_git(["ls-files", "--stage", "-z"], root)
    if result.returncode != 0:
        raise CheckError("git ls-files --stage failed; the index is not readable")
    documents: list[tuple[str, bytes]] = []
    for record in result.stdout.split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", 1)
            mode_text, object_id, stage_text = metadata.split()
            mode = int(mode_text, 8)
            stage = int(stage_text)
        except (ValueError, TypeError) as error:
            raise CheckError("could not parse a Git index entry") from error
        if stage != 0:
            raise CheckError("the index contains unmerged paths; resolve them before committing")
        path = os.fsdecode(raw_path)
        if mode == 0o120000:
            continue
        if mode not in (0o100644, 0o100755):
            continue
        if not path.endswith(DOCUMENT_SUFFIXES):
            continue
        documents.append((path, object_id))
    return documents


def blob_text(root: Path, object_id: bytes, path: str) -> str:
    result = run_git(["cat-file", "blob", object_id.decode("ascii")], root)
    if result.returncode != 0:
        raise CheckError(f"could not read the staged blob of {path}")
    return result.stdout.decode("utf-8", errors="replace")


def link_targets(text: str, yaml_values: bool = False) -> list[tuple[int, str]]:
    """Return the (line, target) pairs of every link the document contains.

    Inline and reference links, HTML href attributes, and angle autolinks are
    always scanned; plain mapping values count as links only in yaml documents.
    """
    found: list[tuple[int, str]] = []
    patterns = (INLINE_LINK, REFERENCE_LINK, HTML_LINK, ANGLE_AUTOLINK)
    if yaml_values:
        patterns += (YAML_PLAIN_VALUE,)
    for pattern in patterns:
        for match in pattern.finditer(text):
            target = match.group("target")
            if target.startswith("<") and target.endswith(">"):
                target = target[1:-1]
            line = text.count("\n", 0, match.start()) + 1
            found.append((line, target))
    return found


def resolved_target(document: str, target: str) -> str | None:
    """Resolve a link target to a repository path, or None when out of scope."""
    target = target.split("#", 1)[0].split("?", 1)[0]
    if not target or target.startswith("/") or URI_SCHEME.match(target):
        return None
    target = urllib.parse.unquote(target)
    if not target:
        return None
    resolved = posixpath.normpath(posixpath.join(posixpath.dirname(document), target))
    if resolved == ".." or resolved.startswith("../"):
        return None
    return resolved


def ignored_paths(root: Path, candidates: set[str]) -> set[str]:
    """Return the candidates that .gitignore covers, ignoring their index state.

    Directory ignore patterns only match with a trailing slash when no index or
    filesystem state is available, so every candidate is also probed with one
    appended; the reported path keeps the candidate's own spelling.
    """
    if not candidates:
        return set()
    probes: dict[str, str] = {}
    for candidate in sorted(candidates):
        probes[candidate] = candidate
        if not candidate.endswith("/"):
            probes[candidate + "/"] = candidate
    payload = b"\0".join(
        path.encode("utf-8", errors="surrogateescape") for path in sorted(probes)
    ) + b"\0"
    if shutil.which("git") is None:
        raise CheckError("required tool git was not found on PATH")
    try:
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", "-z", "--stdin"],
            cwd=root,
            input=payload,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as error:
        raise CheckError(f"could not run git check-ignore: {error}") from error
    if result.returncode not in (0, 1):
        details = result.stderr.decode("utf-8", errors="backslashreplace").strip()
        raise CheckError("git check-ignore failed" + (f": {details}" if details else ""))
    ignored_probes = {os.fsdecode(item) for item in result.stdout.split(b"\0") if item}
    return {candidate for probe, candidate in probes.items() if probe in ignored_probes}


def check(root: Path) -> list[str]:
    violations: list[str] = []
    pending: dict[str, list[tuple[str, int, str]]] = {}
    for path, object_id in staged_documents(root):
        text = blob_text(root, object_id, path)
        yaml_values = path.endswith(YAML_SUFFIXES)
        for line, target in link_targets(text, yaml_values):
            resolved = resolved_target(path, target)
            if resolved is None:
                continue
            pending.setdefault(resolved, []).append((path, line, target))
    covered = ignored_paths(root, set(pending))
    for resolved in sorted(covered):
        for path, line, target in sorted(pending[resolved]):
            violations.append(f"{path}:{line}: link to ignored path {target} ({resolved})")
    return violations


def main() -> int:
    try:
        root = repository_root()
        violations = check(root)
    except CheckError as error:
        print(f"check_doc_links: {error}", file=sys.stderr)
        return 1
    if violations:
        for violation in violations:
            print(f"check_doc_links: {violation}", file=sys.stderr)
        print(
            f"check_doc_links: {len(violations)} link(s) point at ignored paths; "
            "cite them as plain text instead",
            file=sys.stderr,
        )
        return 1
    print("Document link check passed (no tracked document links to an ignored path).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
