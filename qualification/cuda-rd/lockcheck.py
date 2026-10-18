#!/usr/bin/env python3
"""RD-10a-7: the qualification crate's Cargo.lock agrees with the engine's. Usage: lockcheck.py QUAL_LOCK ENGINE_LOCK
Every (name, version, source) entry the qualification lock shares with the engine lock must carry the same checksum; the only
package that may exist in the qualification lock alone is `cuda-rd` itself. Exit 0 = agrees, 1 = differs (named), 2 = typed STOP
(file unreadable, or a lock with no entries at all: an empty lock cannot agree)."""
import re
import sys
from pathlib import Path


def packages(path):
    """Every entry of the lock keyed by (name, version, source) with its checksum, so two versions of a name never collapse.
    A [[patch.unused]] block (cargo's record of an unused [patch.crates-io] entry, the mlx-sys one here) is not a package: skipped."""
    text = Path(path).read_text()
    out = {}
    for block in re.split(r"(?m)^\[\[", text)[1:]:
        if block.startswith("patch.unused]]"):      # 16c F3: 136 entries in this lock, not 137
            continue
        fields = dict(re.findall(r'(?m)^(name|version|source|checksum) = "([^"]*)"', block))
        if "name" not in fields or "version" not in fields:
            raise ValueError("lock entry without name or version")
        out[(fields["name"], fields["version"], fields.get("source"))] = fields.get("checksum")
    return out


def main(argv):
    try:
        q, e = packages(argv[1]), packages(argv[2])
    except (OSError, IndexError, ValueError) as err:
        print(f"STOP: cannot read the lock files: {err}")
        return 2
    for side, pkgs in (("qualification", q), ("engine", e)):
        if not pkgs:
            print(f"STOP: the {side} lock has no package entries: an empty or block-less lock cannot agree")
            return 2
    key = lambda k: (k[0], k[1], k[2] or "")  # noqa: E731
    shown = lambda k: f"{k[0]} {k[1]} ({k[2] or 'no source'})"  # noqa: E731
    bad = [f"{shown(k)}: checksum {q[k]} in the qualification lock, {e[k]} in the engine lock" for k in sorted(set(q) & set(e), key=key) if q[k] != e[k]]
    bad += [f"{shown(k)}: only in the qualification lock" for k in sorted(set(q) - set(e), key=key) if k[0] != "cuda-rd"]
    for b in bad:
        print(b)
    print(f"{len(q)} entries in the qualification lock, {len(set(q) & set(e))} shared with the engine lock: " + ("AGREE" if not bad else "DIFFER"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
