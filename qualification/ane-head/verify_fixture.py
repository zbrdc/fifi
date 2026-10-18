#!/usr/bin/env python3
"""Check a generated ANE decision-head fixture against its manifest."""
import argparse
import json
import os
import sys

F32_BYTES = 4


def check_array(fixture_dir, name, entry):
    """Returns (path, declared_bytes) when the manifest entry is self-consistent, else None."""
    try:
        path = os.path.join(fixture_dir, entry["file"])
        shape = [int(dim) for dim in entry["shape"]]
        declared = int(entry["bytes"])
    except (KeyError, TypeError, ValueError) as error:
        print(f"ane_head_fixture: error: array {name}: bad manifest entry ({error})", file=sys.stderr)
        return None
    cells = 1
    for dim in shape:
        cells *= dim
    if declared != cells * F32_BYTES:
        print(
            f"ane_head_fixture: error: array {name}: manifest bytes {declared} "
            f"disagree with shape {shape} times 4",
            file=sys.stderr,
        )
        return None
    return path, declared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", help="fixture directory; defaults to HEAD_FIXTURE_OUT")
    args = parser.parse_args()
    fixture_dir = args.directory or os.environ.get("HEAD_FIXTURE_OUT")
    if not fixture_dir:
        sys.exit("ane_head_fixture: error: give the fixture directory or set HEAD_FIXTURE_OUT")
    manifest_path = os.path.join(fixture_dir, "manifest.json")
    try:
        with open(manifest_path, encoding="utf-8") as stream:
            manifest = json.load(stream)
    except (OSError, ValueError) as error:
        sys.exit(f"ane_head_fixture: error: cannot read {manifest_path}: {error}")
    arrays = manifest.get("arrays")
    if not isinstance(arrays, dict) or not arrays:
        sys.exit(f"ane_head_fixture: error: {manifest_path} has no arrays table")
    if manifest.get("dtype") not in (None, "f32"):
        sys.exit(f"ane_head_fixture: error: unsupported dtype {manifest.get('dtype')!r}")
    files = 0
    total = 0
    problems = 0
    for name in sorted(arrays):
        checked = check_array(fixture_dir, name, arrays[name])
        if checked is None:
            problems += 1
            continue
        path, declared = checked
        if not os.path.isfile(path):
            print(f"ane_head_fixture: error: array {name}: missing file {path}", file=sys.stderr)
            problems += 1
            continue
        actual = os.path.getsize(path)
        if actual != declared:
            print(
                f"ane_head_fixture: error: array {name}: file {path} has {actual} bytes, "
                f"the manifest says {declared}",
                file=sys.stderr,
            )
            problems += 1
            continue
        files += 1
        total += actual
    ok = problems == 0
    print(f'ane_head_fixture={{"files":{files},"bytes":{total},"ok":{"true" if ok else "false"}}}')
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
