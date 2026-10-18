#!/usr/bin/env python3
"""Convert api/openapi.yaml to the JSON fixture that the Rust server tests read.

The Rust crate has no YAML parser, so engine/src/server/testdata/openapi.json holds the
`components` and `paths` of the contract and the SHA-256 of the YAML bytes.
A Rust test compares that hash with the YAML file and fails when the fixture is stale.

Usage: openapi_to_json.py [--check]
Without --check the script writes the fixture. With --check it exits 1 when the fixture differs.
"""
import hashlib
import json
import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCE = ROOT / "api" / "openapi.yaml"
TARGET = ROOT / "engine" / "src" / "server" / "testdata" / "openapi.json"


def render() -> str:
    raw = SOURCE.read_bytes()
    document = yaml.safe_load(raw)
    fixture = {
        "x-source-sha256": hashlib.sha256(raw).hexdigest(),
        "components": document["components"],
        "paths": document["paths"],
    }
    return json.dumps(fixture, indent=1, ensure_ascii=False) + "\n"


def main() -> int:
    text = render()
    if "--check" in sys.argv:
        if TARGET.read_text() != text:
            print("stale fixture: run scripts/openapi_to_json.py", file=sys.stderr)
            return 1
        print("fixture is current")
        return 0
    TARGET.write_text(text)
    print(f"wrote {TARGET.relative_to(ROOT)}")
    return 0


sys.exit(main())
