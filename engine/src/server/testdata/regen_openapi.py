#!/usr/bin/env python3
"""Regenerates engine/src/server/testdata/openapi.json from api/openapi.yaml.

The contract test `the_fixture_hash_matches_the_yaml_contract` fails with "stale
fixture" whenever the YAML changes. Run this script from anywhere:

    python3 engine/src/server/testdata/regen_openapi.py

The fixture holds the paths and components of the contract as JSON plus the
sha256 of the YAML bytes, so the test can check both without a YAML parser.
"""
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[4]
YAML_PATH = ROOT / "api" / "openapi.yaml"
FIXTURE_PATH = ROOT / "engine" / "src" / "server" / "testdata" / "openapi.json"


def convert(yaml_bytes: bytes) -> str:
    doc = yaml.safe_load(yaml_bytes)
    fixture = {
        "x-source-sha256": hashlib.sha256(yaml_bytes).hexdigest(),
        "components": doc["components"],
        "paths": doc["paths"],
    }
    return json.dumps(fixture, indent=1) + "\n"


def main() -> int:
    fixture = convert(YAML_PATH.read_bytes())
    if "--check" in sys.argv:
        if fixture == FIXTURE_PATH.read_text():
            print("fixture is current")
            return 0
        print("fixture is stale; run regen_openapi.py")
        return 1
    FIXTURE_PATH.write_text(fixture)
    print("wrote", FIXTURE_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
