#!/usr/bin/env python3
"""Print the unit-test executable of the engine library from `cargo test --no-run --message-format=json` (RD-2e's tests are modules of
the lib, `context/launch/tests/repeat.rs` and `resident/matmul_expert/tests/repeat.rs`, so they live in the lib test binary; there is
no `repeat` target). Usage: find_test_exes.py CARGO_LOG. Exit 2 (STOP) unless exactly one lib test executable named
`inference_engine` is found. The tests are then selected by the filter `projection_repeat_`."""
import json
import sys

exes = []
try:
    for ln in open(sys.argv[1], errors="replace"):
        if ln.startswith("{"):
            try:
                m = json.loads(ln)
            except ValueError:
                continue
            if m.get("reason") == "compiler-artifact" and m.get("executable") and m["target"]["kind"] == ["lib"] \
                    and m.get("profile", {}).get("test") is True and m["target"]["name"] == "inference_engine":
                exes.append(m["executable"])
except (OSError, IndexError) as e:
    print(f"STOP: cannot read the cargo log: {e}")
    sys.exit(2)
exes = sorted(set(exes))
for e in exes:
    print(e)
if len(exes) != 1:
    print(f"STOP: {len(exes)} lib test executables named inference_engine, expected 1")
    sys.exit(2)
