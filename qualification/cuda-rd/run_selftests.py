#!/usr/bin/env python3
"""RD-10: run every self-test of the packet, one at a time, and write the checked result file the report requires.
Usage: python3 -I run_selftests.py SCRATCH_DIR OUT.json
OUT.json = {"rc": 0|1, "results": {test: rc}, "scripts": {name: sha256 of every script in MANIFEST-SCRIPTS.txt}}.
assemble_report.py refuses (FAIL) a result with rc != 0 or whose script hashes are not the ones it evaluates with."""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import assemble_report as ar  # noqa: E402

TESTS = [["selftest_tokens.py"], ["selftest_rule.py"], ["selftest_raw.py"], ["selftest_clock.py"], ["selftest_wrapper.py"], ["selftest_render.py"], ["selftest_report.py", "{S}"], ["selftest_attacks.py", "{S}"], ["selftest_tools.py", "{S}"], ["selftest_lock.py", "{S}"], ["selftest_loop.py", "{S}"], ["selftest_deps.py", "{S}"],
         ["selftest_public_items.py", "{S}"], ["bash:selftest_release_check.sh", "{S}"]]


def main(scratch, out):
    Path(scratch).mkdir(parents=True, exist_ok=True)
    results = {}
    for t in TESTS:
        name = t[0].split(":")[-1]
        cmd = ["bash", str(HERE / name)] if t[0].startswith("bash:") else [sys.executable, "-I", str(HERE / name)]
        cmd += [scratch if a == "{S}" else a for a in t[1:]]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        except (OSError, subprocess.TimeoutExpired) as e:
            (Path(scratch) / f"{name}.log").write_text(f"STOP: {name} could not run: {e}\n")
            results[name] = 2
            continue
        (Path(scratch) / f"{name}.log").write_text(p.stdout + p.stderr)
        results[name] = p.returncode
        print(f"{name}: rc={p.returncode}")
    doc = {"rc": 0 if not any(results.values()) else 1, "results": results, "scripts": ar.script_hashes()}
    Path(out).write_text(json.dumps(doc, sort_keys=True) + "\n")
    return doc["rc"]


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
