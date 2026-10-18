#!/usr/bin/env python3
"""End-to-end CPU self-test, part 2: the fabricated-file attacks of blind reviews A and L against assemble_report.py (identity,
commit, GPU, release-json, mode, copies, stale files, old schema, corrupt JSON, memory). Uses the helpers of report_case.py.
Usage: python3 -I selftest_attacks.py SCRATCH_DIR. Exit codes asserted: 0 PASS, 1 FAIL, 2 STOP, 3 REFUSED, 4 PRE-RUN complete."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a5_eval as ev  # noqa: E402
import report_case as sr  # noqa: E402
from report_case import case, release, fx, ref  # noqa: E402,F401


def main():
    sr.scratch.mkdir(parents=True, exist_ok=True)
    case("stop-head-is-not-the-built-commit", 2, head="a" * 40)
    case("stop-head-abbreviated", 2, head=fx.COMMIT[:12])
    case("FAIL-cuobjdump-version-empty-string", 1, rel=release(cuobjdump_version=""))
    case("FAIL-cuobjdump-version-list-with-empty-string", 1, rel=release(cuobjdump_version=[""]))
    for label, key, val in (("unknown", "engine_commit", "unknown"), ("12-character", "engine_commit", fx.COMMIT[:12]), ("gpu-n/a", "gpu", "n/a"),
                            ("gpu-llama-stand-in", "gpu", "llama.cpp stand-in"), ("driver-n/a", "driver", "n/a"), ("driver-empty", "driver", "")):
        docs = fx.make_all(ref)
        for d in docs.values():
            d["identities"][key] = val
        case(f"stop-identity-{label}", 2, docs)
    case("stop-eager-data-without-prerun-flag", 2, fx.make_all(ref, capture="eager"))
    case("stop-prerun-flag-on-graph-data", 2, prerun=True)
    docs = fx.make_all(ref)
    docs["C2"]["identities"]["exe_sha256"] = "b" * 64
    case("stop-mixed-binaries", 2, docs)
    docs = fx.make_all(ref)
    docs["B3"]["identities"]["engine_commit"] = "stale"
    case("stop-stale-file-other-commit", 2, docs)
    docs = fx.make_all(ref)
    docs["C2"] = fx.clone(docs["C1"])
    case("stop-C2-copy-of-C1", 2, docs)
    docs = fx.make_all(ref)
    docs["C1"]["identities"] = {}
    case("stop-empty-identities", 2, docs)
    docs = fx.make_all(ref)
    docs["A1"]["mode"] = "free"
    case("stop-mode-swapped", 2, docs)
    docs = fx.make_all(ref)
    docs["A2"]["prompts"][0]["runs"][0]["syncs"] = 256
    case("stop-syncs-256-in-A", 2, docs)
    docs = fx.make_all(ref)
    docs["A2"]["prompts"][0]["runs"][0]["host_writes"] = 256
    case("stop-host-writes-256-in-A (lane C Q-C6)", 2, docs)
    docs = fx.make_all(ref)
    docs["C1"]["schema"] = "cuda-rd-raw/1"
    case("stop-old-schema", 2, docs)
    case("stop-missing-raw-file", 2, drop=("C5",))
    case("stop-corrupt-json-no-traceback", 2, junk="{not json")
    docs = fx.make_all(ref)
    docs["C1"]["identities"]["mem_free_after_load"] = 500 * 2**20
    case("refused-500MB-free-after-load", 3, docs)
    docs = fx.make_all(ref)
    docs["A1"]["refused"] = {"kind": "memory", "free_after_load": 500 * 2**20}
    case("refused-call-with-the-other-calls-never-run-is-REFUSED-not-STOP", 3, docs, drop=tuple(n for n in ev.FILES if n != "A1"))
    # lane L's N5: an unwritable output path for the REFUSED / BLOCKED record is a typed STOP, not a traceback
    import subprocess
    docs = fx.make_all(ref)
    docs["C1"]["identities"]["mem_free_after_load"] = 500 * 2**20
    sr.fx.write_all(docs, sr.raw)
    cmd = [sys.executable, "-I", str(sr.HERE / "assemble_report.py"), "--raw", str(sr.raw), "--release-check", str(sr.scratch / "release.json"),
           "--wrapper-log", str(sr.scratch / "wrapper.log"), "--selftest-result", str(sr.scratch / "selftest.json"), "--head", fx.COMMIT,
           "--date", "selftest", "--out", str(sr.scratch / "no-such-dir" / "x.md")]
    p = subprocess.run(cmd, capture_output=True, text=True)
    ok = p.returncode == 2 and "cannot write the REFUSED record" in p.stdout and "Traceback" not in p.stderr
    sr.results.append(ok); print(f"{'ok  ' if ok else 'FAIL'} unwritable output path for a REFUSED record -> typed STOP: rc={p.returncode} (want 2)")
    sr.fx.write_all(fx.make_all(ref), sr.raw)
    (sr.raw / "C1.blocked").write_text("blocked\n")
    p = subprocess.run(cmd, capture_output=True, text=True)
    ok = p.returncode == 2 and "cannot write the BLOCKED record" in p.stdout and "Traceback" not in p.stderr
    sr.results.append(ok); print(f"{'ok  ' if ok else 'FAIL'} unwritable output path for a BLOCKED record -> typed STOP: rc={p.returncode} (want 2)")
    print("SELFTEST", "OK" if all(sr.results) else "FAILED", f"({len(sr.results)} cases)")
    sys.exit(0 if all(sr.results) else 1)


if __name__ == "__main__":
    main()
