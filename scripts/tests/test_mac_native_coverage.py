"""Report metadata tests for the Mac coverage observation tool."""

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mac_native_coverage


class MacNativeCoverageTests(unittest.TestCase):
    def test_report_marks_module_accounting_uncomputed(self):
        scratch = Path(os.environ["TMPDIR"])
        self.assertTrue(scratch.is_dir())
        with tempfile.TemporaryDirectory(
                prefix="mac-native-coverage-", dir=str(scratch)) as temporary:
            root = Path(temporary)
            binary = root / "synthetic-test-binary"
            binary.write_bytes(b"synthetic binary")
            profile = root / "synthetic.profdata"
            profile.write_bytes(b"synthetic profile")
            output = root / "run"
            output.mkdir()
            (output / "run.SHA256SUMS").write_text("", encoding="utf-8")
            report_dir = root / "report"

            identity = {
                "worktree": str(root / "worktree"),
                "head": "synthetic-head",
                "engine_tree": "synthetic-engine-tree",
                "engine_tree_path": "engine",
                "cargo_lock_sha256": "synthetic-lock-hash",
                "rustc": {"version": "synthetic rustc", "llvm_version": "22.1.8"},
                "cargo": {"version": "synthetic cargo"},
                "host": "x86_64-unknown-linux-gnu",
                "fixtures": {},
                "tracked_source_files": ["engine/src/lib.rs"],
            }
            manifest = {
                "identity": identity,
                "features": [],
                "rustflags": [],
                "cpp_instrumentation": False,
                "env_set": {},
                "env_removed": [],
                "target_dir": "synthetic-target",
                "cargo_home": "synthetic-cargo-home",
                "build": {"argv": []},
                "test_binaries": [{
                    "name": binary.name,
                    "path": str(binary),
                    "sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                }],
                "test_runs": [],
                "test_totals": {},
                "profraw_files": [],
                "executables_not_attributed": [],
            }
            plan = SimpleNamespace(
                args=SimpleNamespace(
                    allow_llvm_mismatch=False,
                    existing_profdata=str(profile),
                    binaries_dir=None,
                ),
                report_dir=report_dir,
                include_prefixes=[],
                out=output,
                exclude_regex="synthetic-filter",
                excludes=[],
            )
            tools = {
                "selected": {
                    "llvm-profdata": {"realpath": "synthetic-profdata"},
                    "llvm-cov": {"realpath": "synthetic-cov"},
                },
                "llvm_match": {"rustc_llvm_major": 22},
            }
            export = {
                "data": [{
                    "totals": {"lines": {"count": 1, "covered": 1}},
                    "files": [{
                        "filename": str(root / "worktree/engine/src/lib.rs"),
                        "summary": {"lines": {"count": 1, "covered": 1}},
                    }],
                }],
            }

            def synthetic_capture(argv, timeout=300):
                if argv[1] == "report":
                    return {"stdout": "", "stderr": "", "exit": 0}
                return {
                    "stdout": json.dumps(export),
                    "stderr": "",
                    "exit": 0,
                }

            with patch.object(
                    mac_native_coverage, "capture", side_effect=synthetic_capture) as capture:
                summary = mac_native_coverage.do_report(plan, manifest, tools, [])

            self.assertEqual(capture.call_count, 2)
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["module_accounting"], "not_computed")
            self.assertEqual(
                summary["module_accounting_decision"],
                "docs/decisions/0034-coverage-accounting.md",
            )
            self.assertNotIn("proposed_grouping_by_module", summary)
            self.assertEqual(summary["files"][0]["file"], "engine/src/lib.rs")
            self.assertEqual(summary["targets"]["per_file_percent"], 60.0)
            self.assertEqual(summary["targets"]["aggregate_percent"], 80.0)
            self.assertIn("Decision 0034", summary["scope"])
            self.assertIn("Decision 0034", summary["targets"]["note"])

            saved = json.loads(
                (report_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["module_accounting"], "not_computed")
            self.assertEqual(
                saved["module_accounting_decision"],
                "docs/decisions/0034-coverage-accounting.md",
            )


if __name__ == "__main__":
    unittest.main()
