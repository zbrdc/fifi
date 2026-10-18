"""Synthetic CLI tests for the bounded storage-I/O harness."""

import concurrent.futures
import copy
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parent
PROBE = ROOT / "probe.py"
SPEC = importlib.util.spec_from_file_location("storage_io_probe", PROBE)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


@contextmanager
def synthetic_measurement(extent_bytes=100_000, concurrency=(1,), repetitions=1,
                         variants=None):
    scratch = Path(os.environ["TMPDIR"])
    with tempfile.TemporaryDirectory(prefix="storage-io-run-", dir=scratch) as temp:
        root = Path(temp)
        shards = copy.deepcopy(probe.SHARDS)
        ngram = {"data_offset": 192, "rows": 64, "row_bytes": 90,
                 "data_bytes": 64 * 90}
        shard1_bytes = 1_024 + 2 * (extent_bytes + 4_096) + extent_bytes
        shard2_bytes = ngram["data_offset"] + ngram["data_bytes"]
        shards["shard1"]["bytes"] = shard1_bytes
        shards["shard2"]["bytes"] = shard2_bytes
        choices = variants or [{"access": "pread", "cache_treatment": "default_uncontrolled",
                                "advice": "none"}]
        with mock.patch.object(probe, "SHARDS", shards), \
             mock.patch.object(probe, "NGRAM", ngram), \
             mock.patch.object(probe, "CONCURRENCY", tuple(concurrency)), \
             mock.patch.object(probe, "REPETITIONS", repetitions), \
             mock.patch.object(probe, "_variants", return_value=choices), \
             mock.patch.object(probe, "_resource_snapshot", return_value={"synthetic": True}), \
             mock.patch.object(probe, "variant_status", return_value=("candidate", None)):
            manifest = probe._synthetic_manifest()
            extents = manifest["expert_metadata"]["expert_tensor_extents"]
            for index, extent in enumerate(extents):
                extent["offset"] = 1_024 + index * (extent_bytes + 4_096)
                extent["length"] = extent_bytes
            by_name = {item["tensor"]: item for item in extents}
            for group_index, group in enumerate(manifest["expert_metadata"]["groups"]):
                for item in group["ranges"]:
                    item["offset"] = by_name[item["tensor"]]["offset"] + 10 + group_index * 256

            for index, (role, shard) in enumerate(manifest["shards"].items()):
                path = root / f"{role}.synthetic"
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
                try:
                    os.ftruncate(fd, shard["bytes"])
                    info = os.fstat(fd)
                finally:
                    os.close(fd)
                digest = ("a" if index == 0 else "b") * 64
                shard.update(path=str(path), full_sha256=digest,
                             hash_receipt={"method": "one_full_pass_sha256", "sha256": digest,
                                           "bytes": shard["bytes"], "mtime_ns": info.st_mtime_ns,
                                           "inode": info.st_ino})

            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            args = types.SimpleNamespace(
                reservation_id="SYNTHETIC-NOT-A-REAL-RESERVATION",
                manifest=str(manifest_path), output=str(root / "report.json"),
                seed=probe.DEFAULT_SEED, max_seconds=600)
            yield root, manifest, args


class ProbeSelfTest(unittest.TestCase):
    def run_probe(self, *args):
        scratch = Path(os.environ["TMPDIR"])
        env = os.environ.copy()
        env.update(TMP=str(scratch), TEMP=str(scratch))
        return subprocess.run(
            [sys.executable, "-B", str(PROBE), *args],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )

    def test_synthetic_contract_checks_pass_and_clean_up(self):
        before = set(Path(os.environ["TMPDIR"]).iterdir())
        result = self.run_probe("--self-test")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("storage-io self-test: PASS", result.stdout)
        self.assertEqual(set(Path(os.environ["TMPDIR"]).iterdir()), before)

    def test_measurement_requires_explicit_reservation(self):
        result = self.run_probe("--manifest", "not-read.json", "--output", "not-written.json")
        self.assertEqual(result.returncode, 2)
        self.assertIn("--reservation-id", result.stderr)

    def test_self_test_rejects_measurement_inputs(self):
        result = self.run_probe("--self-test", "--manifest", "not-read.json")
        self.assertEqual(result.returncode, 2)
        self.assertIn("cannot be combined", result.stderr)

    def test_measurement_limits_are_checked_before_input_access(self):
        result = self.run_probe(
            "--manifest", "not-read.json", "--output", "not-written.json",
            "--reservation-id", "M1-approved", "--repetitions", "4",
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--repetitions", result.stderr)

    def test_invalid_time_limit_is_checked_before_input_access(self):
        result = self.run_probe(
            "--manifest", "not-read.json", "--output", "not-written.json",
            "--reservation-id", "M1-approved", "--max-seconds", "601",
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--max-seconds", result.stderr)


class RunMeasurementTests(unittest.TestCase):
    def test_run_measurement_accepts_realistic_extents_and_caps_reads(self):
        with synthetic_measurement(extent_bytes=216_268_800) as (root, manifest, args):
            report = probe.run_measurement(args)
            self.assertTrue(report["complete_matrix"])
            self.assertEqual(len(report["results"]), 3)
            self.assertTrue(all(item["length"] == 216_268_800
                                for item in manifest["expert_metadata"]["expert_tensor_extents"]))

            oversized = copy.deepcopy(manifest)
            oversized["expert_metadata"]["groups"][0]["ranges"][0]["length"] = probe.MAX_RANGE_BYTES + 1
            bad_manifest = root / "oversized-read.json"
            bad_manifest.write_text(json.dumps(oversized), encoding="utf-8")
            bad_args = copy.copy(args)
            bad_args.manifest = str(bad_manifest)
            bad_args.output = str(root / "must-not-exist.json")
            with self.assertRaises(probe.InputError):
                probe.run_measurement(bad_args)
            self.assertFalse(Path(bad_args.output).exists())

    def test_run_measurement_preserves_samples_when_variant_fails(self):
        variant = {"access": "pread", "cache_treatment": "default_uncontrolled",
                   "advice": "F_RDADVISE"}
        with synthetic_measurement(concurrency=(2,), variants=[variant]) as (root, manifest, args):
            expert_tasks = probe.sample_tasks(manifest, "expert_three_projection_reads", 0, 2,
                                              args.seed)
            failing_offset = expert_tasks[0]["ranges"][1]["offset"]

            def reject_one_read(_fd, offset, _length):
                if offset == failing_offset:
                    raise probe.UnsupportedVariant("synthetic runtime rejection")

            with mock.patch.object(probe, "rdadvise", side_effect=reject_one_read):
                report = probe.run_measurement(args)

            self.assertFalse(report["complete_matrix"])
            self.assertEqual(report["status"], "bounded_partial")
            self.assertEqual(report["unsupported_variants"][0]["status"], "unsupported-at-runtime")
            rows = {row["workload"]: row for row in report["results"]}
            self.assertIn("ngram_row90", rows)
            self.assertIn("expert_three_projection_reads", rows)
            samples = [sample for row in rows.values() for repetition in row["repetitions"]
                       for sample in repetition["samples"]]
            expert_samples = [sample for sample in samples
                              if sample["sample_id"] in {task["id"] for task in expert_tasks}]
            self.assertCountEqual([sample["status"] for sample in expert_samples],
                                  ["complete", "unsupported"])
            partial = next(sample for sample in expert_samples if sample["status"] == "unsupported")
            self.assertGreater(partial["completed_bytes"], 0)
            self.assertLess(partial["completed_bytes"], partial["useful_bytes"])
            requested = sum(sample["requested_bytes"] for sample in samples)
            self.assertEqual(report["requested_bytes_actual"], requested)
            self.assertEqual(report["unattributed_requested_bytes"], 0)

    def test_failing_future_preserves_sibling_and_progress_bytes(self):
        variant = {"access": "pread", "cache_treatment": "default_uncontrolled",
                   "advice": "none"}
        with synthetic_measurement(concurrency=(2,), variants=[variant]) as (_root, manifest, args):
            tasks = probe.sample_tasks(manifest, "ngram_row90", 0, 2, args.seed)
            failing_id = tasks[0]["id"]
            original_sample = probe._run_sample

            class FutureSampleFailure(RuntimeError):
                def __init__(self, sample_record):
                    super().__init__("synthetic future failed after partial work")
                    self.sample_record = sample_record

            def fail_after_read(task, selected_variant, shard, page_bytes, budget, deadline):
                sample = original_sample(task, selected_variant, shard, page_bytes, budget, deadline)
                if task["id"] == failing_id:
                    sample["status"] = "failed"
                    sample["failure"] = {"type": "FutureSampleFailure",
                                         "message": "synthetic future failed after partial work"}
                    raise FutureSampleFailure(sample)
                return sample

            with mock.patch.object(probe, "_run_sample", side_effect=fail_after_read):
                report = probe.run_measurement(args)

            self.assertFalse(report["complete_matrix"])
            self.assertEqual(report["status"], "bounded_partial")
            self.assertEqual(len(report["results"]), 1)
            samples = report["results"][0]["repetitions"][0]["samples"]
            self.assertCountEqual([sample["status"] for sample in samples], ["complete", "failed"])
            failed = next(sample for sample in samples if sample["status"] == "failed")
            self.assertGreater(failed["requested_bytes"], 0)
            self.assertGreater(failed["completed_bytes"], 0)
            self.assertEqual(report["requested_bytes_actual"],
                             sum(sample["requested_bytes"] for sample in samples))
            self.assertEqual(report["unattributed_requested_bytes"], 0)

    def test_deadline_stops_submissions_and_drains_owned_sample(self):
        with synthetic_measurement(concurrency=(2,), repetitions=3) as (_root, _manifest, args):
            args.max_seconds = 1
            clock = [0.0]
            release_worker = threading.Event()
            original_executor = concurrent.futures.ThreadPoolExecutor
            original_sample = probe._run_sample

            def wait_then_sample(*call_args, **call_kwargs):
                if not release_worker.wait(timeout=5):
                    raise RuntimeError("synthetic worker release timed out")
                return original_sample(*call_args, **call_kwargs)

            class ExpireAfterFirstSubmission(original_executor):
                def __init__(self, *call_args, **call_kwargs):
                    super().__init__(*call_args, **call_kwargs)
                    self.submit_count = 0

                def submit(self, fn, *call_args, **call_kwargs):
                    future = super().submit(fn, *call_args, **call_kwargs)
                    self.submit_count += 1
                    if self.submit_count == 1:
                        clock[0] = 2.0
                        release_worker.set()
                    return future

            with mock.patch.object(probe.time, "monotonic", side_effect=lambda: clock[0]), \
                 mock.patch.object(probe, "_run_sample", side_effect=wait_then_sample), \
                 mock.patch.object(probe.concurrent.futures, "ThreadPoolExecutor",
                                   ExpireAfterFirstSubmission):
                report = probe.run_measurement(args)

            self.assertTrue(report["deadline_expired"])
            self.assertFalse(report["complete_matrix"])
            self.assertEqual(report["status"], "bounded_partial")
            self.assertEqual(len(report["results"]), 1)
            row = report["results"][0]
            self.assertEqual(row["repetitions_attempted"], 1)
            repetition = row["repetitions"][0]
            self.assertEqual(repetition["scheduled_samples"], 2)
            self.assertEqual(repetition["submitted_samples"], 1)
            self.assertEqual(repetition["returned_samples"], 1)
            self.assertEqual(repetition["unsent_samples"], 1)
            self.assertEqual(repetition["samples"][0]["status"], "deadline")
            self.assertEqual(report["requested_bytes_actual"], 0)
            self.assertEqual(report["unattributed_requested_bytes"], 0)
            self.assertIn("not a hard wall-clock guarantee", report["deadline_note"])


if __name__ == "__main__":
    unittest.main()
