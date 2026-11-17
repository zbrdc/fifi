"""Offline manager/unit-environment attestation tests: all manager calls mocked."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from gpu_guarded_handoff import attested_launch, host  # noqa: E402


class EnvironmentFingerprintTests(unittest.TestCase):
    def test_manager_fingerprint_stable_under_reordering_and_sensitive_to_values(self):
        with mock.patch.object(
            host.subprocess,
            "check_output",
            side_effect=[
                b"Z=last\nA=first\n",
                b"A=first\nZ=last\n",
                b"A=changed\nZ=last\n",
            ],
        ):
            first = host.manager_fingerprint()
            self.assertEqual(first, host.manager_fingerprint())
            self.assertNotEqual(first, host.manager_fingerprint())
            self.assertRegex(first, r"^[0-9a-f]{64}$")

    def test_malformed_or_duplicate_environment_is_rejected(self):
        for payload in (b"MALFORMED\n", b"FOO=one\nFOO=two\n"):
            with (
                self.subTest(payload=payload),
                mock.patch.object(
                    host.subprocess, "check_output", return_value=payload
                ),
            ):
                with self.assertRaises(RuntimeError):
                    host.manager_fingerprint()

    def test_unit_fingerprint_requires_complete_unique_metadata(self):
        good = (
            b"Environment=\nPassEnvironment=\nUnsetEnvironment=\n"
            b"FragmentPath=/fake/unit.service\nDropInPaths=\n"
        )
        with mock.patch.object(
            host.subprocess, "check_output", side_effect=[good, b"Environment=ONE=1\n"]
        ):
            self.assertRegex(
                host.unit_environment_fingerprint(host.UNIT), r"^[0-9a-f]{64}$"
            )
            with self.assertRaises(RuntimeError):
                host.unit_environment_fingerprint(host.UNIT)

    def test_startup_and_runtime_manager_and_unit_fingerprints_match(self):
        manager = b"Z=last\nA=first\n"
        unit = (
            b"Environment=\nPassEnvironment=\nUnsetEnvironment=\n"
            b"FragmentPath=/fake/unit.service\nDropInPaths=\n"
        )
        with mock.patch.object(
            host.subprocess, "check_output", side_effect=[manager, manager, unit, unit]
        ):
            self.assertEqual(
                host.manager_fingerprint(), attested_launch.manager_fingerprint()
            )
            self.assertEqual(
                host.unit_environment_fingerprint(host.UNIT),
                attested_launch.unit_environment_fingerprint(host.UNIT),
            )

    def test_startup_and_runtime_environment_files_agree(self):
        for sample in (b"a(sb) 0\n", b'a(sb) 1 "/fake/env" false\n', b"invalid\n"):
            with (
                self.subTest(sample=sample),
                mock.patch.object(
                    host.subprocess, "check_output", side_effect=[sample, sample]
                ),
            ):
                self.assertEqual(
                    host.environment_files_empty(host.UNIT),
                    attested_launch.environment_files_empty(host.UNIT),
                )

    def test_environment_files_fail_closed(self):
        values = (b"a(sb) 0\n", b'a(sb) 1 "/tmp/secret.env" false\n', b"garbage\n")
        with mock.patch.object(host.subprocess, "check_output", side_effect=values):
            self.assertTrue(host.environment_files_empty(host.UNIT))
            self.assertFalse(host.environment_files_empty(host.UNIT))
            self.assertFalse(host.environment_files_empty(host.UNIT))
        with mock.patch.object(
            host.subprocess,
            "check_output",
            side_effect=subprocess.CalledProcessError(1, "busctl"),
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                host.environment_files_empty(host.UNIT)

    def test_attestation_rejects_manager_or_unit_environment_change(self):
        service = host.Host2080(2048, restore_timeout=0)
        fake = {
            b"FIFI_HANDOFF_LAUNCH_PID": b"42",
            b"FIFI_HANDOFF_LAUNCH_START": b"100",
            b"FIFI_HANDOFF_BASH_PATH": b"/fake/bash",
            b"FIFI_HANDOFF_BASH_SHA256": b"digest",
            b"FIFI_HANDOFF_MANAGER_ENV_SHA256": b"a" * 64,
            b"FIFI_HANDOFF_UNIT_ENV_SHA256": b"b" * 64,
            b"FIFI_HANDOFF_START_SHA256": b"digest",
        }
        with (
            mock.patch.object(
                host, "ATTESTATION", {b"FIFI_HANDOFF_START_SHA256": Path("/fake/start")}
            ),
            mock.patch.object(host, "APPROVED_START_SHA256", "digest"),
            mock.patch.object(host, "APPROVED_BUILD_ENV_SHA256", "digest"),
            mock.patch.object(host, "digest", return_value="digest"),
            mock.patch.object(host, "process_environment", return_value=fake),
            mock.patch.object(
                host, "manager_fingerprint", return_value="a" * 64
            ) as manager,
            mock.patch.object(
                host, "unit_environment_fingerprint", return_value="b" * 64
            ) as unit,
        ):
            self.assertTrue(service.attestation_matches(42, 100))
            manager.return_value = "c" * 64
            self.assertFalse(service.attestation_matches(42, 100))
            manager.return_value = "a" * 64
            unit.return_value = "c" * 64
            self.assertFalse(service.attestation_matches(42, 100))

    def test_runtime_snapshot_rejects_manager_change_or_environment_files(self):
        service = host.Host2080(2048, restore_timeout=0)
        service.key_digest = "digest"
        service.manager_digest = "a" * 64
        service.unit_environment_digest = "b" * 64
        with (
            mock.patch.object(host, "digest", return_value="digest"),
            mock.patch.object(
                host, "manager_fingerprint", return_value="a" * 64
            ) as manager,
            mock.patch.object(
                host, "unit_environment_fingerprint", return_value="b" * 64
            ) as unit,
            mock.patch.object(
                host, "environment_files_empty", return_value=True
            ) as files,
        ):
            self.assertTrue(service.inspect_files({"files": {}}))
            manager.return_value = "c" * 64
            self.assertFalse(service.inspect_files({"files": {}}))
            manager.return_value = "a" * 64
            unit.return_value = "c" * 64
            self.assertFalse(service.inspect_files({"files": {}}))
            unit.return_value = "b" * 64
            files.return_value = False
            self.assertFalse(service.inspect_files({"files": {}}))


if __name__ == "__main__":
    unittest.main()
