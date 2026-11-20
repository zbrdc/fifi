import fcntl
import hashlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import p2p_test_support  # noqa: F401 -- make repository packages visible under python -I

from scripts.p2p_concurrent_guard.gated_companion import (
    execute,
    CompanionOptions,
    main,
    verify_lock,
    verify_program,
    wait_for_parent,
)
from scripts.p2p_concurrent_guard.host_attestation import _seal_pinned


class GatedCompanionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.scratch = Path(directory.name)
        self.campaign = self.scratch / "campaign"
        self.campaign.mkdir(mode=0o700)
        patcher = patch(
            "scripts.p2p_concurrent_guard.gated_companion.SCRATCH", self.scratch
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_typed_options_refuse_external_paths_and_bad_values(self):
        options = CompanionOptions("/tmp", "/tmp/out", 1, 10)
        with self.assertRaises(ValueError):
            options.arguments()
        with self.assertRaises(TypeError):
            CompanionOptions("/tmp", "/tmp/out", 1, 10, inherited_lease_fd=3)
        directory = str(self.campaign)
        valid = CompanionOptions(directory, directory + "/result.jsonl", 1, 10)
        self.assertNotIn("--url", valid.arguments())
        self.assertNotIn("--model", valid.arguments())
        self.assertNotIn("--key-file", valid.arguments())
        self.assertNotIn("--prompt-file", valid.arguments())
        with self.assertRaises(ValueError):
            CompanionOptions(directory, "/tmp/out", 1, 10).arguments()
        os.chmod(directory, 0o755)
        with self.assertRaises(ValueError):
            valid.arguments()

    def test_cli_rejects_duplicate_descriptor_and_abbreviated_or_raw_args(self):
        base = [
            "gate",
            "--gate-fd",
            "4",
            "--lease-fd",
            "5",
            "--program-fd",
            "6",
            "--campaign-root",
            "/tmp",
            "--output",
            "/tmp/out",
            "--max-requests",
            "1",
            "--duration-seconds",
            "10",
        ]
        for extra in (
            ["--lease-fd", "7"],
            ["--program-fd=7"],
            ["--lease-f", "7"],
            ["--url", "http://example"],
            ["--inherited-lease-fd", "7"],
        ):
            with (
                self.subTest(extra=extra),
                patch("sys.argv", base + extra),
                patch("sys.stderr", new_callable=io.StringIO),
                self.assertRaises(SystemExit),
            ):
                main()

    def test_only_own_held_flock_on_same_inode_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.lock"
            path.touch(mode=0o600)
            fd = os.open(path, os.O_RDWR | os.O_CLOEXEC)
            try:
                with self.assertRaises(ValueError):
                    verify_lock(fd, path)
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                verify_lock(fd, path)
                other = Path(directory) / "other.lock"
                other.touch(mode=0o600)
                with self.assertRaises(ValueError):
                    verify_lock(fd, other)
            finally:
                os.close(fd)

    @unittest.skipUnless(hasattr(os, "memfd_create"), "Linux memfd required")
    def test_requires_expected_fully_sealed_program(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "script.py"
            content = b"print('hello')\n"
            path.write_bytes(content)
            digest = hashlib.sha256(content).hexdigest()
            fd = _seal_pinned(path, digest)
            try:
                verify_program(fd, digest)
                with self.assertRaisesRegex(ValueError, "digest"):
                    verify_program(fd, "0" * 64)
            finally:
                os.close(fd)
            raw = os.open(path, os.O_RDONLY)
            try:
                with self.assertRaises((OSError, ValueError)):
                    verify_program(raw, digest)
            finally:
                os.close(raw)

    def test_gate_rejects_timeout_eof_and_wrong_byte(self):
        read_fd, write_fd = os.pipe()
        try:
            with self.assertRaises(TimeoutError):
                wait_for_parent(read_fd, 0.001)
            os.write(write_fd, b"x")
            with self.assertRaises(ValueError):
                wait_for_parent(read_fd, 1)
        finally:
            os.close(read_fd)
            os.close(write_fd)
        read_fd, write_fd = os.pipe()
        os.close(write_fd)
        try:
            with self.assertRaises(ValueError):
                wait_for_parent(read_fd, 1)
        finally:
            os.close(read_fd)

    def test_exec_is_not_called_until_parent_release_and_rechecks(self):
        directory = str(self.campaign)
        options = CompanionOptions(directory, directory + "/out", 1, 10)
        read_fd, write_fd = os.pipe()
        events = []
        try:
            with (
                patch(
                    "scripts.p2p_concurrent_guard.gated_companion.verify_lock",
                    side_effect=lambda _: events.append("lock"),
                ),
                patch(
                    "scripts.p2p_concurrent_guard.gated_companion.verify_program",
                    side_effect=lambda _: events.append("program"),
                ),
                patch(
                    "scripts.p2p_concurrent_guard.gated_companion.os.execv",
                    side_effect=RuntimeError("exec test"),
                ),
                patch(
                    "scripts.p2p_concurrent_guard.gated_companion.os.set_inheritable"
                ),
            ):
                with self.assertRaises(TimeoutError):
                    execute(
                        read_fd,
                        91,
                        92,
                        options,
                        timeout=0.001,
                    )
                self.assertEqual(events, ["lock", "program"])
                os.write(write_fd, b"\x01")
                with self.assertRaisesRegex(RuntimeError, "exec test"):
                    execute(
                        read_fd,
                        91,
                        92,
                        options,
                        timeout=1,
                    )
                self.assertEqual(
                    events, ["lock", "program", "lock", "program", "lock", "program"]
                )
                read_fd = None  # execute closes its gate descriptor after release
        finally:
            if read_fd is not None:
                os.close(read_fd)
            os.close(write_fd)

    def test_campaign_permissions_are_rechecked_before_exec(self):
        options = CompanionOptions(
            str(self.campaign), str(self.campaign / "out"), 1, 10
        )
        read_fd, write_fd = os.pipe()
        try:

            def change_permissions(_fd, _timeout):
                os.chmod(self.campaign, 0o755)

            with (
                patch("scripts.p2p_concurrent_guard.gated_companion.verify_lock"),
                patch("scripts.p2p_concurrent_guard.gated_companion.verify_program"),
                patch(
                    "scripts.p2p_concurrent_guard.gated_companion.wait_for_parent",
                    side_effect=change_permissions,
                ),
                patch(
                    "scripts.p2p_concurrent_guard.gated_companion.os.execv"
                ) as execute_program,
                self.assertRaises(ValueError),
            ):
                execute(read_fd, 91, 92, options)
            read_fd = None  # execute closed the pipe after release
            execute_program.assert_not_called()
        finally:
            if read_fd is not None:
                os.close(read_fd)
            os.close(write_fd)

    def test_replaced_campaign_directory_is_rejected_before_exec(self):
        options = CompanionOptions(
            str(self.campaign), str(self.campaign / "out"), 1, 10
        )
        read_fd, write_fd = os.pipe()
        try:

            def replace_directory(_fd, _timeout):
                self.campaign.rename(self.scratch / "old-campaign")
                self.campaign.mkdir(mode=0o700)

            with (
                patch("scripts.p2p_concurrent_guard.gated_companion.verify_lock"),
                patch("scripts.p2p_concurrent_guard.gated_companion.verify_program"),
                patch(
                    "scripts.p2p_concurrent_guard.gated_companion.wait_for_parent",
                    side_effect=replace_directory,
                ),
                patch(
                    "scripts.p2p_concurrent_guard.gated_companion.os.execv"
                ) as execute_program,
                self.assertRaises(ValueError),
            ):
                execute(read_fd, 91, 92, options)
            read_fd = None
            execute_program.assert_not_called()
        finally:
            if read_fd is not None:
                os.close(read_fd)
            os.close(write_fd)


if __name__ == "__main__":
    unittest.main()
