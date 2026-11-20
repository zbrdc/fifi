import fcntl
import hashlib
import os
from pathlib import Path
import select
import tempfile
import unittest
from unittest.mock import patch

import p2p_test_support  # noqa: F401 -- make repository packages visible under python -I

from scripts.p2p_concurrent_guard.core import Interval, run
from scripts.p2p_concurrent_guard.host_adapter import (
    GPU_2080,
    HostAdapter,
    SecondaryIdentity,
)
from scripts.p2p_concurrent_guard.host_attestation import _seal_pinned


def identity():
    return SecondaryIdentity(
        2080, 123, "/user.slice/2080.service", *(["a" * 64] * 4), GPU_2080, (2080,)
    )


class FakeLease:
    def __init__(self):
        self.fd = 999
        self.held = False

    def acquire(self):
        self.held = True

    def verify(self):
        return self.held

    def close(self):
        self.held = False

    def verify_released(self):
        return not self.held


class FakeBackend:
    test_backend_only = True

    def __init__(self):
        self.events = []
        self.secondary = identity()
        self.gate_read = None
        self.pin_gate = True
        self.clean_client = True
        self.saw_gate_eof = False
        self.fail_pair_after_spawn = False
        self.fail_client_after_spawn = False

    def snapshot_pair_and_admit(self):
        self.events.append("admit")
        return True

    def capture_secondary(self):
        return self.secondary

    def check_secondary_clients(self, _client):
        return True

    def stop_pair(self):
        self.events.append("stop-pair")

    def verify_pair_stopped(self):
        return True

    def start_pair_scope(self, _lease_fd):
        self.events.append("start-pair")
        if self.fail_pair_after_spawn:
            raise RuntimeError("injected partial pair start")
        return "pair-scope"

    def verify_pair_scope(self, _scope):
        return True

    def start_gated_companion(self, read_fd, _lease_fd, _program_fd, _gate_fd):
        self.gate_read = os.dup(read_fd)
        self.events.append("start-gate")
        if self.fail_client_after_spawn:
            raise RuntimeError("injected partial client start")
        return "gated-scope"

    def verify_gate_waiting(self, _scope, _lease_fd, _program_fd, _gate_fd):
        self.events.append("pin-gate")
        assert not select.select([self.gate_read], [], [], 0)[0], (
            "gate released too soon"
        )
        return self.pin_gate

    def verify_companion_exec(self, _scope):
        self.events.append("verify-exec")
        return os.read(self.gate_read, 1) == b"\x01"

    def wait_with_monitor(self, _pair, _client, verify_secondary, verify_lease):
        verify_secondary()
        verify_lease()
        return Interval(10, 30), Interval(20, 40)

    def drain_client_scope(self, _scope):
        if self.gate_read is not None:
            if select.select([self.gate_read], [], [], 0)[0]:
                self.saw_gate_eof = os.read(self.gate_read, 1) == b""
            os.close(self.gate_read)
            self.gate_read = None
        self.events.append("drain-client")
        return self.clean_client

    def drain_pair_scope(self, _scope):
        self.events.append("drain-pair")
        return True

    def restore_pair(self):
        self.events.append("restore-pair")

    def verify_pair_restored(self):
        return True

    def verify_secondary_idle(self):
        return True


@unittest.skipUnless(hasattr(os, "memfd_create"), "Linux memfd required")
class HostAdapterTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        image = Path(directory.name) / "program.py"
        image.write_bytes(b"print('fixture only')\n")
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        for name in ("sealed_companion", "sealed_gate"):
            patcher = patch(
                f"scripts.p2p_concurrent_guard.host_adapter.{name}",
                side_effect=lambda _path: _seal_pinned(image, digest),
            )
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_nonfake_backend_is_refused(self):
        backend = FakeBackend()
        backend.test_backend_only = False
        with self.assertRaises(RuntimeError):
            HostAdapter(backend, FakeLease())

    def test_gate_is_attested_before_release_and_lease_released_after_restore(self):
        backend, lease = FakeBackend(), FakeLease()
        adapter = HostAdapter(backend, lease)
        result = run(adapter, concurrent=True)
        self.assertEqual(result.status, "passed")
        self.assertEqual(
            backend.events,
            [
                "admit",
                "stop-pair",
                "start-pair",
                "start-gate",
                "pin-gate",
                "verify-exec",
                "drain-client",
                "drain-pair",
                "restore-pair",
            ],
        )
        self.assertFalse(lease.held)

    def test_unattested_gate_receives_eof_and_no_execution(self):
        backend, lease = FakeBackend(), FakeLease()
        backend.pin_gate = False
        result = run(HostAdapter(backend, lease), concurrent=True)
        self.assertEqual(result.status, "failed")
        self.assertNotIn("verify-exec", backend.events)
        self.assertTrue(backend.saw_gate_eof)
        self.assertFalse(lease.held)

    def test_unverified_drain_keeps_cleanup_inconclusive(self):
        backend, lease = FakeBackend(), FakeLease()
        backend.clean_client = False
        result = run(HostAdapter(backend, lease), concurrent=True)
        self.assertEqual(result.status, "inconclusive")
        self.assertNotIn("restore-pair", backend.events)
        self.assertTrue(lease.held)

    def test_unverified_release_reports_unknown(self):
        class RetainedLease(FakeLease):
            def verify_released(self):
                return False

        backend, lease = FakeBackend(), RetainedLease()
        result = run(HostAdapter(backend, lease), concurrent=True)
        self.assertEqual(result.status, "inconclusive")
        self.assertIsNone(result.lease_retained)

    def test_duplicate_flock_fd_makes_release_inconclusive(self):
        class DuplicatedLease:
            def __init__(self, path):
                self.path = path
                self.fd = self.duplicate = None

            def acquire(self):
                self.fd = os.open(self.path, os.O_RDWR)
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.duplicate = os.dup(self.fd)

            def verify(self):
                return self.fd is not None

            def close(self):
                os.close(self.fd)
                self.fd = None

            def verify_released(self):
                probe = os.open(self.path, os.O_RDWR)
                try:
                    try:
                        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        return False
                    return True
                finally:
                    os.close(probe)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lease"
            path.touch()
            lease = DuplicatedLease(path)
            try:
                result = run(HostAdapter(FakeBackend(), lease), concurrent=True)
                self.assertEqual(result.status, "inconclusive")
                self.assertIsNone(result.lease_retained)
            finally:
                if lease.fd is not None:
                    os.close(lease.fd)
                if lease.duplicate is not None:
                    os.close(lease.duplicate)

    def test_partial_spawn_without_handle_never_releases_lease(self):
        for side in ("pair", "client"):
            backend, lease = FakeBackend(), FakeLease()
            setattr(backend, f"fail_{side}_after_spawn", True)
            adapter = HostAdapter(backend, lease)
            try:
                result = run(adapter, concurrent=True)
                self.assertEqual(result.status, "inconclusive")
                self.assertTrue(lease.held)
                self.assertNotIn("restore-pair", backend.events)
                self.assertNotIn("verify-exec", backend.events)
            finally:
                if backend.gate_read is not None:
                    os.close(backend.gate_read)
                for name in ("program_fd", "gate_fd"):
                    fd = getattr(adapter, name)
                    if fd is not None:
                        os.close(fd)

    def test_invalid_or_multiple_gpu_owners_are_refused(self):
        backend, lease = FakeBackend(), FakeLease()
        bad = backend.secondary
        backend.secondary = SecondaryIdentity(
            bad.pid,
            bad.start_ticks,
            bad.cgroup,
            bad.executable_sha256,
            bad.unit_sha256,
            bad.environment_sha256,
            bad.config_sha256,
            bad.gpu_uuid,
            (bad.pid, 42),
        )
        result = run(HostAdapter(backend, lease), concurrent=True)
        self.assertEqual(result.status, "failed")
        self.assertNotIn("stop-pair", backend.events)


if __name__ == "__main__":
    unittest.main()
