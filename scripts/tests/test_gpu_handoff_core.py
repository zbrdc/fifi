import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpu_guarded_handoff import core as h
from gpu_guarded_handoff import workload as s


class FakeService:
    def __init__(self, path, active=True):
        self.path, self.active = path, active
        self.calls = []
        self.match = True
        self.restore_ok = True
        self.stop_error = False
        self.assert_lock = True

    def _action(self, name):
        if self.assert_lock:
            fd = os.open(self.path, os.O_RDWR)
            try:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    pass
                else:
                    raise AssertionError("service action without exclusive lease")
            finally:
                os.close(fd)
        self.calls.append(name)

    def snapshot(self):
        self.calls.append("snapshot")
        return {"active": self.active, "identity": "original"}

    def was_active(self, snapshot):
        return snapshot["active"]

    def admit(self, snapshot):
        self.calls.append("admit")

    def still_matches(self, snapshot):
        return self.match

    def stop(self, snapshot):
        self._action("stop")
        if self.stop_error:
            raise TimeoutError("stop may have been accepted")

    def verify_stopped(self, snapshot):
        self.calls.append("stopped")

    def restore(self, snapshot):
        self._action("restore")

    def verify_restored(self, snapshot):
        self.calls.append("verify")
        return self.restore_ok


class FakeWork:
    def __init__(self):
        self.calls, self.exit, self.wait_error, self.cleanup_ok = [], 0, None, True
        self.start_error = None
        self.send_signal = None
        self.before_check = None

    def start(self):
        self.calls.append("start")
        if self.start_error:
            raise self.start_error

    def wait(self, timeout, lease_check):
        self.calls.append("wait")
        if self.before_check:
            self.before_check()
        lease_check()
        if self.send_signal:
            os.kill(os.getpid(), self.send_signal)
        if self.wait_error:
            raise self.wait_error
        return self.exit

    def cleanup_and_verify_empty(self):
        self.calls.append("cleanup")
        return self.cleanup_ok


class HandoffTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "gpu.lock"
        self.path.write_text("never truncate this marker")
        self.lease = h.Lease(self.path)
        self.service = FakeService(self.path)
        self.work = FakeWork()

    def tearDown(self):
        self.lease.release()
        self.tmp.cleanup()

    def test_denied_lease_causes_zero_service_or_work_actions(self):
        with self.path.open("r+") as other:
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(h.Busy):
                h.handoff(self.lease, self.service, self.work, 1)
        self.assertEqual(self.service.calls, [])
        self.assertEqual(self.work.calls, [])
        self.assertEqual(self.path.read_text(), "never truncate this marker")

    def test_shared_lease_pins_fixed_configured_host_path(self):
        with (
            patch.object(h, "SHARED_LOCK", self.path),
            patch.object(h, "CANONICAL_LOCK", self.path),
        ):
            fixed = h.SharedLease()
            self.assertEqual(fixed.logical_path, self.path)
            fixed.acquire()
            try:
                with self.assertRaises(h.Busy):
                    h.Lease(self.path).acquire()
            finally:
                fixed.release()

    def test_shared_entrypoint_cannot_select_an_uncontended_wrong_lock(self):
        alternate = Path(self.tmp.name) / "alternate.lock"
        alternate.touch()
        with (
            self.path.open("r+") as holder,
            patch.object(h, "SHARED_LOCK", self.path),
            patch.object(h, "CANONICAL_LOCK", self.path),
        ):
            fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(h.Busy):
                h.handoff_shared(self.service, self.work, 1)
            # Supplying an alternate Lease to the fixed-host entrypoint is
            # not part of its API and must never reach service actions.
            with self.assertRaises(TypeError):
                h.handoff_shared(h.Lease(alternate), self.service, self.work, 1)
        self.assertEqual(self.service.calls, [])
        self.assertEqual(self.work.calls, [])

    def test_shared_entrypoint_rejects_retargeted_lock_alias_before_stop(self):
        root = Path(self.tmp.name)
        first, second, link = root / "a", root / "b", root / "shared"
        first.mkdir()
        second.mkdir()
        (first / "gpu.lock").touch()
        (second / "gpu.lock").touch()
        link.symlink_to(first, target_is_directory=True)
        self.service.path = first / "gpu.lock"

        def retarget(_snapshot):
            link.unlink()
            link.symlink_to(second, target_is_directory=True)
            return True

        self.service.still_matches = retarget
        with (
            patch.object(h, "SHARED_LOCK", link / "gpu.lock"),
            patch.object(h, "CANONICAL_LOCK", first / "gpu.lock"),
        ):
            out = h.handoff_shared(self.service, self.work, 1)
        self.assertEqual(out.result, "failed")
        self.assertIsNone(out.restore_confirmed)
        self.assertNotIn("stop", self.service.calls)
        self.assertEqual(self.work.calls, [])

    def test_shared_entrypoint_rejects_wrong_alias_before_stop(self):
        alternate = self.path.with_name("wrong.lock")
        alternate.touch()
        with (
            patch.object(h, "SHARED_LOCK", alternate),
            patch.object(h, "CANONICAL_LOCK", self.path),
        ):
            with self.assertRaises(h.LeaseChanged):
                h.handoff_shared(self.service, self.work, 1)
        self.assertEqual(self.service.calls, [])

    def test_active_service_stop_work_restore_under_one_lease(self):
        out = h.handoff(self.lease, self.service, self.work, 1)
        self.assertEqual(out.result, "passed")
        self.assertEqual(
            self.service.calls,
            ["snapshot", "admit", "stop", "stopped", "restore", "verify"],
        )
        self.assertEqual(self.work.calls, ["start", "wait", "cleanup"])
        self.assertIsNone(out.retained_lease)
        self.assertEqual(self.path.read_text(), "never truncate this marker")

    def test_non_boolean_restore_is_inconclusive_and_keeps_lease(self):
        self.service.restore_ok = None
        outcome = h.handoff(self.lease, self.service, self.work, 1)
        self.assertEqual(outcome.result, "inconclusive")
        self.assertIs(outcome.retained_lease, self.lease)
        with self.assertRaises(h.Busy):
            h.Lease(self.path).acquire()

    def test_non_boolean_cleanup_is_inconclusive_and_keeps_lease(self):
        self.work.cleanup_ok = None
        outcome = h.handoff(self.lease, self.service, self.work, 1)
        self.assertEqual(outcome.result, "inconclusive")
        self.assertIs(outcome.retained_lease, self.lease)
        self.assertNotIn("restore", self.service.calls)

    def test_inactive_service_is_not_started_or_stopped(self):
        self.service.active = False
        out = h.handoff(self.lease, self.service, self.work, 1)
        self.assertEqual(out.result, "passed")
        self.assertNotIn("stop", self.service.calls)
        self.assertNotIn("restore", self.service.calls)
        self.assertIn("verify", self.service.calls)

    def test_inactive_changed_during_work_retains_lease(self):
        self.service.active = False
        self.service.restore_ok = False
        out = h.handoff(self.lease, self.service, self.work, 1)
        self.assertEqual(out.result, "inconclusive")
        self.assertIs(out.retained_lease, self.lease)
        self.assertNotIn("restore", self.service.calls)

    def test_same_and_other_lease_instances_cannot_reenter(self):
        self.lease.acquire()
        with self.assertRaises(h.Busy):
            self.lease.acquire()
        with self.assertRaises(h.Busy):
            h.Lease(self.path).acquire()

    def test_ambiguous_active_state_is_rejected_before_actions(self):
        self.service.was_active = lambda _snapshot: None
        out = h.handoff(self.lease, self.service, self.work, 1)
        self.assertEqual(out.result, "failed")
        self.assertIsNone(out.restore_confirmed)
        self.assertNotIn("stop", self.service.calls)
        self.assertNotIn("start", self.work.calls)

    def test_changed_inode_prevents_stop(self):
        def swap(_snapshot):
            old = self.path.with_name("old.lock")
            self.path.rename(old)
            self.path.write_text("different inode")
            return True

        self.service.still_matches = swap
        out = h.handoff(self.lease, self.service, self.work, 1)
        self.assertEqual(out.result, "failed")
        self.assertIsNone(out.restore_confirmed)
        self.assertNotIn("stop", self.service.calls)
        self.assertNotIn("start", self.work.calls)

    def test_alias_retarget_prevents_stop(self):
        root = Path(self.tmp.name)
        first, second, link = root / "first", root / "second", root / "current"
        first.mkdir()
        second.mkdir()
        (first / "gpu.lock").touch()
        (second / "gpu.lock").touch()
        link.symlink_to(first, target_is_directory=True)
        self.lease = h.Lease(link / "gpu.lock")
        self.service.path = first / "gpu.lock"

        def retarget(_snapshot):
            link.unlink()
            link.symlink_to(second, target_is_directory=True)
            return True

        self.service.still_matches = retarget
        out = h.handoff(self.lease, self.service, self.work, 1)
        self.assertEqual(out.result, "failed")
        self.assertNotIn("stop", self.service.calls)
        self.assertNotIn("start", self.work.calls)

    def test_inode_change_during_work_is_inconclusive_and_never_restores_unfenced(self):
        def swap():
            self.path.rename(self.path.with_name("old.lock"))
            self.path.write_text("new inode")

        self.work.before_check = swap
        out = h.handoff(self.lease, self.service, self.work, 1)
        self.assertEqual(out.result, "inconclusive")
        self.assertIs(out.retained_lease, self.lease)
        self.assertEqual(self.work.calls[-1], "cleanup")
        self.assertNotIn("restore", self.service.calls)

    def test_failed_stop_still_attempts_restoration(self):
        self.service.stop_error = True
        out = h.handoff(self.lease, self.service, self.work, 1)
        self.assertEqual(out.result, "failed")
        self.assertIn("restore", self.service.calls)
        self.assertNotIn("start", self.work.calls)

    def test_timeout_cleans_then_restores(self):
        self.work.wait_error = TimeoutError("deadline")
        out = h.handoff(self.lease, self.service, self.work, 1)
        self.assertEqual(out.result, "failed")
        self.assertEqual(self.work.calls[-1], "cleanup")
        self.assertIn("restore", self.service.calls)

    def test_signal_cleans_then_restores_and_resets_handler(self):
        old = signal.getsignal(signal.SIGTERM)
        self.work.send_signal = signal.SIGTERM
        out = h.handoff(self.lease, self.service, self.work, 1)
        self.assertEqual(out.result, "failed")
        self.assertEqual(self.work.calls[-1], "cleanup")
        self.assertIn("restore", self.service.calls)
        self.assertEqual(signal.getsignal(signal.SIGTERM), old)

    def test_partial_start_requires_cleanup(self):
        self.work.start_error = RuntimeError("started but scope not pinned")
        self.work.cleanup_ok = False
        out = h.handoff(self.lease, self.service, self.work, 1)
        self.assertEqual(out.result, "inconclusive")
        self.assertIs(out.retained_lease, self.lease)
        self.assertNotIn("restore", self.service.calls)

    def test_cleanup_failure_retains_lease_and_does_not_touch_service(self):
        self.work.cleanup_ok = False
        out = h.handoff(self.lease, self.service, self.work, 1)
        self.assertEqual(out.result, "inconclusive")
        self.assertIs(out.retained_lease, self.lease)
        self.assertNotIn("restore", self.service.calls)
        with self.assertRaises(h.Busy):
            h.Lease(self.path).acquire()

    def test_restore_failure_retains_lease(self):
        self.service.restore_ok = False
        out = h.handoff(self.lease, self.service, self.work, 1)
        self.assertEqual(out.result, "inconclusive")
        self.assertIs(out.retained_lease, self.lease)
        with self.assertRaises(h.Busy):
            h.Lease(self.path).acquire()

    def test_work_failure_restores_but_not_pass(self):
        self.work.exit = 7
        out = h.handoff(self.lease, self.service, self.work, 1)
        self.assertEqual(out.result, "failed")
        self.assertTrue(out.restore_confirmed)

    def test_scope_runner_does_not_pass_lease_to_workload(self):
        self.lease.acquire()
        self.assertFalse(os.get_inheritable(self.lease.fd))
        fake = unittest.mock.MagicMock()
        scope = unittest.mock.MagicMock()
        with (
            patch.object(s.scope_guard, "check_manager"),
            patch.object(s.scope_guard, "Scope", return_value=scope),
            patch.object(s.subprocess, "Popen", return_value=fake) as launch,
        ):
            s.ScopedCommand(["true"], lambda: None).start()
        self.assertTrue(launch.call_args.kwargs["close_fds"])
        self.assertNotIn("pass_fds", launch.call_args.kwargs)

    def test_scope_cleanup_does_not_signal_foreign_process(self):
        # A real unrelated CPU process survives scope cleanup; scope identity
        # checks and detached-child cleanup are separately tested in scope_guard.
        foreign = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(20)"]
        )
        fake = unittest.mock.MagicMock()
        scope = unittest.mock.MagicMock()
        scope.populated.return_value = False
        fake.poll.return_value = 0
        try:
            with patch.object(s.scope_guard, "Scope", return_value=scope):
                command = s.ScopedCommand(["true"], lambda: None)
                command.child = fake
                self.assertTrue(command.cleanup_and_verify_empty())
            self.assertIsNone(foreign.poll())
        finally:
            foreign.terminate()
            foreign.wait(timeout=5)

    @unittest.skipUnless(
        os.environ.get("FIFI_SCOPE_REAL_TEST") == "1",
        "explicit CPU-only systemd-scope integration test",
    )
    def test_real_scope_cleans_detached_child_and_preserves_foreign_process(self):
        marker = str(Path(self.tmp.name) / "detached.pid")
        code = (
            "import os,signal,sys,time\n"
            "if os.fork()==0:\n"
            " os.setsid()\n"
            " signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
            ' open(sys.argv[1],"w").write(str(os.getpid()))\n'
            " while True: time.sleep(1)\n"
            "while True: time.sleep(1)\n"
        )
        foreign = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(40)"]
        )
        command = s.ScopedCommand([sys.executable, "-c", code, marker], lambda: None)
        try:
            out = h.handoff(self.lease, self.service, command, 2)
            self.assertEqual(out.result, "failed")  # deliberate deadline
            self.assertTrue(out.workload_drained)
            self.assertTrue(out.restore_confirmed)
            self.assertTrue(Path(marker).is_file(), "detached child never started")
            detached = int(Path(marker).read_text())
            deadline = time.monotonic() + 3
            while Path(f"/proc/{detached}").exists() and time.monotonic() < deadline:
                state = (
                    Path(f"/proc/{detached}/stat")
                    .read_text()
                    .rsplit(")", 1)[1]
                    .split()[0]
                )
                if state == "Z":
                    break  # already dead; waiting for its reaper is not our job
                time.sleep(0.1)
            if Path(f"/proc/{detached}").exists():
                state = (
                    Path(f"/proc/{detached}/stat")
                    .read_text()
                    .rsplit(")", 1)[1]
                    .split()[0]
                )
                self.assertEqual(state, "Z", "detached child survived scope cleanup")
            self.assertIsNone(foreign.poll(), "unrelated process was signaled")
        finally:
            foreign.terminate()
            foreign.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
