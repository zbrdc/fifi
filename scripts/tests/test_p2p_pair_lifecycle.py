"""CPU-only lifecycle tests: no real lease, service, GPU, or HTTP access."""

import unittest

import p2p_test_support  # noqa: F401 -- make repository packages visible under python -I

from scripts.p2p_pair_host.lifecycle import Admission, PairLifecycle, UnitState
from scripts.p2p_pair_host.observer import PairSnapshot, ProcessIdentity

GPU0 = "GPU-704fef60-5280-9b88-fa3d-d11135caf576"
GPU1 = "GPU-282e266a-24ac-6fa8-a16d-4281e52b4c0a"
OLD_PID = 111
NEW_PID = 222
UNIT = "fifi-qwen36-engine.service"


def admission():
    process = ProcessIdentity(OLD_PID, 123, "/pair", "/engine", "a", "b", "c")
    snapshot = PairSnapshot(
        "boot",
        process,
        "/unit",
        "d",
        "e",
        (),
        ((GPU0, OLD_PID), (GPU1, OLD_PID)),
        ("123",),
    )
    return Admission(
        UNIT,
        snapshot,
        ((GPU0, 9000), (GPU1, 8000)),
        ((GPU0, 12288), (GPU1, 10240)),
        True,
    )


def active(pid=OLD_PID):
    return UnitState("active", pid, "", False, False, ())


def stopped(free=((GPU0, 11000), (GPU1, 9500))):
    return UnitState("inactive", 0, "", True, True, free)


class FakeLease:
    def __init__(self):
        self.valid = True
        self.closed = False
        self.checks = 0

    def verify(self, *, deadline):
        self.checks += 1
        return self.valid

    def close(self):
        self.closed = True


class FakeBackend:
    test_backend_only = True

    def __init__(self):
        self.view = active()
        self.secondary = True
        self.current = True
        self.queue = True
        self.inputs = True
        self.restored = True
        self.start_provenance = True
        self.start_provenance_checks = 0
        self.after_start_proof = None
        self.match_calls = 0
        self.restore_calls = []
        self.after_match = None
        self.after_queue = None
        self.after_inputs = None

    def secondary_unchanged(self, *, deadline):
        return self.secondary

    def current_pair_matches(self, snapshot, *, deadline):
        self.match_calls += 1
        if self.after_match:
            self.after_match(self.match_calls)
        return self.current

    def queue_ready_and_idle(self, *, deadline):
        if self.after_queue:
            self.after_queue()
        return self.queue

    def observe_pair(self, *, deadline):
        return self.view

    def restored_pair_matches(self, snapshot, *, allow_replacement, deadline):
        self.restore_calls.append(allow_replacement)
        return self.restored

    def started_pair_from_attempt(self, snapshot, *, deadline):
        self.start_provenance_checks += 1
        if self.after_start_proof:
            self.after_start_proof(self.start_provenance_checks)
        return self.start_provenance

    def restore_inputs_match(self, snapshot, *, deadline):
        if self.after_inputs:
            self.after_inputs()
        return self.inputs


class FakeRunner:
    test_runner_only = True

    def __init__(self, backend):
        self.backend = backend
        self.commands = []
        self.on_stop = None
        self.on_start = None
        self.stop_exit = 0
        self.start_exit = 0

    def run(self, argv, *, timeout, deadline):
        self.commands.append((argv, timeout))
        if argv == ["systemctl", "--user", "stop", UNIT]:
            if self.on_stop:
                self.on_stop()
                return self.stop_exit
            self.backend.view = stopped()
            return self.stop_exit
        elif argv == ["systemctl", "--user", "start", "--no-block", UNIT]:
            if self.on_start:
                self.on_start()
                return self.start_exit
            self.backend.view = active(NEW_PID)
            return self.start_exit
        else:
            raise AssertionError("unexpected command")


class FakeClock:
    def __init__(self):
        self.now = 0

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class PairLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.backend, self.lease, self.clock = FakeBackend(), FakeLease(), FakeClock()
        self.runner = FakeRunner(self.backend)
        self.lifecycle = PairLifecycle(
            admission(),
            self.lease,
            self.backend,
            self.runner,
            expected_gpu_uuids=(GPU0, GPU1),
            clock=self.clock.clock,
            sleep=self.clock.sleep,
        )

    def assert_retained(self):
        self.assertFalse(self.lease.closed)

    def test_stop_and_restart_use_only_exact_pair_unit_and_keep_lease(self):
        self.lifecycle.stop()
        self.assertEqual(self.lifecycle.phase, "stopped")
        self.lifecycle.restore()
        self.assertEqual(self.lifecycle.phase, "restored")
        self.assertEqual(
            self.runner.commands,
            [
                (["systemctl", "--user", "stop", UNIT], 45),
                (["systemctl", "--user", "start", "--no-block", UNIT], 10),
            ],
        )
        self.assertEqual(self.backend.restore_calls, [True, True])
        self.assert_retained()

    def test_no_live_backend_or_runner_can_be_used(self):
        self.backend.test_backend_only = False
        with self.assertRaisesRegex(RuntimeError, "live pair lifecycle"):
            PairLifecycle(
                admission(),
                self.lease,
                self.backend,
                self.runner,
                expected_gpu_uuids=(GPU0, GPU1),
            )
        self.backend.test_backend_only = True
        self.runner.test_runner_only = False
        with self.assertRaisesRegex(RuntimeError, "live systemd runner"):
            PairLifecycle(
                admission(),
                self.lease,
                self.backend,
                self.runner,
                expected_gpu_uuids=(GPU0, GPU1),
            )

    def test_admission_and_stop_guards_fail_before_commands(self):
        for field, value in (
            ("secondary", False),
            ("current", False),
            ("queue", False),
            ("current", 1),
            ("queue", 1),
        ):
            with self.subTest(field=field, value=value):
                backend, lease = FakeBackend(), FakeLease()
                setattr(backend, field, value)
                runner = FakeRunner(backend)
                lifecycle = PairLifecycle(
                    admission(), lease, backend, runner, expected_gpu_uuids=(GPU0, GPU1)
                )
                with self.assertRaises(RuntimeError):
                    lifecycle.stop()
                self.assertEqual(runner.commands, [])
                self.assertFalse(lease.closed)
        self.lease.valid = False
        with self.assertRaises(RuntimeError):
            self.lifecycle.stop()
        self.assertEqual(self.runner.commands, [])
        self.assert_retained()

    def test_identity_rechecked_after_readiness_before_stop(self):
        self.backend.after_match = lambda calls: setattr(
            self.backend, "current", calls == 1
        )
        with self.assertRaisesRegex(RuntimeError, "before stop"):
            self.lifecycle.stop()
        self.assertEqual(self.runner.commands, [])
        self.assert_retained()

    def test_prestop_failure_can_confirm_original_without_start(self):
        self.backend.queue = False
        with self.assertRaisesRegex(RuntimeError, "readiness"):
            self.lifecycle.stop()
        self.backend.queue = True
        self.lifecycle.restore()
        self.assertEqual(self.lifecycle.phase, "restored")
        self.assertEqual(self.runner.commands, [])
        self.assert_retained()

    def test_unexpected_stop_before_our_handoff_is_not_restarted(self):
        self.backend.view = stopped()
        with self.assertRaisesRegex(RuntimeError, "without our handoff"):
            self.lifecycle.restore()
        self.assertEqual(self.runner.commands, [])
        self.assert_retained()

    def test_partial_stop_and_signal_leave_recovery_possible(self):
        def partial():
            self.backend.view = stopped()
            raise KeyboardInterrupt()

        self.runner.on_stop = partial
        with self.assertRaises(KeyboardInterrupt):
            self.lifecycle.stop()
        self.assertEqual(self.lifecycle.phase, "uncertain")
        self.assert_retained()
        self.lifecycle.restore()
        self.assertEqual(self.lifecycle.phase, "restored")
        self.assert_retained()

    def test_timeout_does_not_release_lease(self):
        self.runner.on_stop = lambda: None
        with self.assertRaises(TimeoutError):
            self.lifecycle.stop(timeout=3)
        self.assertEqual(self.lifecycle.phase, "uncertain")
        self.assert_retained()

    def test_stop_requires_empty_contexts_sockets_and_memory_margin(self):
        for view in (
            UnitState("inactive", 0, "", False, True, stopped().free_mib),
            UnitState("inactive", 0, "", True, False, stopped().free_mib),
            UnitState("inactive", 123, "", True, True, stopped().free_mib),
            stopped(((GPU0, 100), (GPU1, 9500))),
            stopped(((GPU0, 11000),)),
            stopped(((GPU0, 11000), (GPU0, 11000))),
        ):
            with self.subTest(view=view):
                backend, lease, clock = FakeBackend(), FakeLease(), FakeClock()
                runner = FakeRunner(backend)
                runner.on_stop = lambda: setattr(backend, "view", view)
                lifecycle = PairLifecycle(
                    admission(),
                    lease,
                    backend,
                    runner,
                    expected_gpu_uuids=(GPU0, GPU1),
                    clock=clock.clock,
                    sleep=clock.sleep,
                )
                with self.assertRaises(RuntimeError):
                    lifecycle.stop()
                self.assertEqual(lifecycle.phase, "uncertain")
                self.assertFalse(lease.closed)

    def test_drift_before_restore_refuses_start(self):
        self.lifecycle.stop()
        self.backend.inputs = False
        with self.assertRaisesRegex(RuntimeError, "restoration inputs"):
            self.lifecycle.restore()
        self.assertEqual(len(self.runner.commands), 1)
        self.assert_retained()

    def test_inputs_rechecked_immediately_before_start(self):
        self.lifecycle.stop()
        count = 0

        def drift():
            nonlocal count
            count += 1
            if count == 2:
                self.backend.inputs = False

        self.backend.after_inputs = drift
        with self.assertRaisesRegex(RuntimeError, "changed before start"):
            self.lifecycle.restore()
        self.assertEqual(len(self.runner.commands), 1)
        self.assert_retained()

    def test_unexpected_replacement_before_our_start_is_refused(self):
        self.lifecycle.stop()
        self.backend.view = active(NEW_PID)
        with self.assertRaisesRegex(RuntimeError, "replaced before"):
            self.lifecycle.restore()
        self.assertEqual(len(self.runner.commands), 1)
        self.assert_retained()

    def test_failed_start_does_not_repeat_or_release(self):
        self.lifecycle.stop()

        def partial():
            raise OSError("injected")

        self.runner.on_start = partial
        with self.assertRaises(OSError):
            self.lifecycle.restore()
        self.assertEqual(self.lifecycle.phase, "uncertain")
        with self.assertRaisesRegex(RuntimeError, "prior start outcome"):
            self.lifecycle.restore()
        self.assertEqual(len(self.runner.commands), 2)
        self.assert_retained()

    def test_partial_start_can_be_verified_without_a_second_start(self):
        self.lifecycle.stop()

        def partial():
            self.backend.view = active(NEW_PID)
            raise OSError("injected")

        self.runner.on_start = partial
        with self.assertRaises(OSError):
            self.lifecycle.restore()
        self.lifecycle.restore()
        self.assertEqual(self.lifecycle.phase, "restored")
        self.assertEqual(len(self.runner.commands), 2)
        self.assert_retained()

    def test_replacement_requires_independent_start_provenance(self):
        self.lifecycle.stop()
        self.backend.start_provenance = False
        with self.assertRaisesRegex(RuntimeError, "start provenance"):
            self.lifecycle.restore()
        self.assertEqual(self.lifecycle.phase, "uncertain")
        self.assert_retained()

    def test_replacement_provenance_is_rechecked_after_readiness(self):
        self.lifecycle.stop()

        def drift(count):
            if count == 2:
                self.backend.start_provenance = False

        self.backend.after_start_proof = drift
        with self.assertRaisesRegex(RuntimeError, "provenance changed"):
            self.lifecycle.restore()
        self.assertEqual(self.lifecycle.phase, "uncertain")
        self.assert_retained()

    def test_nonzero_stop_and_start_status_are_never_success(self):
        self.runner.stop_exit = 1
        with self.assertRaisesRegex(RuntimeError, "stop command"):
            self.lifecycle.stop()
        self.assertEqual(self.lifecycle.phase, "uncertain")
        self.assert_retained()
        self.runner.start_exit = 1
        with self.assertRaisesRegex(RuntimeError, "start command"):
            self.lifecycle.restore()
        self.assertEqual(self.lifecycle.phase, "uncertain")
        self.assert_retained()

    def test_command_deadline_includes_prior_checks_and_command_runtime(self):
        self.backend.after_queue = lambda: setattr(
            self.clock, "now", self.clock.now + 2
        )
        self.runner.on_stop = lambda: setattr(self.clock, "now", self.clock.now + 1.1)
        with self.assertRaises(TimeoutError):
            self.lifecycle.stop(timeout=3)
        self.assertEqual(self.runner.commands[0][1], 1)
        self.assertEqual(self.lifecycle.phase, "uncertain")
        self.assert_retained()

    def test_impossible_total_and_wrong_expected_gpu_are_refused(self):
        bad = admission()
        bad = Admission(
            bad.unit, bad.snapshot, bad.used_mib, ((GPU0, 9500), (GPU1, 10240)), True
        )
        with self.assertRaisesRegex(ValueError, "exceeds GPU total"):
            PairLifecycle(
                bad,
                self.lease,
                self.backend,
                self.runner,
                expected_gpu_uuids=(GPU0, GPU1),
            )
        with self.assertRaisesRegex(ValueError, "ownership"):
            PairLifecycle(
                admission(),
                self.lease,
                self.backend,
                self.runner,
                expected_gpu_uuids=(GPU0, "GPU-unexpected"),
            )

    def test_restore_free_floor_is_one_gib_above_prior_used(self):
        exact = stopped(((GPU0, 10024), (GPU1, 9024)))
        self.assertTrue(self.lifecycle._stopped(exact))
        for free in (
            ((GPU0, 10023), (GPU1, 9024)),
            ((GPU0, 10024), (GPU1, 9023)),
            ((GPU0, 12289), (GPU1, 9024)),
        ):
            with self.subTest(free=free), self.assertRaises(RuntimeError):
                self.lifecycle._stopped(stopped(free))

    def test_active_service_with_pid_zero_is_never_restored(self):
        self.lifecycle.stop()
        self.backend.view = active(0)
        with self.assertRaisesRegex(RuntimeError, "invalid pair state"):
            self.lifecycle.restore()
        self.assertEqual(self.lifecycle.phase, "uncertain")
        self.assert_retained()

    def test_identity_or_queue_failure_after_start_never_claims_restored(self):
        self.lifecycle.stop()
        self.backend.restored = False
        with self.assertRaisesRegex(RuntimeError, "identity"):
            self.lifecycle.restore()
        self.assertEqual(self.lifecycle.phase, "uncertain")
        self.assert_retained()
        self.backend.restored = True
        self.backend.queue = False
        with self.assertRaisesRegex(RuntimeError, "readiness"):
            self.lifecycle.restore()
        self.assertEqual(self.lifecycle.phase, "uncertain")
        self.assert_retained()

    def test_unmodified_original_service_can_be_verified_without_start(self):
        self.runner.on_stop = lambda: (_ for _ in ()).throw(OSError("before stop"))
        with self.assertRaises(OSError):
            self.lifecycle.stop()
        self.lifecycle.restore()
        self.assertEqual(self.lifecycle.phase, "restored")
        self.assertEqual(self.backend.restore_calls, [False, False])
        self.assertEqual(len(self.runner.commands), 1)
        self.assert_retained()


if __name__ == "__main__":
    unittest.main()
