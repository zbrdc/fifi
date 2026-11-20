"""CPU-only contract tests; no service, socket, or CUDA operations."""

import unittest

import p2p_test_support  # noqa: F401 -- make repository packages visible under python -I

from scripts.p2p_concurrent_guard.core import Interval, intervals_overlap, run


class FakeAdapter:
    def __init__(self, fail=None, false=None, intervals=None):
        self.calls = []
        self.fail = fail
        self.false = false
        self.intervals = intervals or (Interval(10, 30), Interval(20, 40))

    def call(self, name, result=None):
        self.calls.append(name)
        if self.fail == name:
            raise RuntimeError("secret must not appear")
        if self.false == name:
            return False
        return result

    def acquire(self):
        return self.call("acquire")

    def verify_lease(self):
        return self.call("verify_lease")

    def release(self):
        return self.call("release")

    def snapshot_and_admit(self):
        return self.call("snapshot_and_admit")

    def verify_secondary(self):
        return self.call("verify_secondary")

    def stop_pair(self):
        return self.call("stop_pair")

    def verify_pair_stopped(self):
        return self.call("verify_pair_stopped")

    def start_pair_scope(self):
        return self.call("start_pair_scope")

    def start_client_scope(self):
        return self.call("start_client_scope")

    def wait_with_monitor(self):
        return self.call("wait_with_monitor", self.intervals)

    def drain_client_scope(self):
        return self.call("drain_client_scope", True)

    def drain_pair_scope(self):
        return self.call("drain_pair_scope", True)

    def restore_pair(self):
        return self.call("restore_pair")

    def verify_pair_restored(self):
        return self.call("verify_pair_restored", True)

    def verify_secondary_idle(self):
        return self.call("verify_secondary_idle", True)


class ConcurrentGuardTests(unittest.TestCase):
    def test_requires_explicit_boolean_opt_in_and_does_nothing(self):
        for value in (False, None, 1, "true"):
            with self.subTest(value=value):
                fake = FakeAdapter()
                self.assertEqual(run(fake, concurrent=value).status, "disabled")
                self.assertEqual(fake.calls, [])

    def test_valid_overlap_and_cleanup_order(self):
        fake = FakeAdapter()
        outcome = run(fake, concurrent=True)
        self.assertEqual((outcome.status, outcome.lease_retained), ("passed", False))
        self.assertLess(
            fake.calls.index("drain_client_scope"), fake.calls.index("restore_pair")
        )
        self.assertLess(
            fake.calls.index("drain_pair_scope"), fake.calls.index("restore_pair")
        )
        self.assertLess(
            fake.calls.index("verify_pair_restored"), fake.calls.index("release")
        )
        self.assertLess(
            fake.calls.index("verify_secondary_idle"), fake.calls.index("release")
        )

    def test_touching_intervals_and_invalid_intervals_do_not_overlap(self):
        self.assertFalse(intervals_overlap(Interval(1, 2), Interval(2, 3)))
        for invalid in (
            Interval(2, 2),
            Interval(-1, 2),
            Interval(True, 2),
            Interval(3, 2),
        ):
            with self.subTest(invalid=invalid):
                self.assertFalse(intervals_overlap(invalid, Interval(1, 4)))

    def test_nonoverlap_fails_but_cleans_up(self):
        fake = FakeAdapter(intervals=(Interval(1, 2), Interval(2, 3)))
        outcome = run(fake, concurrent=True)
        self.assertEqual((outcome.status, outcome.lease_retained), ("failed", False))
        self.assertEqual(fake.calls[-1], "release")

    def test_partially_started_scopes_are_drained_on_failure(self):
        for failure, required in (
            ("start_pair_scope", "drain_pair_scope"),
            ("start_client_scope", "drain_client_scope"),
        ):
            with self.subTest(failure=failure):
                fake = FakeAdapter(fail=failure)
                outcome = run(fake, concurrent=True)
                self.assertEqual(outcome.status, "failed")
                self.assertIn(required, fake.calls)
                self.assertNotIn("secret", outcome.reason)
                self.assertIn("restore_pair", fake.calls)

    def test_uncertain_cleanup_retains_lease_and_skips_restore(self):
        for failure in ("drain_client_scope", "drain_pair_scope"):
            for kind in ("fail", "false"):
                with self.subTest(failure=failure, kind=kind):
                    fake = FakeAdapter(**{kind: failure})
                    outcome = run(fake, concurrent=True)
                    self.assertEqual(
                        (outcome.status, outcome.lease_retained), ("inconclusive", None)
                    )
                    self.assertNotIn("restore_pair", fake.calls)
                    self.assertNotIn("release", fake.calls)

    def test_uncertain_restoration_or_idle_retains_lease(self):
        for failure in ("verify_pair_restored", "verify_secondary_idle"):
            for kind in ("fail", "false"):
                with self.subTest(failure=failure, kind=kind):
                    fake = FakeAdapter(**{kind: failure})
                    outcome = run(fake, concurrent=True)
                    self.assertEqual(
                        (outcome.status, outcome.lease_retained), ("inconclusive", None)
                    )
                    self.assertNotIn("release", fake.calls)

    def test_failure_to_verify_secondary_retains_after_pair_handoff(self):
        fake = FakeAdapter()
        original = fake.verify_secondary

        def fail_late():
            if fake.calls.count("verify_secondary") >= 2:
                raise RuntimeError("identity changed")
            return original()

        fake.verify_secondary = fail_late
        outcome = run(fake, concurrent=True)
        self.assertEqual(
            (outcome.status, outcome.lease_retained), ("inconclusive", None)
        )
        self.assertNotIn("release", fake.calls)

    def test_release_failure_is_inconclusive(self):
        fake = FakeAdapter(fail="release")
        outcome = run(fake, concurrent=True)
        self.assertEqual(
            (outcome.status, outcome.lease_retained), ("inconclusive", None)
        )

    def test_partial_stop_failure_attempts_restoration(self):
        for failure in ("stop_pair", "verify_pair_stopped"):
            with self.subTest(failure=failure):
                fake = FakeAdapter(fail=failure)
                outcome = run(fake, concurrent=True)
                self.assertEqual(outcome.status, "failed")
                self.assertIn("restore_pair", fake.calls)
                self.assertIn("verify_pair_restored", fake.calls)
                self.assertNotIn("start_pair_scope", fake.calls)

    def test_rejected_admission_releases_without_handoff(self):
        fake = FakeAdapter(fail="snapshot_and_admit")
        outcome = run(fake, concurrent=True)
        self.assertEqual(outcome.status, "failed")
        self.assertNotIn("stop_pair", fake.calls)
        self.assertNotIn("restore_pair", fake.calls)
        self.assertEqual(fake.calls[-1], "release")


if __name__ == "__main__":
    unittest.main()
