"""CPU-only fake manager/filesystem checks; no systemd or GPU calls."""

import errno
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import p2p_test_support  # noqa: F401 -- make repository packages visible under python -I

from scripts.p2p_pair_host.scoped_unit import ScopedUnit, ScopeUncertain

UNIT = "fifi-test-123.scope"
INVOCATION = "a" * 32
CGROUP = "/user.slice/user-42.slice/user@42.service/app.slice/" + UNIT


class Manager:
    def __init__(self):
        self.fields = {
            "Id": UNIT,
            "LoadState": "loaded",
            "ActiveState": "active",
            "InvocationID": INVOCATION,
            "ControlGroup": CGROUP,
            "Delegate": "no",
        }
        self.signals = []
        self.replies = []

    def __call__(self, argv, timeout):
        if argv[2] == "kill":
            self.signals.append(argv)
            return ""
        value = self.replies.pop(0) if self.replies else self.fields
        return "".join(f"{key}={item}\n" for key, item in value.items())


class ScopedUnitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.group = self.root / CGROUP.lstrip("/")
        self.group.mkdir(parents=True)
        (self.group / "cgroup.events").write_text("populated 1\n")
        self.manager = Manager()
        self.scope = ScopedUnit(
            UNIT,
            uid=42,
            cgroup_root=self.root,
            command=self.manager,
            clock=lambda: 10.0,
        )

    def test_pin_and_same_inactive_empty_scope(self):
        identity = self.scope.pin(20)
        self.assertEqual(identity.invocation, INVOCATION)
        self.assertFalse(self.scope.drained(20))
        self.manager.fields["ActiveState"] = "inactive"
        (self.group / "cgroup.events").write_text("populated 0\n")
        self.assertTrue(self.scope.drained(20))

    def test_removed_scope_requires_unit_not_found_and_group_absent(self):
        self.scope.pin(20)
        (self.group / "cgroup.events").unlink()
        self.group.rmdir()
        self.manager.fields.update(
            LoadState="not-found",
            ActiveState="inactive",
            InvocationID="",
            ControlGroup="",
            Delegate="",
        )
        self.assertTrue(self.scope.drained(20))

    def test_removed_scope_with_omitted_properties_still_requires_absent_group(self):
        self.scope.pin(20)
        self.manager.fields = {"Id": UNIT, "LoadState": "not-found"}
        self.assertFalse(self.scope.drained(20))
        (self.group / "cgroup.events").unlink()
        self.group.rmdir()
        self.assertTrue(self.scope.drained(20))

    def test_not_found_rejects_missing_id_or_contradictory_properties(self):
        self.scope.pin(20)
        for fields in (
            {"LoadState": "not-found"},
            {"Id": UNIT, "LoadState": "not-found", "InvocationID": "b" * 32},
            {"Id": UNIT, "LoadState": "not-found", "ControlGroup": CGROUP},
            {"Id": UNIT, "LoadState": "not-found", "ActiveState": "active"},
        ):
            with self.subTest(fields=fields):
                self.manager.fields = fields
                with self.assertRaises(ScopeUncertain):
                    self.scope.drained(20)

    def test_changed_or_recreated_invocation_never_signalled(self):
        self.scope.pin(20)
        self.manager.fields["InvocationID"] = "b" * 32
        with self.assertRaises(ScopeUncertain):
            self.scope.signal("TERM", 20)
        self.assertEqual(self.manager.signals, [])

    def test_signal_is_only_issued_after_exact_identity_check(self):
        self.scope.pin(20)
        self.scope.signal("TERM", 20)
        self.assertEqual(len(self.manager.signals), 1)
        self.assertIn("--signal=TERM", self.manager.signals[0])
        with self.assertRaises(ValueError):
            self.scope.signal("HUP", 20)

    def test_recreated_scope_during_drain_is_inconclusive(self):
        self.scope.pin(20)
        first = dict(self.manager.fields, ActiveState="inactive")
        second = dict(first, InvocationID="b" * 32)
        self.manager.replies = [first, second]
        (self.group / "cgroup.events").write_text("populated 0\n")
        with self.assertRaises(ScopeUncertain):
            self.scope.drained(20)

    def test_permission_or_device_error_does_not_prove_absence(self):
        self.scope.pin(20)
        for code in (errno.EACCES, errno.ENODEV):
            with (
                self.subTest(code=code),
                patch.object(
                    Path, "read_text", side_effect=OSError(errno.ENODEV, "events")
                ),
                patch.object(Path, "stat", side_effect=OSError(code, "ambiguous")),
            ):
                with self.assertRaises(ScopeUncertain):
                    self.scope._population()

    def test_malformed_or_unbounded_inputs_fail_closed(self):
        with self.assertRaises(ValueError):
            ScopedUnit(
                "other.scope", uid=42, cgroup_root=self.root, command=self.manager
            )
        with self.assertRaises(ScopeUncertain):
            self.scope.pin(10)
        self.manager.fields = {"Id": UNIT, "LoadState": "not-found"}
        with self.assertRaises(ScopeUncertain):
            self.scope.pin(20)
        self.manager.fields = Manager().fields
        self.manager.fields["Delegate"] = "yes"
        with self.assertRaises(ScopeUncertain):
            self.scope.pin(20)


if __name__ == "__main__":
    unittest.main()
