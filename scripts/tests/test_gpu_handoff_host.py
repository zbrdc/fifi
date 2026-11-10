"""CPU-only admission and restoration tests for the 2080 handoff adapter.

All host-facing calls are replaced with mocks: these tests never contact
systemd, CUDA, sockets or a live service.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from gpu_guarded_handoff import host as adapter  # noqa: E402


class HostAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.host = adapter.Host2080(min_free_mib=2048, restore_timeout=0)
        self.scope_patch = mock.patch.object(adapter.scope, "check_manager")
        self.sleep_patch = mock.patch.object(adapter.time, "sleep")
        self.samefile_patch = mock.patch.object(
            adapter.os.path, "samefile", side_effect=self.fake_samefile
        )
        self.inode_patch = mock.patch.object(
            adapter, "inode", side_effect=self.fake_inode
        )
        self.attested_patch = mock.patch.object(
            adapter, "attested_execstart", return_value=True
        )
        self.scope_patch.start()
        self.sleep_patch.start()
        self.samefile_patch.start()
        self.inode_patch.start()
        self.attested_patch.start()
        self.addCleanup(self.scope_patch.stop)
        self.addCleanup(self.sleep_patch.stop)
        self.addCleanup(self.samefile_patch.stop)
        self.addCleanup(self.inode_patch.stop)
        self.addCleanup(self.attested_patch.stop)
        self.target = adapter.TARGET_GPU
        self.pair = sorted(adapter.PAIR_GPUS)
        self.snap = {
            "pid": 42,
            "starttime": 100,
            "exe": "/example/owner/enginesrv",
            "files": {
                "/example/owner/enginesrv": "a" * 64,
                "/example/owner/start.sh": "b" * 64,
                "/example/owner/config.json": "c" * 64,
            },
            "model": "fifi-q36-2080ti",
            "context": 65536,
            "primary_pid": 77,
            "primary_starttime": 200,
            "primary_exe": "/example/primary/enginesrv",
            "exe_inode": (1, 1000),
            "primary_inode": (1, 2000),
        }
        self.states = {
            adapter.UNIT: {
                "ActiveState": "active",
                "MainPID": "42",
                "Job": "",
                "FragmentPath": "/fake/secondary.service",
                "DropInPaths": "",
                "ExecStart": "{ path=/fake/launcher ; argv[]=/fake/launcher ; start_time=old ; }",
                "NeedDaemonReload": "no",
            },
            adapter.PRIMARY: {
                "ActiveState": "active",
                "MainPID": "77",
                "Job": "",
                "FragmentPath": "/fake/primary.service",
                "DropInPaths": "/fake/old.conf",
                "ExecStart": "{ path=/fake/primary ; argv[]=/fake/primary ; start_time=old ; }",
                "NeedDaemonReload": "no",
            },
        }
        self.snap["unit_meta"] = adapter.unit_metadata(self.states[adapter.UNIT])
        self.snap["primary_meta"] = adapter.unit_metadata(self.states[adapter.PRIMARY])
        self.devices = [(self.target, 42)] + [(gpu, 77) for gpu in self.pair]
        self.identities = {
            42: (100, "/example/owner/enginesrv"),
            77: (200, "/example/primary/enginesrv"),
        }
        self.host.state = mock.Mock(
            side_effect=lambda unit: deepcopy(self.states[unit])
        )
        self.host.processes = mock.Mock(side_effect=lambda: list(self.devices))
        self.host.active_clients = mock.Mock(return_value=False)
        self.host.port_open = mock.Mock(return_value=False)
        self.host.listening = mock.Mock(return_value=False)
        self.host.ready = mock.Mock(return_value=True)
        self.host.quiet = mock.Mock(return_value=True)
        self.host.inspect_files = mock.Mock(return_value=True)
        self.host.device_free_mib = mock.Mock(return_value=4096)
        self.host.admission_memory_possible = mock.Mock(return_value=True)
        self.host.competing = mock.Mock(return_value=False)
        self.host.live_identity = mock.Mock(
            side_effect=lambda pid: self.identities[pid]
        )
        self.host.uses_config = mock.Mock(return_value=True)
        self.host.environment_matches = mock.Mock(return_value=True)
        self.host.attestation_matches = mock.Mock(return_value=True)

    def fake_path(self, path):
        path = str(path)
        if path.startswith("/proc/") and path.endswith("/exe"):
            return self.identities[int(path.split("/")[2])][1]
        return path

    def fake_samefile(self, one, two):
        return self.fake_path(one) == self.fake_path(two)

    def fake_inode(self, path):
        resolved = self.fake_path(path)
        if resolved == "/example/owner/enginesrv":
            return (1, 1000)
        if resolved == "/example/primary/enginesrv":
            return (1, 2000)
        return (1, 9999)

    def test_valid_captured_service_is_admitted_without_service_action(self):
        with (
            mock.patch.object(self.host, "stop") as stop,
            mock.patch.object(self.host, "restore") as restore,
        ):
            self.host.admit(self.snap)
            stop.assert_not_called()
            restore.assert_not_called()

    def test_active_client_blocks_before_stop(self):
        self.host.active_clients.return_value = True
        with mock.patch.object(self.host, "stop") as stop:
            with self.assertRaises((ValueError, RuntimeError)):
                self.host.admit(self.snap)
            stop.assert_not_called()

    def test_busy_or_other_assistant_blocks_before_stop(self):
        self.host.competing.return_value = True
        with mock.patch.object(self.host, "stop") as stop:
            with self.assertRaises((ValueError, RuntimeError)):
                self.host.admit(self.snap)
            stop.assert_not_called()

    def test_impossible_declared_memory_blocks_before_stop(self):
        self.host.admission_memory_possible.return_value = False
        with mock.patch.object(self.host, "stop") as stop:
            with self.assertRaises((ValueError, RuntimeError)):
                self.host.admit(self.snap)
            stop.assert_not_called()

    def test_wrong_primary_gpu_owner_blocks_before_stop(self):
        self.devices = [(self.target, 42), (self.pair[0], 77), (self.pair[1], 99)]
        with mock.patch.object(self.host, "stop") as stop:
            with self.assertRaises((ValueError, RuntimeError)):
                self.host.admit(self.snap)
            stop.assert_not_called()

    def test_wrong_2080_gpu_owner_blocks_before_stop(self):
        self.devices = [(self.target, 99)] + [(gpu, 77) for gpu in self.pair]
        with mock.patch.object(self.host, "stop") as stop:
            with self.assertRaises((ValueError, RuntimeError)):
                self.host.admit(self.snap)
            stop.assert_not_called()

    def test_alternate_live_executable_blocks_before_stop(self):
        self.identities[42] = (100, "/example/other/enginesrv")
        with mock.patch.object(self.host, "stop") as stop:
            with self.assertRaises((ValueError, RuntimeError)):
                self.host.admit(self.snap)
            stop.assert_not_called()

    def test_other_live_config_blocks_before_stop(self):
        self.host.uses_config.return_value = False
        with mock.patch.object(self.host, "stop") as stop:
            with self.assertRaises((ValueError, RuntimeError)):
                self.host.admit(self.snap)
            stop.assert_not_called()

    def test_changed_live_environment_blocks_before_stop(self):
        self.host.environment_matches.return_value = False
        with mock.patch.object(self.host, "stop") as stop:
            with self.assertRaises((ValueError, RuntimeError)):
                self.host.admit(self.snap)
            stop.assert_not_called()

    def test_missing_live_attestation_blocks_before_stop(self):
        self.host.attestation_matches.return_value = False
        with mock.patch.object(self.host, "stop") as stop:
            with self.assertRaises((ValueError, RuntimeError)):
                self.host.admit(self.snap)
            stop.assert_not_called()

    def test_wrong_unit_entrypoint_blocks_before_stop(self):
        adapter.attested_execstart.return_value = False
        with mock.patch.object(self.host, "stop") as stop:
            with self.assertRaises((ValueError, RuntimeError)):
                self.host.admit(self.snap)
            stop.assert_not_called()

    def test_loaded_unit_metadata_change_refuses_before_stop(self):
        changes = (
            (adapter.UNIT, "FragmentPath", "/fake/other.service"),
            (adapter.UNIT, "DropInPaths", "/fake/new.conf"),
            (adapter.UNIT, "ExecStart", "{ path=/fake/other ; argv[]=/fake/other ; }"),
            (adapter.PRIMARY, "FragmentPath", "/fake/other-primary.service"),
            (adapter.PRIMARY, "DropInPaths", "/fake/old.conf /fake/new.conf"),
            (
                adapter.PRIMARY,
                "ExecStart",
                "{ path=/fake/other ; argv[]=/fake/other ; }",
            ),
        )
        for unit, key, altered in changes:
            with self.subTest(unit=unit, field=key):
                original = self.states[unit][key]
                self.states[unit][key] = altered
                try:
                    with mock.patch.object(self.host, "stop") as stop:
                        with self.assertRaises((ValueError, RuntimeError)):
                            self.host.admit(self.snap)
                        stop.assert_not_called()
                finally:
                    self.states[unit][key] = original

    def test_loaded_unit_metadata_change_refuses_before_restart(self):
        self.states[adapter.UNIT]["ActiveState"] = "inactive"
        self.states[adapter.UNIT]["MainPID"] = "0"
        self.devices = [(gpu, 77) for gpu in self.pair]
        changes = (
            (adapter.UNIT, "FragmentPath", "/fake/other.service"),
            (adapter.UNIT, "DropInPaths", "/fake/new.conf"),
            (adapter.UNIT, "ExecStart", "{ path=/fake/other ; argv[]=/fake/other ; }"),
            (adapter.PRIMARY, "FragmentPath", "/fake/other-primary.service"),
            (adapter.PRIMARY, "DropInPaths", "/fake/old.conf /fake/new.conf"),
            (
                adapter.PRIMARY,
                "ExecStart",
                "{ path=/fake/other ; argv[]=/fake/other ; }",
            ),
        )
        for unit, key, altered in changes:
            with self.subTest(unit=unit, field=key):
                original = self.states[unit][key]
                self.states[unit][key] = altered
                try:
                    with mock.patch.object(adapter.subprocess, "run") as systemctl:
                        with self.assertRaises((ValueError, RuntimeError)):
                            self.host.restore(self.snap)
                        systemctl.assert_not_called()
                finally:
                    self.states[unit][key] = original

    def test_need_daemon_reload_refuses_before_stop_or_restart(self):
        self.states[adapter.UNIT]["NeedDaemonReload"] = "yes"
        with mock.patch.object(self.host, "stop") as stop:
            with self.assertRaises((ValueError, RuntimeError)):
                self.host.admit(self.snap)
            stop.assert_not_called()
        self.states[adapter.UNIT]["ActiveState"] = "inactive"
        self.states[adapter.UNIT]["MainPID"] = "0"
        self.devices = [(gpu, 77) for gpu in self.pair]
        with mock.patch.object(adapter.subprocess, "run") as systemctl:
            with self.assertRaises((ValueError, RuntimeError)):
                self.host.restore(self.snap)
            systemctl.assert_not_called()

    def test_process_timing_fields_do_not_change_pinned_unit_metadata(self):
        old = self.states[adapter.UNIT]["ExecStart"]
        self.states[adapter.UNIT]["ExecStart"] = old.replace(
            "start_time=old", "start_time=new"
        )
        self.assertEqual(
            adapter.unit_metadata(self.states[adapter.UNIT]), self.snap["unit_meta"]
        )
        self.assertTrue(self.host.still_matches(self.snap))

    def test_changed_pid_after_snapshot_rejected(self):
        self.states[adapter.UNIT]["MainPID"] = "43"
        self.assertFalse(self.host.still_matches(self.snap))

    def test_reused_pid_after_snapshot_rejected(self):
        self.identities[42] = (101, "/example/owner/enginesrv")
        self.assertFalse(self.host.still_matches(self.snap))

    def test_changed_original_files_rejected_before_stop(self):
        self.host.inspect_files.return_value = False
        self.assertFalse(self.host.still_matches(self.snap))
        with mock.patch.object(self.host, "stop") as stop:
            with self.assertRaises((ValueError, RuntimeError)):
                self.host.admit(self.snap)
            stop.assert_not_called()

    def test_not_ready_or_not_quiet_is_rejected(self):
        for method in (self.host.ready, self.host.quiet):
            with self.subTest(method=method):
                method.return_value = False
                with self.assertRaises((ValueError, RuntimeError)):
                    self.host.admit(self.snap)
                method.return_value = True

    def test_after_stop_requires_declared_device_memory(self):
        self.devices = [(gpu, 77) for gpu in self.pair]
        self.host.device_free_mib.return_value = 1024
        with self.assertRaises((ValueError, RuntimeError)):
            self.host.verify_stopped(self.snap)

    def test_after_stop_rechecks_competing_work_before_launch(self):
        self.devices = [(gpu, 77) for gpu in self.pair]
        self.host.competing.return_value = True
        with self.assertRaises((ValueError, RuntimeError)):
            self.host.verify_stopped(self.snap)

    def test_after_stop_rechecks_gpu_quiet_before_launch(self):
        self.devices = [(gpu, 77) for gpu in self.pair]
        self.host.quiet.return_value = False
        with self.assertRaises((ValueError, RuntimeError)):
            self.host.verify_stopped(self.snap)

    def test_after_stop_rejects_changed_service_files(self):
        self.devices = [(gpu, 77) for gpu in self.pair]
        self.host.inspect_files.return_value = False
        with self.assertRaises((ValueError, RuntimeError)):
            self.host.verify_stopped(self.snap)

    def test_after_stop_rejects_wrong_primary_gpu_ownership(self):
        self.devices = [(self.pair[0], 77), (self.pair[1], 99)]
        with self.assertRaises((ValueError, RuntimeError)):
            self.host.verify_stopped(self.snap)

    def test_original_service_was_active(self):
        self.assertTrue(self.host.was_active(self.snap))

    def test_restored_service_accepts_new_pid_only_with_same_binary_and_device(self):
        self.states[adapter.UNIT]["MainPID"] = "43"
        self.identities[43] = (300, "/example/owner/enginesrv")
        self.devices = [(self.target, 43)] + [(gpu, 77) for gpu in self.pair]
        self.assertTrue(self.host.verify_restored(self.snap))

    def test_wrong_restored_binary_is_not_accepted(self):
        self.states[adapter.UNIT]["MainPID"] = "43"
        self.identities[43] = (300, "/example/wrong/enginesrv")
        self.devices = [(self.target, 43)] + [(gpu, 77) for gpu in self.pair]
        self.assertFalse(self.host.verify_restored(self.snap))

    def test_wrong_restored_config_is_not_accepted(self):
        self.states[adapter.UNIT]["MainPID"] = "43"
        self.identities[43] = (300, "/example/owner/enginesrv")
        self.devices = [(self.target, 43)] + [(gpu, 77) for gpu in self.pair]
        self.host.uses_config.return_value = False
        self.assertFalse(self.host.verify_restored(self.snap))

    def test_changed_restored_environment_is_not_accepted(self):
        self.states[adapter.UNIT]["MainPID"] = "43"
        self.identities[43] = (300, "/example/owner/enginesrv")
        self.devices = [(self.target, 43)] + [(gpu, 77) for gpu in self.pair]
        self.host.environment_matches.return_value = False
        self.assertFalse(self.host.verify_restored(self.snap))

    def test_missing_restored_attestation_is_not_accepted(self):
        self.states[adapter.UNIT]["MainPID"] = "43"
        self.identities[43] = (300, "/example/owner/enginesrv")
        self.devices = [(self.target, 43)] + [(gpu, 77) for gpu in self.pair]
        self.host.attestation_matches.return_value = False
        self.assertFalse(self.host.verify_restored(self.snap))

    def test_wrong_restored_unit_entrypoint_is_not_accepted(self):
        self.states[adapter.UNIT]["MainPID"] = "43"
        self.identities[43] = (300, "/example/owner/enginesrv")
        self.devices = [(self.target, 43)] + [(gpu, 77) for gpu in self.pair]
        adapter.attested_execstart.return_value = False
        self.assertFalse(self.host.verify_restored(self.snap))

    def test_wrong_restored_gpu_owner_is_not_accepted(self):
        self.states[adapter.UNIT]["MainPID"] = "43"
        self.identities[43] = (300, "/example/owner/enginesrv")
        self.devices = [(self.target, 99)] + [(gpu, 77) for gpu in self.pair]
        self.assertFalse(self.host.verify_restored(self.snap))

    def test_wrong_restored_model_or_context_is_not_accepted(self):
        self.states[adapter.UNIT]["MainPID"] = "43"
        self.identities[43] = (300, "/example/owner/enginesrv")
        self.devices = [(self.target, 43)] + [(gpu, 77) for gpu in self.pair]
        self.host.ready.return_value = False
        self.assertFalse(self.host.verify_restored(self.snap))

    def test_changed_files_after_restart_are_not_accepted(self):
        self.states[adapter.UNIT]["MainPID"] = "43"
        self.identities[43] = (300, "/example/owner/enginesrv")
        self.devices = [(self.target, 43)] + [(gpu, 77) for gpu in self.pair]
        self.host.inspect_files.return_value = False
        self.assertFalse(self.host.verify_restored(self.snap))


if __name__ == "__main__":
    unittest.main()
