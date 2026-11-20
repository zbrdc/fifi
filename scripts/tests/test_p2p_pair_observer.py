"""CPU-only tests; all process, service, GPU and socket data are synthetic."""

from pathlib import Path
import tempfile
import unittest

import p2p_test_support  # noqa: F401 -- make repository packages visible under python -I

from scripts.p2p_pair_host.observer import LinuxPairObserver, PairSpec, _UNIT_PROPERTIES

GPU0 = "GPU-704fef60-5280-9b88-fa3d-d11135caf576"
GPU1 = "GPU-282e266a-24ac-6fa8-a16d-4281e52b4c0a"
OTHER = "GPU-699004f8-e2f0-747e-5c02-30141e9d4b7c"
BOOT = "7b2415a1-fab1-4aa0-b52b-c1bce142f145"
PID = 4321
PORT = 18791
SECRET = "unit-test-only-secret"


class FakeHost:
    def __init__(self, root):
        self.root = Path(root)
        self.proc = self.root / "proc"
        self.files = self.root / "inputs"
        self.files.mkdir()
        for name in ("launch", "engine", "config", "env", "key", "unit", "dropin"):
            (self.files / name).write_text(SECRET if name in ("env", "key") else name)
        (self.files / "launch").write_text(
            f'#!/bin/sh\nexec {self.files / "engine"} --config "$CONFIG"\n'
        )
        process = self.proc / str(PID)
        (process / "fd").mkdir(parents=True)
        (self.proc / "net").mkdir()
        (self.proc / "sys/kernel/random").mkdir(parents=True)
        (self.proc / "sys/kernel/random/boot_id").write_text(BOOT + "\n")
        self.start_ticks = 1000
        self.write_stat()
        (process / "cmdline").write_bytes(
            b"enginesrv\x00--config\x00" + str(self.files / "config").encode() + b"\x00"
        )
        (process / "environ").write_bytes(("KEY=" + SECRET + "\0").encode())
        (process / "cgroup").write_text("0::/user.slice/pair.service\n")
        (process / "exe").symlink_to(self.files / "engine")
        (process / "fd/9").symlink_to("socket:[12345]")
        self.write_sockets()
        self.spec = PairSpec(
            "fifi-qwen36-engine.service",
            self.files / "launch",
            self.files / "engine",
            self.files / "config",
            self.files / "env",
            self.files / "key",
            (GPU0, GPU1),
            PORT,
        )
        self.fields = {key: "" for key in _UNIT_PROPERTIES}
        self.fields.update(
            {
                "ActiveState": "active",
                "MainPID": str(PID),
                "Job": "",
                "NeedDaemonReload": "no",
                "ControlGroup": "/user.slice/pair.service",
                "FragmentPath": str(self.files / "unit"),
                "DropInPaths": str(self.files / "dropin"),
                "ExecStart": f"{{ path={self.files / 'launch'} ; argv[]={self.files / 'launch'} ; ignore_errors=no ; start_time=[x] }}",
                "Environment": "SECRET=" + SECRET,
            }
        )
        self.manager = ("MANAGER_KEY=" + SECRET).encode()
        self.gpu = f"{GPU0}, {PID}\n{GPU1}, {PID}\n{OTHER}, 55\n".encode()
        self.util = f"{GPU0}, 0\n{GPU1}, 0\n".encode()
        self.ss = b"LISTEN 0 128 127.0.0.1:18791 0.0.0.0:*\n"
        self.unit_object = b'o "/org/freedesktop/systemd1/unit/fake_2eservice"\n'
        self.dbus = {
            name: b"a(sasbttttuii) 0\n"
            for name in (
                "ExecStartPre",
                "ExecStartPost",
                "ExecCondition",
                "ExecStop",
                "ExecStopPost",
            )
        }
        self.dbus["EnvironmentFiles"] = b"a(sb) 0\n"
        self.calls = []
        self.on_command = None

    def write_stat(self):
        fields = ["S"] + ["0"] * 18 + [str(self.start_ticks)]
        (self.proc / str(PID) / "stat").write_text(
            f"{PID} (engine name) " + " ".join(fields)
        )

    def write_sockets(self, state="0A", inode="12345"):
        row = f"0: 0100007F:{PORT:04X} 00000000:0000 {state} 0:0 0:0 0 0 0 {inode}\n"
        (self.proc / "net/tcp").write_text("header\n" + row)
        (self.proc / "net/tcp6").write_text("header\n")

    def command(self, argv):
        self.calls.append(argv)
        if self.on_command:
            self.on_command(argv)
        if argv[:3] == ["systemctl", "--user", "show"]:
            return "".join(
                f"{key}={self.fields[key]}\n" for key in _UNIT_PROPERTIES
            ).encode()
        if argv == ["systemctl", "--user", "show-environment"]:
            return self.manager
        if argv[:3] == ["busctl", "--user", "call"]:
            return self.unit_object
        if argv[:3] == ["busctl", "--user", "get-property"]:
            return self.dbus[argv[-1]]
        if argv[:2] == ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid"]:
            return self.gpu
        if argv[:2] == ["nvidia-smi", "-i"]:
            return self.util
        if argv[:2] == ["ss", "-Htan"]:
            return self.ss
        raise AssertionError(f"unexpected command {argv}")

    def observer(self):
        return LinuxPairObserver(
            self.spec, proc=self.proc, command=self.command, sleep=lambda _: None
        )


class PairObserverTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="p2p-pair-observer-")
        self.addCleanup(self.tmp.cleanup)
        self.host = FakeHost(self.tmp.name)

    def test_stable_identity_is_opaque_and_all_queries_are_read_only(self):
        observer = self.host.observer()
        snapshot = observer.snapshot()
        self.assertTrue(observer.verify(snapshot))
        self.assertEqual(
            (snapshot.process.pid, snapshot.process.start_ticks), (PID, 1000)
        )
        self.assertNotIn(SECRET, repr(snapshot))
        self.assertTrue(
            all(
                cmd[0] in ("systemctl", "busctl", "nvidia-smi", "ss")
                for cmd in self.host.calls
            )
        )
        shows = [
            cmd for cmd in self.host.calls if cmd[:3] == ["systemctl", "--user", "show"]
        ]
        self.assertTrue(shows)
        self.assertTrue(all("--all" in cmd for cmd in shows))
        self.assertFalse(
            any("start" in cmd or "stop" in cmd for cmd in self.host.calls)
        )
        self.assertGreaterEqual(
            sum(cmd[:2] == ["nvidia-smi", "-i"] for cmd in self.host.calls), 6
        )

    def test_identity_config_and_environment_changes_refuse(self):
        for mutate in (
            lambda: (self.host.files / "config").write_text("changed"),
            lambda: (self.host.files / "key").write_text("new-secret"),
            lambda: (self.host.files / "engine").write_text("new-binary"),
            lambda: setattr(self.host, "manager", b"changed"),
            lambda: self.host.fields.update(Environment="different"),
            lambda: (self.host.proc / str(PID) / "environ").write_bytes(b"KEY=other\0"),
            lambda: (self.host.proc / str(PID) / "cmdline").write_bytes(b"other\0"),
            lambda: (self.host.proc / str(PID) / "cgroup").write_text("0::/other\n"),
        ):
            with self.subTest(mutation=mutate):
                with tempfile.TemporaryDirectory(prefix="pair-isolation-") as directory:
                    self.host = FakeHost(directory)
                    observer = self.host.observer()
                    pinned = observer.snapshot()
                    mutate()
                    with self.assertRaises((ValueError, RuntimeError, OSError)):
                        observer.verify(pinned)

    def test_pid_reuse_and_mid_observation_restart_are_rejected(self):
        observer = self.host.observer()
        pinned = observer.snapshot()
        self.host.start_ticks = 1001
        self.host.write_stat()
        with self.assertRaises(RuntimeError):
            observer.verify(pinned)
        self.host.start_ticks = 1000
        self.host.write_stat()
        changed = False

        def restart(argv):
            nonlocal changed
            if not changed and argv[:2] == [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
            ]:
                self.host.start_ticks = 1001
                self.host.write_stat()
                changed = True

        self.host.on_command = restart
        with self.assertRaisesRegex(RuntimeError, "changed during"):
            observer.snapshot()

    def test_unit_rejects_stale_pending_hooks_environment_files_and_wrong_launcher(
        self,
    ):
        for field, value in (
            ("ActiveState", "inactive"),
            ("Job", "42"),
            ("NeedDaemonReload", "yes"),
            ("EnvironmentFiles", "/tmp/env"),
            ("ExecStop", "/bin/true"),
            ("ExecStart", "unrecognized"),
            ("MainPID", "0"),
            ("ControlGroup", "/user.slice/unrelated.service"),
        ):
            with self.subTest(field=field):
                old = self.host.fields[field]
                self.host.fields[field] = value
                with self.assertRaises((ValueError, RuntimeError)):
                    self.host.observer().snapshot()
                self.host.fields[field] = old

    def test_launcher_target_must_match_running_binary(self):
        launcher = self.host.files / "launch"
        for text in (
            "#!/bin/sh\necho no-exec\n",
            "exec /bin/false\n",
            "exec $BINARY\n",
        ):
            launcher.write_text(text)
            with self.assertRaises(RuntimeError):
                self.host.observer().snapshot()

    def test_process_config_argument_must_be_unambiguous_and_pinned(self):
        command = self.host.proc / str(PID) / "cmdline"
        good = str(self.host.files / "config").encode()
        for raw in (
            b"engine\0--config\0relative\0",
            b"engine\0--config\0" + str(self.host.files / "key").encode() + b"\0",
            b"engine\0--config\0" + good + b"\0--config\0" + good + b"\0",
            b"engine\0--config=" + good + b"\0",
            b"engine\0--config\0" + good,
        ):
            with self.subTest(raw=raw):
                command.write_bytes(raw)
                with self.assertRaises((ValueError, RuntimeError, OSError)):
                    self.host.observer().snapshot()

    def test_missing_systemd_property_needs_exact_empty_dbus_proof(self):
        original = self.host.command

        def omit_property(argv):
            result = original(argv)
            if argv[:3] == ["systemctl", "--user", "show"]:
                return result.replace(b"ExecStop=\n", b"")
            return result

        observer = LinuxPairObserver(
            self.host.spec,
            proc=self.host.proc,
            command=omit_property,
            sleep=lambda _: None,
        )
        observer.snapshot()
        self.host.dbus["ExecStop"] = b"a(sasbttttuii) 1 [hidden]\n"
        with self.assertRaisesRegex(RuntimeError, "cannot prove") as failure:
            observer.snapshot()
        self.assertNotIn("hidden", str(failure.exception))
        self.host.dbus["ExecStop"] = b"unexpected-signature 0\n"
        with self.assertRaisesRegex(RuntimeError, "cannot prove"):
            observer.snapshot()

    def test_missing_unsupported_property_and_invalid_unit_object_are_refused(self):
        original = self.host.command

        def omit_pid(argv):
            result = original(argv)
            return result.replace(f"MainPID={PID}\n".encode(), b"")

        observer = LinuxPairObserver(
            self.host.spec, proc=self.host.proc, command=omit_pid, sleep=lambda _: None
        )
        with self.assertRaisesRegex(ValueError, "missing unit properties"):
            observer.snapshot()
        self.host.unit_object = b'o "/another/service"\n'
        with self.assertRaisesRegex(RuntimeError, "D-Bus"):
            self.host.observer().snapshot()

    def test_all_six_omitted_properties_require_typed_empty_arrays(self):
        original = self.host.command

        def omit_six(argv):
            result = original(argv)
            if argv[:3] == ["systemctl", "--user", "show"]:
                for key in self.host.dbus:
                    result = result.replace((key + "=\n").encode(), b"")
            return result

        observer = LinuxPairObserver(
            self.host.spec, proc=self.host.proc, command=omit_six, sleep=lambda _: None
        )
        observer.snapshot()
        self.host.dbus["EnvironmentFiles"] = b'a(sb) 1 "/secret/path" false\n'
        with self.assertRaisesRegex(RuntimeError, "cannot prove") as failure:
            observer.snapshot()
        self.assertNotIn("/secret/path", str(failure.exception))

    def test_changed_dbus_unit_object_is_refused(self):
        calls = 0

        def change_path(argv):
            nonlocal calls
            if argv[:3] == ["busctl", "--user", "call"]:
                calls += 1
                if calls == 2:
                    self.host.unit_object = (
                        b'o "/org/freedesktop/systemd1/unit/changed_2eservice"\n'
                    )

        self.host.on_command = change_path
        with self.assertRaisesRegex(RuntimeError, "object changed"):
            self.host.observer().snapshot()

    def test_pair_gpu_owner_and_idle_are_fail_closed(self):
        for bad in (
            f"{GPU0}, {PID}\n".encode(),
            f"{GPU0}, {PID}\n{GPU1}, {PID}\n{GPU1}, 88\n".encode(),
            f"{GPU0}, nope\n{GPU1}, {PID}\n".encode(),
        ):
            self.host.gpu = bad
            with self.assertRaises((ValueError, RuntimeError)):
                self.host.observer().snapshot()
        self.host.gpu = f"{GPU0}, {PID}\n{GPU1}, {PID}\n".encode()
        for bad in (
            f"{GPU0}, 1\n{GPU1}, 0\n".encode(),
            f"{GPU0}, 0\n".encode(),
            b"invalid\n",
        ):
            self.host.util = bad
            with self.assertRaises((ValueError, RuntimeError)):
                self.host.observer().snapshot()

    def test_socket_clients_and_unowned_listener_are_refused(self):
        self.host.ss = b"ESTAB 0 0 127.0.0.1:18791 127.0.0.1:50000\n"
        with self.assertRaises(RuntimeError):
            self.host.observer().snapshot()
        self.host.ss = b"LISTEN 0 128 127.0.0.1:18791 0.0.0.0:*\n"
        self.host.write_sockets(state="01")
        with self.assertRaises(RuntimeError):
            self.host.observer().snapshot()
        self.host.write_sockets(inode="98765")
        with self.assertRaises(RuntimeError):
            self.host.observer().snapshot()
        (self.host.proc / "net/tcp").write_text("header\n")
        with self.assertRaises(RuntimeError):
            self.host.observer().snapshot()

    def test_bad_spec_and_missing_observations_are_refused(self):
        with self.assertRaises(ValueError):
            PairSpec(
                "bad",
                self.host.spec.launcher,
                self.host.spec.executable,
                self.host.spec.config,
                self.host.spec.build_environment,
                self.host.spec.api_key,
                (GPU0, GPU1),
                PORT,
            )
        with self.assertRaises(ValueError):
            PairSpec(
                self.host.spec.unit,
                self.host.spec.launcher,
                self.host.spec.executable,
                self.host.spec.config,
                self.host.spec.build_environment,
                self.host.spec.api_key,
                (GPU0, GPU0),
                PORT,
            )
        (self.host.proc / "net/tcp6").unlink()
        with self.assertRaises(OSError):
            self.host.observer().snapshot()


if __name__ == "__main__":
    unittest.main()
