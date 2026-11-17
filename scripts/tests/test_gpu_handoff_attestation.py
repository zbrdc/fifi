"""Offline tests for handoff process attestation and owner-change detection."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from gpu_guarded_handoff import attested_launch, host  # noqa: E402


class AssistantConflictTests(unittest.TestCase):
    def setUp(self):
        self.owner = (101, "/usr/bin/claude", (1, 123), b"claude\0")
        self.child = (202, "/usr/bin/uv", (1, 456), b"uv\0")
        self.base = ({10: self.owner}, {(10, 20): (self.child, True)})

    def test_unchanged_persistent_child_is_allowed(self):
        self.assertFalse(host.assistant_conflict(self.base, self.base))

    def test_new_owner_is_blocked(self):
        self.assertTrue(
            host.assistant_conflict(self.base, ({10: self.owner, 11: self.owner}, {}))
        )

    def test_changed_owner_start_exe_inode_or_argv_is_blocked(self):
        variants = [
            (102, *self.owner[1:]),
            (101, "/other", *self.owner[2:]),
            (*self.owner[:2], (1, 999), self.owner[3]),
            (*self.owner[:3], b"changed\0"),
        ]
        for changed in variants:
            with self.subTest(changed=changed):
                self.assertTrue(host.assistant_conflict(self.base, ({10: changed}, {})))

    def test_new_persistent_child_is_blocked(self):
        # A newly launched uv/npm child must not pass merely because its name is persistent.
        self.assertTrue(
            host.assistant_conflict(
                self.base,
                (
                    {10: self.owner},
                    {(10, 20): (self.child, True), (10, 21): (self.child, True)},
                ),
            )
        )

    def test_reused_child_pid_or_changed_identity_is_blocked(self):
        changed = (203, *self.child[1:])
        self.assertTrue(
            host.assistant_conflict(
                self.base, ({10: self.owner}, {(10, 20): (changed, True)})
            )
        )

    def test_foreground_child_blocks_even_if_preexisting(self):
        foreground = ({10: self.owner}, {(10, 20): (self.child, False)})
        self.assertTrue(host.assistant_conflict(foreground, foreground))

    def test_exited_owner_is_a_conflict(self):
        self.assertTrue(host.assistant_conflict(self.base, ({}, {})))

    def test_disappearing_child_is_a_conflict(self):
        self.assertTrue(host.assistant_conflict(self.base, ({10: self.owner}, {})))


class AttestationTests(unittest.TestCase):
    def test_attested_unit_parser_rejects_nonattested_or_reload(self):
        attestor = str(host.ATTESTOR)
        valid = {
            "NeedDaemonReload": "no",
            "ExecStart": "{ path=/usr/bin/python3 ; argv[]=/usr/bin/python3 "
            + attestor
            + " ; ignore_errors=no ; }",
        }
        with mock.patch.object(
            host.os.path, "samefile", side_effect=lambda a, b: str(a) == str(b)
        ):
            self.assertTrue(host.attested_execstart(valid))
            self.assertFalse(
                host.attested_execstart({**valid, "NeedDaemonReload": "yes"})
            )
            self.assertFalse(
                host.attested_execstart(
                    {
                        **valid,
                        "ExecStart": "{ path=/bin/sh ; argv[]=/bin/sh /tmp/start ; }",
                    }
                )
            )
            self.assertFalse(
                host.attested_execstart(
                    {
                        **valid,
                        "ExecStart": valid["ExecStart"].replace(
                            attestor, attestor + " --extra"
                        ),
                    }
                )
            )
            self.assertFalse(host.attested_execstart({}))

    def test_environment_fingerprint_excludes_volatile_fields_only(self):
        env = {
            b"FOO": b"one",
            b"INVOCATION_ID": b"old",
            b"FIFI_HANDOFF_LAUNCH_PID": b"1",
        }
        same = {
            b"FOO": b"one",
            b"INVOCATION_ID": b"new",
            b"FIFI_HANDOFF_LAUNCH_PID": b"2",
        }
        changed = {**same, b"FOO": b"two"}
        self.assertEqual(
            host.environment_fingerprint(env), host.environment_fingerprint(same)
        )
        self.assertNotEqual(
            host.environment_fingerprint(env), host.environment_fingerprint(changed)
        )

    def test_attestor_execs_only_after_stable_hashes_and_injects_exact_receipt(self):
        with tempfile.TemporaryDirectory(prefix="fifi-attest-") as directory:
            root = Path(directory)
            start = root / "start.sh"
            config = root / "config.json"
            bash = root / "bash"
            build_env = root / "build-env.sh"
            start.write_text("exec fake\n")
            config.write_text('{"safe": true}\n')
            bash.write_text("fake bash\n")
            build_env.write_text("export EXAMPLE=1\n")
            files = {
                "FIFI_HANDOFF_START_SHA256": start,
                "FIFI_HANDOFF_CONFIG_SHA256": config,
                "FIFI_HANDOFF_BUILD_ENV_SHA256": build_env,
            }
            output, errors = io.StringIO(), io.StringIO()
            with (
                mock.patch.object(attested_launch, "INPUTS", files),
                mock.patch.object(attested_launch, "START", start),
                mock.patch.object(
                    attested_launch, "APPROVED_START_SHA256", host.digest(start)
                ),
                mock.patch.object(
                    attested_launch, "APPROVED_BUILD_ENV_SHA256", host.digest(build_env)
                ),
                mock.patch.object(
                    attested_launch.os.path, "samefile", return_value=True
                ),
                mock.patch.object(
                    attested_launch.shutil, "which", return_value=str(bash)
                ),
                mock.patch.object(attested_launch.os, "execve") as launch,
                mock.patch.object(
                    attested_launch, "environment_files_empty", return_value=True
                ),
                mock.patch.object(
                    attested_launch, "inherited_manager_matches", return_value=True
                ),
                mock.patch.object(
                    attested_launch.subprocess,
                    "check_output",
                    return_value=b"SECRET=dont-print-me\n",
                ),
                mock.patch.object(
                    attested_launch, "manager_fingerprint", return_value="a" * 64
                ),
                mock.patch.object(
                    attested_launch,
                    "unit_environment_fingerprint",
                    return_value="b" * 64,
                ),
                mock.patch.dict(
                    attested_launch.os.environ, {"SECRET": "dont-print-me"}, clear=True
                ),
                redirect_stdout(output),
                redirect_stderr(errors),
            ):
                attested_launch.main()
            launch.assert_called_once()
            executable, argv, env = launch.call_args.args
            self.assertEqual(executable, str(start))
            self.assertEqual(argv, [str(start)])
            self.assertEqual(env["FIFI_HANDOFF_MANAGER_ENV_SHA256"], "a" * 64)
            self.assertEqual(env["FIFI_HANDOFF_UNIT_ENV_SHA256"], "b" * 64)
            self.assertEqual(
                env["FIFI_HANDOFF_LAUNCH_PID"], str(attested_launch.os.getpid())
            )
            self.assertTrue(env["FIFI_HANDOFF_LAUNCH_START"].isdigit())
            self.assertEqual(env["FIFI_HANDOFF_START_SHA256"], host.digest(start))
            self.assertEqual(env["FIFI_HANDOFF_CONFIG_SHA256"], host.digest(config))
            self.assertEqual(env["FIFI_HANDOFF_BASH_SHA256"], host.digest(bash))
            self.assertEqual(env["FIFI_HANDOFF_BASH_PATH"], str(bash))
            self.assertNotIn("dont-print-me", output.getvalue() + errors.getvalue())

    def test_attestor_rejects_hash_change_before_exec(self):
        with tempfile.TemporaryDirectory(prefix="fifi-attest-") as directory:
            root = Path(directory)
            target = root / "start.sh"
            bash = root / "bash"
            build_env = root / "build-env.sh"
            target.write_text("test\n")
            bash.write_text("bash\n")
            build_env.write_text("env\n")
            fake = [
                ((1, 1, 1, 1), "before"),
                ((1, 3, 1, 1), "env"),
                ((1, 2, 1, 1), "bash"),
                ((1, 1, 1, 1), "after"),
            ]
            with (
                mock.patch.object(
                    attested_launch,
                    "INPUTS",
                    {
                        "FIFI_HANDOFF_START_SHA256": target,
                        "FIFI_HANDOFF_BUILD_ENV_SHA256": build_env,
                    },
                ),
                mock.patch.object(attested_launch, "START", target),
                mock.patch.object(attested_launch, "APPROVED_START_SHA256", "before"),
                mock.patch.object(attested_launch, "APPROVED_BUILD_ENV_SHA256", "env"),
                mock.patch.object(
                    attested_launch.os.path, "samefile", return_value=True
                ),
                mock.patch.object(
                    attested_launch.shutil, "which", return_value=str(bash)
                ),
                mock.patch.object(attested_launch, "identity", side_effect=fake),
                mock.patch.object(attested_launch.os, "execve") as launch,
            ):
                with self.assertRaisesRegex(RuntimeError, "changed"):
                    attested_launch.main()
            launch.assert_not_called()

    def test_unreviewed_start_or_build_environment_is_refused(self):
        with tempfile.TemporaryDirectory(prefix="fifi-attest-") as directory:
            root = Path(directory)
            start, build_env, bash = root / "start", root / "env", root / "bash"
            for item in (start, build_env, bash):
                item.write_text("test\n")
            files = {
                "FIFI_HANDOFF_START_SHA256": start,
                "FIFI_HANDOFF_BUILD_ENV_SHA256": build_env,
            }
            with (
                mock.patch.object(attested_launch, "INPUTS", files),
                mock.patch.object(attested_launch, "START", start),
                mock.patch.object(attested_launch, "APPROVED_START_SHA256", "wrong"),
                mock.patch.object(
                    attested_launch, "APPROVED_BUILD_ENV_SHA256", host.digest(build_env)
                ),
                mock.patch.object(
                    attested_launch.os.path, "samefile", return_value=True
                ),
                mock.patch.object(
                    attested_launch.shutil, "which", return_value=str(bash)
                ),
                mock.patch.object(attested_launch.os, "execve") as launch,
            ):
                with self.assertRaisesRegex(RuntimeError, "not reviewed"):
                    attested_launch.main()
            launch.assert_not_called()

    def test_manager_change_after_fork_or_during_receipt_rejected_before_exec(self):
        # All manager queries and executable actions are mocked; no live bus access.
        with tempfile.TemporaryDirectory(prefix="fifi-attest-manager-") as directory:
            root = Path(directory)
            start, build_env, bash = root / "start", root / "env", root / "bash"
            for path in (start, build_env, bash):
                path.write_text("data\n")
            inputs = {
                "FIFI_HANDOFF_START_SHA256": start,
                "FIFI_HANDOFF_BUILD_ENV_SHA256": build_env,
            }
            cases = (
                ([b"VALUE=changed\n"], {"VALUE": "original"}),
                ([b"VALUE=original\n", b"VALUE=changed\n"], {"VALUE": "original"}),
            )
            for outputs, inherited in cases:
                with (
                    self.subTest(outputs=outputs),
                    mock.patch.object(attested_launch, "INPUTS", inputs),
                    mock.patch.object(attested_launch, "START", start),
                    mock.patch.object(
                        attested_launch, "APPROVED_START_SHA256", host.digest(start)
                    ),
                    mock.patch.object(
                        attested_launch,
                        "APPROVED_BUILD_ENV_SHA256",
                        host.digest(build_env),
                    ),
                    mock.patch.object(
                        attested_launch.os.path, "samefile", return_value=True
                    ),
                    mock.patch.object(
                        attested_launch.shutil, "which", return_value=str(bash)
                    ),
                    mock.patch.object(
                        attested_launch, "environment_files_empty", return_value=True
                    ),
                    mock.patch.object(
                        attested_launch,
                        "unit_environment_fingerprint",
                        return_value="b" * 64,
                    ),
                    mock.patch.object(
                        attested_launch.subprocess, "check_output", side_effect=outputs
                    ),
                    mock.patch.object(attested_launch.os, "execve") as launch,
                    mock.patch.dict(attested_launch.os.environ, inherited, clear=True),
                ):
                    with self.assertRaisesRegex(RuntimeError, "changed"):
                        attested_launch.main()
                    launch.assert_not_called()

    def test_attestor_rejects_wrong_python_before_exec(self):
        with (
            mock.patch.object(attested_launch.os.path, "samefile", return_value=False),
            mock.patch.object(attested_launch.os, "execve") as launch,
        ):
            with self.assertRaisesRegex(RuntimeError, "Python"):
                attested_launch.main()
        launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
