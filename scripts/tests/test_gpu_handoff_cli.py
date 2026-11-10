"""CPU-only CLI tests: never touch systemd, sockets or a CUDA device."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from gpu_guarded_handoff import __main__ as cli  # noqa: E402
from gpu_guarded_handoff import core  # noqa: E402


class HandoffCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="fifi-handoff-cli-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.scratch = self.root / "task"
        self.scratch.mkdir()
        self.service = mock.Mock()
        self.service.snapshot_value = {"pid": 42, "model": "test", "context": 65536}
        self.command = mock.Mock()
        self.command.scope = mock.Mock()
        self.patches = [
            mock.patch.object(cli, "SCRATCH_ROOT", self.root),
            mock.patch.object(cli.host, "Host2080", return_value=self.service),
            mock.patch.object(cli.workload, "ScopedCommand", return_value=self.command),
            mock.patch.object(cli.core, "handoff_shared"),
            mock.patch.object(cli.signal, "signal", return_value=None),
        ]
        for patch in self.patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.output = io.StringIO()
        self.out_context = redirect_stdout(self.output)
        self.out_context.__enter__()
        self.addCleanup(self.out_context.__exit__, None, None, None)

    def args(self):
        return [
            "--scratch-dir",
            str(self.scratch),
            "--required-free-mib",
            "2048",
            "--min-disk-mib",
            "1",
            "--timeout-seconds",
            "5",
            "--",
            "/fake/workload",
            "--check",
        ]

    def receipt(self):
        campaigns = list(self.scratch.glob("fifi-handoff-*"))
        self.assertEqual(len(campaigns), 1)
        return json.loads((campaigns[0] / "outcome.json").read_text())

    def outcome(self, result="passed", lease=None):
        return core.Outcome(
            result=result,
            reason="test reason",
            service_was_active=True,
            restore_confirmed=True,
            workload_drained=True,
            retained_lease=lease,
        )

    def test_help_exits_without_host_or_workload_action(self):
        with self.assertRaises(SystemExit) as raised:
            cli.main(["--help"])
        self.assertEqual(raised.exception.code, 0)
        cli.host.Host2080.assert_not_called()
        cli.workload.ScopedCommand.assert_not_called()
        cli.core.handoff_shared.assert_not_called()

    def test_missing_command_fails_before_any_service_action(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            cli.main(self.args()[:-3])
        self.assertNotEqual(raised.exception.code, 0)
        cli.host.Host2080.assert_not_called()
        cli.core.handoff_shared.assert_not_called()

    def test_invalid_timeout_fails_before_any_service_action(self):
        args = self.args()
        args[args.index("--timeout-seconds") + 1] = "0"
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.main(args)
        cli.host.Host2080.assert_not_called()
        cli.core.handoff_shared.assert_not_called()

    def test_busy_lease_refuses_without_stopping_service_or_starting_work(self):
        cli.core.handoff_shared.side_effect = core.Busy("already reserved")
        with self.assertRaises(core.Busy):
            cli.main(self.args())
        self.service.stop.assert_not_called()
        self.service.restore.assert_not_called()
        self.command.start.assert_not_called()
        self.assertEqual(self.receipt()["result"], "refused")
        self.assertFalse(self.receipt()["service_action"])

    def test_success_pins_gpu_and_scratch_and_does_not_claim_qualification(self):
        cli.core.handoff_shared.return_value = self.outcome()
        self.assertEqual(cli.main(self.args()), 0)
        args, kwargs = cli.workload.ScopedCommand.call_args
        self.assertEqual(args[0], ["/fake/workload", "--check"])
        env = kwargs["env"]
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], cli.host.TARGET_GPU)
        self.assertEqual(env["INFERENCE_CUDA_REQUIRE_DEVICE"], "1")
        campaign = Path(env["TMPDIR"])
        self.assertTrue(campaign.is_relative_to(self.scratch))
        self.assertEqual(env["TMP"], str(campaign))
        self.assertEqual(env["TEMP"], str(campaign))
        receipt = self.receipt()
        self.assertEqual(receipt["result"], "workload_exit_0_restored")
        self.assertEqual(
            receipt["qualification"], "not checked; inspect workload artifacts"
        )
        self.assertTrue(receipt["restore_confirmed"])
        cli.core.handoff_shared.assert_called_once_with(self.service, self.command, 5)

    def test_receipt_redacts_loaded_unit_metadata_and_environment(self):
        self.service.snapshot_value.update(
            {
                "unit_meta": {"command": "SECRET-TOKEN-UNIT"},
                "primary_meta": {"command": "SECRET-TOKEN-PRIMARY"},
                "manager_env": ["SECRET-TOKEN-MANAGER"],
            }
        )
        cli.core.handoff_shared.return_value = self.outcome()
        self.assertEqual(cli.main(self.args()), 0)
        text = json.dumps(self.receipt())
        self.assertNotIn("SECRET-TOKEN-UNIT", text)
        self.assertNotIn("SECRET-TOKEN-PRIMARY", text)
        self.assertNotIn("SECRET-TOKEN-MANAGER", text)
        self.assertNotIn("unit_meta", text)
        self.assertNotIn("primary_meta", text)

    def test_nonzero_workload_returns_failure_without_success_label(self):
        cli.core.handoff_shared.return_value = self.outcome(result="failed")
        self.assertNotEqual(cli.main(self.args()), 0)
        self.assertNotEqual(self.receipt()["result"], "workload_exit_0_restored")

    def test_inconclusive_held_lease_enters_recovery(self):
        lease = mock.Mock()
        outcome = self.outcome(result="inconclusive", lease=lease)
        outcome.restore_confirmed = False
        cli.core.handoff_shared.return_value = outcome
        with mock.patch.object(cli, "hold_for_recovery", return_value=1) as recovery:
            self.assertEqual(cli.main(self.args()), 1)
        recovery.assert_called_once_with(outcome, self.service, self.command)
        lease.release.assert_not_called()
        self.assertEqual(self.receipt()["result"], "inconclusive")

    def test_inconclusive_receipt_write_failure_still_holds(self):
        lease = mock.Mock()
        outcome = self.outcome(result="inconclusive", lease=lease)
        outcome.restore_confirmed = False
        cli.core.handoff_shared.return_value = outcome
        with (
            mock.patch.object(Path, "write_text", side_effect=OSError("disk full")),
            mock.patch.object(cli, "safe_print"),
            mock.patch.object(cli, "hold_for_recovery", return_value=1) as recovery,
        ):
            self.assertEqual(cli.main(self.args()), 1)
        recovery.assert_called_once_with(outcome, self.service, self.command)
        lease.release.assert_not_called()

    def test_closed_stderr_pipe_does_not_raise_in_recovery_notice(self):
        with mock.patch("builtins.print", side_effect=BrokenPipeError("closed")):
            cli.safe_print("keep the lease")


if __name__ == "__main__":
    unittest.main()
