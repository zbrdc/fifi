"""Run a trusted 2080 workload inside the cooperative host lease.

Clients and resource idleness are sampled, not fenced. The existing coordinator
must first confirm the 2080 and pair are idle. This cannot stop a same-user
process bypassing the advisory lease with a direct systemctl call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import tempfile
import time

from . import core, host, workload

SCRATCH_ROOT = Path("/home/ben/fifi-scratch")


def arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scratch-dir",
        type=Path,
        required=True,
        help="existing task-owned writable scratch directory",
    )
    parser.add_argument(
        "--required-free-mib",
        type=int,
        required=True,
        help="declared minimum free 2080 memory, at least 2048 MiB",
    )
    parser.add_argument(
        "--min-disk-mib",
        type=int,
        required=True,
        help="minimum task scratch free space",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        required=True,
        help="maximum workload time; cleanup/restoration may take longer",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="trusted command after --; no shell is started implicitly",
    )
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command or args.timeout_seconds <= 0 or args.min_disk_mib <= 0:
        parser.error(
            "a command, positive timeout and positive disk requirement are required"
        )
    return args


def hold_for_recovery(outcome, service, command):
    """Do not silently release the lease until an operator restores safety.

    SIGKILL, process death and machine failure cannot be fenced by flock; this
    is not a durable broker and does not replace supervisor crash recovery.
    """
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, signal.SIG_IGN)
    lease = outcome.retained_lease
    snapshot = service.snapshot_value
    safe_print(
        "INCONCLUSIVE: lease retained; restore the pinned service and empty its "
        "workload scope before retrying. SIGKILL defeats the safety hold."
    )
    while True:
        try:
            lease.verify()
            if (
                command.cleanup_and_verify_empty() is True
                and service.verify_restored(snapshot) is True
            ):
                lease.verify()
                lease.release()
                safe_print(
                    "Pinned service ready and workload scope empty; lease released."
                )
                return 1
        except Exception as exc:
            safe_print(f"Recovery not yet verified: {type(exc).__name__}")
        time.sleep(2)


def safe_print(message):
    try:
        print(message, file=sys.stderr, flush=True)
    except (OSError, ValueError):
        # A closed output pipe must not release an unresolved lease.
        pass


def main(argv=None):
    args = arguments(sys.argv[1:] if argv is None else argv)
    root = args.scratch_dir.resolve(strict=True)
    scratch = SCRATCH_ROOT.resolve(strict=True)
    if (
        not root.is_relative_to(scratch)
        or not root.is_dir()
        or not os.access(root, os.W_OK | os.X_OK)
    ):
        raise RuntimeError("scratch directory is not writable")
    if shutil.disk_usage(root).free < args.min_disk_mib * 1024 * 1024:
        raise RuntimeError("insufficient task scratch space")
    service = host.Host2080(args.required_free_mib)
    campaign = Path(tempfile.mkdtemp(prefix="fifi-handoff-", dir=root))
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": host.TARGET_GPU,
            "TMPDIR": str(campaign),
            "TMP": str(campaign),
            "TEMP": str(campaign),
            "CUDA_CACHE_PATH": str(campaign / "cuda-cache"),
            "INFERENCE_CUDA_REQUIRE_DEVICE": "1",
        }
    )
    cmd = workload.ScopedCommand(
        args.command, lambda: service.monitor(cmd.scope), env=env
    )
    print(f"Campaign directory: {campaign}", flush=True)
    # core installs interruption handlers for the active handoff and restores
    # these ignored handlers before returning. Thus an unresolved lease cannot
    # be lost in the gap between its return and the recovery loop.
    previous = {
        sig: signal.signal(sig, signal.SIG_IGN)
        for sig in (signal.SIGINT, signal.SIGTERM)
    }
    outcome = None
    try:
        outcome = core.handoff_shared(service, cmd, args.timeout_seconds)
    except Exception as error:
        (campaign / "outcome.json").write_text(
            json.dumps(
                {
                    "result": "refused",
                    "reason": type(error).__name__,
                    "service_action": False,
                },
                indent=2,
            )
            + "\n"
        )
        raise
    finally:
        if outcome is None:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    result = (
        "workload_exit_0_restored" if outcome.result == "passed" else outcome.result
    )
    receipt = {
        "result": result,
        "reason": (
            "workload exited zero; inspect its own artifacts"
            if outcome.result == "passed"
            else "handoff failed or is inconclusive; inspect the caller's protected logs"
        ),
        "service_was_active": outcome.service_was_active,
        "restore_confirmed": outcome.restore_confirmed,
        "workload_drained": outcome.workload_drained,
        "device_uuid": host.TARGET_GPU,
        "command_sha256": hashlib.sha256(json.dumps(args.command).encode()).hexdigest(),
        # Whitelist only non-secret metadata: do not serialize unit arguments,
        # environment, API keys, or hashes of low-entropy secrets.
        "service_snapshot": (
            {
                key: service.snapshot_value[key]
                for key in (
                    "pid",
                    "starttime",
                    "exe",
                    "primary_pid",
                    "model",
                    "context",
                )
                if key in service.snapshot_value
            }
            if isinstance(service.snapshot_value, dict)
            else None
        ),
        "qualification": "not checked; inspect workload artifacts",
    }
    try:
        (campaign / "outcome.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt), flush=True)
    except (OSError, ValueError) as error:
        if outcome.retained_lease is not None:
            safe_print(f"Cannot write receipt: {error}")
            return hold_for_recovery(outcome, service, cmd)
        raise
    if outcome.retained_lease is not None:
        return hold_for_recovery(outcome, service, cmd)
    for sig, handler in previous.items():
        signal.signal(sig, handler)
    return 0 if outcome.result == "passed" else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Guarded handoff refused: {type(error).__name__}", file=sys.stderr)
        sys.exit(1)
