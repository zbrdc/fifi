"""Opt-in test of FD inheritance through a real user systemd scope.

This test uses only a fresh temporary lock. It does not access the GPU lock,
send requests, or change an existing service.
"""

import fcntl
import json
import os
from pathlib import Path
import secrets
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

import p2p_test_support  # noqa: F401 -- make repository packages visible under python -I

from scripts.tests.p2p_scope_helpers import (
    group_populated,
    scope_can_stop,
    scope_state,
    scope_terminated,
)


CHILD = r"""
import fcntl,json,os,sys
fd=int(sys.argv[1]); lock_path=sys.argv[2]
st=os.fstat(fd)
cgroups=open('/proc/self/cgroup', encoding='utf-8').read().splitlines()
paths=[line.split(':', 2)[2] for line in cgroups if line.startswith('0::')]
independent=os.open(lock_path, os.O_RDWR)
try:
    try:
        fcntl.flock(independent, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        blocked=True
    else:
        blocked=False
finally:
    os.close(independent)
print(json.dumps({'dev':st.st_dev, 'ino':st.st_ino, 'blocked':blocked,
                  'cgroups':paths}), flush=True)
sys.stdin.readline()
"""

CONTENDER = r"""
import fcntl,os,sys
fd=os.open(sys.argv[1], os.O_RDWR)
try:
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('blocked')
    else:
        print('acquired')
finally:
    os.close(fd)
"""


@unittest.skipUnless(
    os.environ.get("FIFI_SCOPE_REAL_TEST") == "1",
    "explicit CPU-only systemd-scope integration test",
)
class ScopeFdTest(unittest.TestCase):
    def test_scope_inherits_locked_fd_and_drains(self):
        if not shutil.which("systemd-run") or not shutil.which("systemctl"):
            self.skipTest("systemd user tools unavailable")
        try:
            subprocess.run(
                ["systemctl", "--user", "show-environment"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            self.skipTest("systemd user manager unavailable")

        unit = f"fifi-fd-probe-{os.getpid()}-{secrets.token_hex(8)}.scope"
        with tempfile.TemporaryDirectory(prefix="fifi-fd-probe-") as directory:
            lock_path = Path(directory) / "fresh.lock"
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            process = None
            group = None
            invocation = None
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                expected = os.fstat(fd)
                process = subprocess.Popen(
                    [
                        "systemd-run",
                        "--user",
                        "--scope",
                        "--quiet",
                        "--unit=" + unit,
                        sys.executable,
                        "-c",
                        CHILD,
                        str(fd),
                        str(lock_path),
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    pass_fds=(fd,),
                    start_new_session=True,
                )
                ready, _, _ = select.select([process.stdout], [], [], 10)
                self.assertTrue(ready, "scope child did not report within deadline")
                line = process.stdout.readline()
                if not line:
                    self.fail(
                        "scope child exited before reporting: " + process.stderr.read()
                    )
                report = json.loads(line)
                self.assertEqual(
                    (report["dev"], report["ino"]), (expected.st_dev, expected.st_ino)
                )
                self.assertIs(report["blocked"], True)
                self.assertEqual(len(report["cgroups"]), 1)
                group = Path("/sys/fs/cgroup") / report["cgroups"][0].lstrip("/")
                self.assertEqual(group.name, unit)
                self.assertIn("populated 1", (group / "cgroup.events").read_text())
                observed = scope_state(unit)
                self.assertEqual(observed.get("Id"), unit)
                self.assertEqual(observed.get("LoadState"), "loaded")
                self.assertEqual(observed.get("ActiveState"), "active")
                self.assertEqual(observed.get("ControlGroup"), report["cgroups"][0])
                invocation = observed.get("InvocationID", "")
                self.assertRegex(invocation, r"^[0-9a-f]{32}$")

                contender = subprocess.run(
                    [sys.executable, "-c", CONTENDER, str(lock_path)],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=5,
                    close_fds=True,
                )
                self.assertEqual(contender.stdout.strip(), "blocked")
                process.stdin.write("done\n")
                process.stdin.flush()
                _, stderr = process.communicate(timeout=10)
                self.assertEqual(process.returncode, 0, stderr)
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    if scope_terminated(
                        scope_state(unit), unit, invocation, group_populated(group)
                    ):
                        break
                    time.sleep(0.05)
                else:
                    self.fail("exact scope did not terminate within deadline")

                # Once our only remaining owner closes, an independent process
                # must be able to acquire the same temporary lock.
                os.close(fd)
                fd = None
                contender = subprocess.run(
                    [sys.executable, "-c", CONTENDER, str(lock_path)],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=5,
                    close_fds=True,
                )
                self.assertEqual(contender.stdout.strip(), "acquired")
            finally:
                if process is not None and process.poll() is None:
                    try:
                        current = scope_state(unit)
                    except (OSError, ValueError, subprocess.SubprocessError):
                        current = {}
                    if scope_can_stop(current, unit, invocation):
                        subprocess.run(
                            ["systemctl", "--user", "stop", unit],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            timeout=5,
                            check=False,
                        )
                    if process.poll() is None:
                        try:
                            os.killpg(process.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        try:
                            process.communicate(timeout=5)
                        except subprocess.TimeoutExpired:
                            try:
                                os.killpg(process.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                            process.communicate(timeout=5)
                if fd is not None:
                    os.close(fd)


if __name__ == "__main__":
    unittest.main()
