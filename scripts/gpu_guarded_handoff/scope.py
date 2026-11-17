"""Contain one workload in a uniquely named, nondelegated user systemd scope."""

import errno
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import time

CGROOT = Path("/sys/fs/cgroup")


def check_manager():
    for name in ("/usr/bin/systemd-run", "/usr/bin/systemctl"):
        if not shutil.which(name):
            raise ValueError(f"{name} unavailable")
    if not (CGROOT / "cgroup.controllers").is_file():
        raise ValueError("cgroup v2 unavailable")
    subprocess.run(
        ["/usr/bin/systemctl", "--user", "show-environment"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        timeout=10,
    )


class Scope:
    def __init__(self):
        self.unit = f"fifi-2080-handoff-{os.getpid()}-{secrets.token_hex(8)}.scope"
        self.invocation = None
        self.cgroup = None

    def command(self, args):
        return [
            "/usr/bin/systemd-run",
            "--user",
            "--scope",
            "--quiet",
            "--unit=" + self.unit,
            *args,
        ]

    def query(self):
        text = subprocess.check_output(
            [
                "/usr/bin/systemctl",
                "--user",
                "show",
                self.unit,
                "-p",
                "Id",
                "-p",
                "LoadState",
                "-p",
                "ActiveState",
                "-p",
                "InvocationID",
                "-p",
                "ControlGroup",
                "-p",
                "Delegate",
            ],
            text=True,
            timeout=10,
        )
        data = dict(line.split("=", 1) for line in text.splitlines())
        if data.get("Id") != self.unit:
            raise RuntimeError("unexpected transient unit identity")
        return data

    def observe(self):
        data = self.query()
        if data.get("LoadState") == "not-found":
            return False
        group, invocation = data.get("ControlGroup", ""), data.get("InvocationID", "")
        # `systemctl show` may see the newly loaded unit before systemd assigns
        # its cgroup and invocation ID. Retry only this exact uninitialized state;
        # after pinning, or with a partial/mismatched identity, refuse it.
        if (
            self.invocation is None
            and data.get("LoadState") == "loaded"
            and data.get("ActiveState") == "inactive"
            and data.get("Delegate") == "no"
            and not group
            and not invocation
        ):
            return False
        expected = f"/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/app.slice/{self.unit}"
        if (
            group != expected
            or data.get("Delegate") != "no"
            or not re.fullmatch(r"[0-9a-f]{32}", invocation)
        ):
            raise RuntimeError("transient scope has unexpected cgroup or invocation")
        if self.invocation is not None and (
            invocation != self.invocation or group != self.cgroup
        ):
            raise RuntimeError("transient scope identity changed")
        self.cgroup, self.invocation = group, invocation
        return True

    def contains(self, pid):
        if self.cgroup is None:
            return False
        entries = Path(f"/proc/{pid}/cgroup").read_text().splitlines()
        paths = [entry.split(":", 2)[2] for entry in entries if entry.startswith("0::")]
        return len(paths) == 1 and (
            paths[0] == self.cgroup or paths[0].startswith(self.cgroup + "/")
        )

    def populated(self):
        if self.cgroup is None:
            return None
        events = CGROOT / self.cgroup.lstrip("/") / "cgroup.events"
        try:
            fields = dict(line.split() for line in events.read_text().splitlines())
        except OSError as err:
            # systemd can remove the empty cgroup between open and read.
            if err.errno in (errno.ENOENT, errno.ENODEV) and not events.exists():
                return False
            raise
        if fields.get("populated") not in ("0", "1"):
            raise RuntimeError("cannot prove cgroup population")
        return fields["populated"] == "1"

    def wait_started(self, child):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if self.observe():
                return
            if child.poll() is not None:
                raise RuntimeError(
                    "scope launcher exited before scope identity was observed"
                )
            time.sleep(0.1)
        raise TimeoutError("scope identity not visible after 15 seconds")

    def kill(self, sig):
        if not self.observe():
            return
        subprocess.run(
            [
                "/usr/bin/systemctl",
                "--user",
                "kill",
                "--kill-whom=all",
                "--signal=" + sig,
                self.unit,
            ],
            check=True,
            timeout=15,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    def cleanup(self, child):
        # Never signal a changed unit. Never restore/release the lease until the
        # exact scope is known empty; unknown state holds the lease for recovery.
        while self.invocation is None:
            try:
                if self.observe():
                    break
                if child.poll() is not None:
                    print(
                        "Launcher exited before scope identity was pinned; retaining lease until resolved",
                        flush=True,
                    )
            except (
                OSError,
                ValueError,
                RuntimeError,
                subprocess.SubprocessError,
            ) as err:
                print(f"Cannot identify scope; retaining lease: {err}", flush=True)
            time.sleep(1)
        try:
            if self.populated():
                self.kill("SIGTERM")
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as err:
            print(f"Could not terminate scope: {err}", flush=True)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            try:
                if not self.populated():
                    break
            except (OSError, ValueError, RuntimeError) as err:
                print(f"Cannot inspect scope; retaining lease: {err}", flush=True)
            time.sleep(0.25)
        try:
            if self.populated():
                self.kill("SIGKILL")
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as err:
            print(f"Could not kill scope: {err}", flush=True)
        while True:
            try:
                if not self.populated():
                    break
            except (OSError, ValueError, RuntimeError) as err:
                print(f"Cannot inspect scope; retaining lease: {err}", flush=True)
            print(f"Waiting for scope {self.unit} to empty; lease retained", flush=True)
            time.sleep(1)
        # The systemd-run launcher is outside the scope. Reap it too; an
        # unexpectedly stuck launcher must not outlive lease release.
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                while child.poll() is None:
                    print(
                        "Waiting for scope launcher to exit; lease retained", flush=True
                    )
                    time.sleep(1)
        child.wait(timeout=10)
