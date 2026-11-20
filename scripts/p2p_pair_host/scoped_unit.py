"""Scope identity/drain observation for a future guarded pair campaign.

There is no service controller or entry point here. This observer only pins a
nondelegated, uniquely named systemd user scope and samples its state. Calls
must be run under an independently held lease and a separately supervised,
deadline-enforced controller. Same-UID systemd actions can still race any
sample and a direct path replacement is not fenced.
"""

from dataclasses import dataclass
import errno
import os
from pathlib import Path
import re
import subprocess
import time


class ScopeUncertain(RuntimeError):
    """The exact scope's identity or state could not be proven."""


@dataclass(frozen=True)
class ScopeIdentity:
    unit: str
    invocation: str
    cgroup: str


def _run(argv, timeout):
    return subprocess.check_output(argv, text=True, timeout=timeout)


class ScopedUnit:
    """Pin and verify one scope; never acquire or release a GPU lease."""

    def __init__(
        self,
        unit,
        *,
        uid=None,
        cgroup_root=Path("/sys/fs/cgroup"),
        command=_run,
        clock=time.monotonic,
    ):
        if not isinstance(unit, str) or not re.fullmatch(
            r"fifi-[a-z0-9-]{1,100}\.scope", unit
        ):
            raise ValueError("invalid unique scope name")
        self.unit = unit
        self.uid = os.getuid() if uid is None else uid
        if type(self.uid) is not int or self.uid < 0:
            raise ValueError("invalid scope user")
        self.root, self.command, self.clock = Path(cgroup_root), command, clock
        self.identity = None

    def _remaining(self, deadline):
        if type(deadline) not in (int, float):
            raise ScopeUncertain("scope deadline missing")
        remaining = deadline - self.clock()
        if not 0 < remaining <= 86400:
            raise ScopeUncertain("scope deadline expired or invalid")
        return min(remaining, 5.0)

    def _query(self, deadline):
        args = ["systemctl", "--user", "show", self.unit]
        names = (
            "Id",
            "LoadState",
            "ActiveState",
            "InvocationID",
            "ControlGroup",
            "Delegate",
        )
        for name in names:
            args += ["-p", name]
        try:
            text = self.command(args, self._remaining(deadline))
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            raise ScopeUncertain("scope query failed") from error
        self._remaining(deadline)
        if not isinstance(text, str):
            raise ScopeUncertain("scope properties malformed")
        fields = {}
        for line in text.splitlines():
            key, sep, value = line.partition("=")
            if not sep or key not in names or key in fields:
                raise ScopeUncertain("scope properties malformed")
            fields[key] = value
        if not {"Id", "LoadState"} <= set(fields) or fields["Id"] != self.unit:
            raise ScopeUncertain("scope unit identity unavailable")
        # systemctl may omit unset fields for a collected unit. Missing fields
        # are not treated as evidence: drained() also requires a confirmed
        # absent cgroup and repeats this query. Loaded scopes require all fields.
        if fields["LoadState"] == "loaded" and set(fields) != set(names):
            raise ScopeUncertain("loaded scope properties incomplete")
        return fields

    def _loaded_identity(self, fields):
        group = f"/user.slice/user-{self.uid}.slice/user@{self.uid}.service/app.slice/{self.unit}"
        if (
            fields["LoadState"] != "loaded"
            or fields["Delegate"] != "no"
            or fields["ControlGroup"] != group
            or not re.fullmatch(r"[0-9a-f]{32}", fields["InvocationID"])
        ):
            raise ScopeUncertain("scope identity is not provable")
        return ScopeIdentity(self.unit, fields["InvocationID"], group)

    def pin(self, deadline):
        if self.identity is not None:
            raise ScopeUncertain("scope already pinned")
        fields = self._query(deadline)
        if fields.get("ActiveState") != "active":
            raise ScopeUncertain("scope is not active")
        self.identity = self._loaded_identity(fields)
        return self.identity

    def _same(self, fields):
        if self.identity is None or self._loaded_identity(fields) != self.identity:
            raise ScopeUncertain("scope invocation changed or was not pinned")

    def _population(self):
        if self.identity is None:
            raise ScopeUncertain("scope is not pinned")
        group = self.root / self.identity.cgroup.lstrip("/")
        try:
            lines = (group / "cgroup.events").read_text().splitlines()
        except OSError as error:
            if error.errno in (errno.ENOENT, errno.ENODEV):
                try:
                    group.stat()
                except OSError as status:
                    if status.errno == errno.ENOENT:
                        return None
                    raise ScopeUncertain("scope existence unavailable") from status
            raise ScopeUncertain("scope population unavailable") from error
        try:
            pairs = [line.split() for line in lines]
            fields = dict(pairs)
        except ValueError as error:
            raise ScopeUncertain("scope population malformed") from error
        if len(fields) != len(pairs) or fields.get("populated") not in ("0", "1"):
            raise ScopeUncertain("scope population malformed")
        return fields["populated"] == "1"

    def drained(self, deadline):
        """Report empty only after exact unit and cgroup checks on both sides."""
        if self.identity is None:
            raise ScopeUncertain("scope was never pinned")
        before = self._query(deadline)
        if before["LoadState"] == "loaded":
            self._same(before)
        elif (
            before["LoadState"] != "not-found"
            or before.get("InvocationID", "") not in ("", self.identity.invocation)
            or before.get("ControlGroup", "") != ""
            or before.get("ActiveState", "inactive") != "inactive"
            or before.get("Delegate", "") not in ("", "no")
        ):
            raise ScopeUncertain("scope disappearance is ambiguous")
        population = self._population()
        after = self._query(deadline)
        if after != before:
            raise ScopeUncertain("scope changed during drain observation")
        if before["LoadState"] == "not-found":
            return population is None
        return before["ActiveState"] in ("inactive", "failed") and population is False

    def signal(self, signum, deadline):
        """Signal only the exact currently pinned scope; sampled, not fenced."""
        if signum not in ("TERM", "KILL"):
            raise ValueError("unsupported scope signal")
        fields = self._query(deadline)
        self._same(fields)
        if fields["ActiveState"] not in ("active", "activating", "deactivating"):
            raise ScopeUncertain("scope is not signalable")
        try:
            self.command(
                [
                    "systemctl",
                    "--user",
                    "kill",
                    "--kill-whom=all",
                    "--signal=" + signum,
                    self.unit,
                ],
                self._remaining(deadline),
            )
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            raise ScopeUncertain("scope signal failed") from error
        self._remaining(deadline)
