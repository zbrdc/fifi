"""Read-only, fail-closed observation of a running P2P pair service.

This is a partial host-backend slice, not an admission or handoff. The caller
must independently hold/verify the canonical lease and establish engine queue
readiness, graphics/unreported GPU contexts, the other GPU's identity, and
containment before any stop. Only reported compute-app owners are checked.
Sampling cannot prevent a new request or a same-user process changing state.
No credentials or environment values are written to logs by this module.
The running process's config argument is verified, but the supplied build-env
and key paths are caller assertions: this observer does not prove the launcher
reads them or enumerate model and restoration dependencies. A snapshot hashes
the executable several times and waits between utilization samples; it is not
suitable for per-second performance monitoring.

Derived from the read-only identity and no-client checks of the historical
pair reservation driver in the P2P evidence scratch directory; unlike that
driver, this module has no executable entry point or hardcoded host paths.
"""

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
import uuid

_UNIT_NAME = re.compile(r"[A-Za-z0-9_.@:-]+\.service\Z")
_GPU_UUID = re.compile(r"GPU-[0-9a-fA-F-]{36}\Z")
_UNIT_PROPERTIES = (
    "ActiveState",
    "MainPID",
    "Job",
    "FragmentPath",
    "DropInPaths",
    "NeedDaemonReload",
    "ExecStart",
    "ExecStartPre",
    "ExecStartPost",
    "ExecCondition",
    "ExecStop",
    "ExecStopPost",
    "Environment",
    "EnvironmentFiles",
    "PassEnvironment",
    "UnsetEnvironment",
    "WorkingDirectory",
    "ControlGroup",
    "RootDirectory",
    "User",
    "Group",
    "Slice",
    "Delegate",
    "Restart",
    "TimeoutStopUSec",
    "TimeoutStartUSec",
)
_DYNAMIC_PROPERTIES = {"ActiveState", "MainPID", "Job"}
_HOOKS = ("ExecStartPre", "ExecStartPost", "ExecCondition", "ExecStop", "ExecStopPost")
_PROVABLE_EMPTY = {name: b"a(sasbttttuii) 0" for name in _HOOKS}
_PROVABLE_EMPTY["EnvironmentFiles"] = b"a(sb) 0"


@dataclass(frozen=True)
class PairSpec:
    unit: str
    launcher: Path
    executable: Path
    config: Path
    build_environment: Path
    api_key: Path
    gpu_uuids: tuple[str, str]
    port: int

    @property
    def inputs(self):
        return self.launcher, self.config, self.build_environment, self.api_key

    def __post_init__(self):
        if not isinstance(self.unit, str) or not _UNIT_NAME.fullmatch(self.unit):
            raise ValueError("invalid pair unit name")
        if type(self.port) is not int or not 1 <= self.port <= 65535:
            raise ValueError("invalid pair port")
        if (
            not isinstance(self.gpu_uuids, tuple)
            or len(self.gpu_uuids) != 2
            or len(set(self.gpu_uuids)) != 2
            or any(
                not isinstance(g, str) or not _GPU_UUID.fullmatch(g)
                for g in self.gpu_uuids
            )
        ):
            raise ValueError("exactly two distinct GPU UUIDs required")
        if (
            any(not isinstance(p, Path) or not p.is_absolute() for p in self.inputs)
            or not isinstance(self.executable, Path)
            or not self.executable.is_absolute()
            or len(set(self.inputs)) != len(self.inputs)
        ):
            raise ValueError(
                "absolute, distinct launcher/config/environment/key paths required"
            )


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    start_ticks: int
    cgroup: str
    executable: str
    executable_sha256: str = field(repr=False)
    command_sha256: str = field(repr=False)
    environment_sha256: str = field(repr=False)


@dataclass(frozen=True)
class PairSnapshot:
    boot_id: str
    process: ProcessIdentity
    unit_object_path: str
    # The hashes cover configuration and environment values, potentially
    # including secrets. Do not print or persist them in logs or reports.
    unit_sha256: str = field(repr=False)
    manager_environment_sha256: str = field(repr=False)
    file_fingerprints: tuple = field(repr=False)
    gpu_owners: tuple
    listen_inodes: tuple


def _run(argv):
    # stderr may contain environment or credentials. Never place it in raised
    # exception text or logs, and never run a shell.
    return subprocess.check_output(argv, stderr=subprocess.DEVNULL, timeout=10)


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _file_fingerprint(path):
    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError("input path must be absolute")
    resolved = path.resolve(strict=True)
    with resolved.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if not os.path.isfile(resolved):
            raise ValueError("input is not a regular file")
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
        after = os.fstat(stream.fileno())

    def identity(s):
        return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns

    if identity(before) != identity(after) or path.resolve(strict=True) != resolved:
        raise RuntimeError("input changed while fingerprinting")
    # After the fd closes, ensure the resolved name still identifies that file.
    if identity(resolved.stat()) != identity(after):
        raise RuntimeError("input changed after fingerprinting")
    return (str(path), str(resolved), *identity(after), h.hexdigest())


def _parse_properties(raw):
    try:
        lines = raw.decode("utf-8").splitlines()
        fields = {}
        for line in lines:
            key, value = line.split("=", 1)
            if key in fields:
                raise ValueError("duplicate unit property")
            fields[key] = value
    except (UnicodeError, ValueError) as error:
        raise ValueError("invalid unit property response") from error
    if set(fields) - set(_UNIT_PROPERTIES):
        raise ValueError("unexpected unit properties")
    return fields


def _validated_properties(fields):
    if set(fields) != set(_UNIT_PROPERTIES):
        raise ValueError("missing unit properties")
    if fields["ActiveState"] != "active" or fields["Job"] != "":
        raise RuntimeError("pair service is not settled and active")
    try:
        pid = int(fields["MainPID"])
    except ValueError as error:
        raise ValueError("invalid pair service PID") from error
    if pid <= 0 or str(pid) != fields["MainPID"]:
        raise ValueError("invalid pair service PID")
    if fields["NeedDaemonReload"] != "no" or not fields["FragmentPath"]:
        raise RuntimeError("pair unit is stale or fragment missing")
    if fields["EnvironmentFiles"] or any(fields[name] for name in _HOOKS):
        raise RuntimeError("unit hooks or environment files need separate provenance")
    start = fields["ExecStart"]
    if start.count(" ; start_time=") != 1:
        raise RuntimeError("cannot normalize pair ExecStart")
    fields["ExecStart"] = start.split(" ; start_time=", 1)[0]
    return pid, fields


def _validate_exec_start(fields, launcher):
    match = re.fullmatch(
        r"\{ path=(.+?) ; argv\[\]=(.+?) ; ignore_errors=no", fields["ExecStart"]
    )
    if not match:
        raise RuntimeError("unrecognized pair ExecStart")
    try:
        argv = shlex.split(match.group(2))
        matches = Path(match.group(1)).resolve(strict=True) == launcher.resolve(
            strict=True
        )
    except (OSError, ValueError) as error:
        raise RuntimeError("cannot verify pair launcher") from error
    if argv != [match.group(1)] or not matches:
        raise RuntimeError("pair ExecStart does not match expected launcher")


def _validate_launcher(launcher, executable):
    try:
        lines = [
            line
            for line in launcher.read_text().splitlines()
            if line.startswith("exec ")
        ]
        commands = [shlex.split(line) for line in lines]
        if len(commands) != 1 or len(commands[0]) < 2:
            raise ValueError("launcher must have one explicit exec")
        binary = Path(commands[0][1])
        if not binary.is_absolute() or binary.resolve(
            strict=True
        ) != executable.resolve(strict=True):
            raise ValueError("launcher exec target differs from the expected binary")
    except (OSError, UnicodeError, ValueError) as error:
        raise RuntimeError("cannot verify launcher exec target") from error


def _process(proc, pid, expected_executable, expected_config):
    root = proc / str(pid)
    raw = (root / "stat").read_text(encoding="utf-8")
    # /proc stat's comm may contain spaces or right parentheses. The first
    # numeric field after the final ')' is field 3; starttime is field 22.
    head, sep, tail = raw.rpartition(")")
    fields = tail.split()
    if not sep or not head.startswith(f"{pid} (") or len(fields) < 20:
        raise RuntimeError("invalid process stat")
    try:
        start_ticks = int(fields[19])
    except ValueError as error:
        raise RuntimeError("invalid process start time") from error
    if start_ticks <= 0:
        raise RuntimeError("invalid process start time")
    executable = (root / "exe").resolve(strict=True)
    if executable != expected_executable.resolve(strict=True):
        raise RuntimeError("pair executable identity differs")
    cgroups = [
        line[3:]
        for line in (root / "cgroup").read_text().splitlines()
        if line.startswith("0::")
    ]
    if len(cgroups) != 1 or not cgroups[0].startswith("/"):
        raise RuntimeError("pair cgroup identity unavailable")
    command = (root / "cmdline").read_bytes()
    if not command or not command.endswith(b"\0"):
        raise RuntimeError("pair command identity unavailable")
    argv = [os.fsdecode(item) for item in command[:-1].split(b"\0")]
    if argv.count("--config") != 1 or any(arg.startswith("--config=") for arg in argv):
        raise RuntimeError("pair config argument is ambiguous")
    config_index = argv.index("--config")
    if config_index + 1 >= len(argv):
        raise RuntimeError("pair config argument is missing")
    config = Path(argv[config_index + 1])
    if not config.is_absolute() or config.resolve(
        strict=True
    ) != expected_config.resolve(strict=True):
        raise RuntimeError("pair config argument differs from pinned input")
    environment = (root / "environ").read_bytes()
    binary = _file_fingerprint(executable)
    return ProcessIdentity(
        pid,
        start_ticks,
        cgroups[0],
        str(executable),
        binary[-1],
        _digest(command),
        _digest(environment),
    )


def _gpu_owners(raw, uuids, pid):
    try:
        rows = []
        for line in raw.decode("ascii").splitlines():
            if not line.strip():
                continue
            parts = [part.strip() for part in line.split(",")]
            if (
                len(parts) != 2
                or not _GPU_UUID.fullmatch(parts[0])
                or not parts[1].isdigit()
            ):
                raise ValueError("invalid GPU owner row")
            if parts[0] in uuids:
                rows.append((parts[0], int(parts[1])))
    except UnicodeError as error:
        raise ValueError("invalid GPU owner response") from error
    if len(rows) != 2 or set(rows) != {(gpu, pid) for gpu in uuids}:
        raise RuntimeError(
            "reported pair compute-app owners do not match expected service"
        )
    return tuple(sorted(rows))


def _zero_utilization(raw, uuids):
    try:
        values = {}
        for line in raw.decode("ascii").splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 2 or parts[0] in values or not parts[1].isdigit():
                raise ValueError("invalid GPU utilization response")
            values[parts[0]] = int(parts[1])
    except UnicodeError as error:
        raise ValueError("invalid GPU utilization response") from error
    if set(values) != set(uuids) or any(values.values()):
        raise RuntimeError("pair GPU activity or unexpected UUID detected")


def _socket_inodes(proc, pid, port, raw):
    # ss catches TCP closing states as well as established clients; refuse all
    # non-listeners. A stable snapshot cannot fence future requests.
    try:
        for line in raw.decode("utf-8").splitlines():
            fields = line.split()
            if len(fields) < 5 or fields[0] != "LISTEN":
                raise RuntimeError("pair has a client or malformed socket row")
    except UnicodeError as error:
        raise ValueError("invalid socket response") from error
    listeners = set()
    for name in ("tcp", "tcp6"):
        rows = (proc / "net" / name).read_text(encoding="ascii").splitlines()
        if not rows:
            raise ValueError("missing TCP table header")
        for line in rows[1:]:
            parts = line.split()
            if len(parts) < 10:
                raise ValueError("malformed TCP table")
            try:
                local_port = int(parts[1].rsplit(":", 1)[1], 16)
            except (IndexError, ValueError) as error:
                raise ValueError("malformed TCP local port") from error
            if local_port == port:
                if parts[3] != "0A" or not parts[9].isdigit() or int(parts[9]) <= 0:
                    raise RuntimeError("pair socket is not a verified listener")
                listeners.add(parts[9])
    owned = set()
    for fd in (proc / str(pid) / "fd").iterdir():
        try:
            target = os.readlink(fd)
        except FileNotFoundError:
            continue
        if target.startswith("socket:[") and target.endswith("]"):
            owned.add(target[8:-1])
    if not listeners or not listeners <= owned:
        raise RuntimeError("pair listener ownership unverified")
    return tuple(sorted(listeners))


class LinuxPairObserver:
    """Read-only service observation; never starts or stops a service."""

    def __init__(self, spec, *, proc=Path("/proc"), command=_run, sleep=time.sleep):
        if not isinstance(spec, PairSpec) or not isinstance(proc, Path):
            raise ValueError("invalid observer configuration")
        self.spec, self.proc, self.command, self.sleep = spec, proc, command, sleep

    def _unit(self):
        command = ["systemctl", "--user", "show", "--all", self.spec.unit]
        for name in _UNIT_PROPERTIES:
            command.extend(["-p", name])
        fields = _parse_properties(self.command(command))
        missing = set(_UNIT_PROPERTIES) - set(fields)
        if missing - set(_PROVABLE_EMPTY):
            raise ValueError("missing unit properties without a supported proof")

        def unit_object():
            result = self.command(
                [
                    "busctl",
                    "--user",
                    "call",
                    "org.freedesktop.systemd1",
                    "/org/freedesktop/systemd1",
                    "org.freedesktop.systemd1.Manager",
                    "GetUnit",
                    "s",
                    self.spec.unit,
                ]
            )
            match = re.fullmatch(
                rb'o "(/org/freedesktop/systemd1/unit/[A-Za-z0-9_]+)"', result.strip()
            )
            if not match:
                raise RuntimeError("cannot resolve unit D-Bus object")
            return match.group(1).decode("ascii")

        obj = unit_object()
        for name in sorted(missing):
            result = self.command(
                [
                    "busctl",
                    "--user",
                    "get-property",
                    "org.freedesktop.systemd1",
                    obj,
                    "org.freedesktop.systemd1.Service",
                    name,
                ]
            )
            # Never print a nonempty property, especially EnvironmentFiles.
            if result.strip() != _PROVABLE_EMPTY[name]:
                raise RuntimeError("cannot prove missing unit property empty")
            fields[name] = ""
        if unit_object() != obj:
            raise RuntimeError("unit D-Bus object changed during property proof")
        pid, fields = _validated_properties(fields)
        _validate_exec_start(fields, self.spec.launcher)
        return pid, fields, obj

    def _observe(self):
        boot = (self.proc / "sys/kernel/random/boot_id").read_text().strip()
        if str(uuid.UUID(boot)) != boot:
            raise RuntimeError("invalid boot identity")
        pid, fields, unit_object = self._unit()
        process = _process(self.proc, pid, self.spec.executable, self.spec.config)
        if not fields["ControlGroup"] or process.cgroup != fields["ControlGroup"]:
            raise RuntimeError("pair process is not in the loaded unit control group")
        manager = self.command(["systemctl", "--user", "show-environment"])
        fragment = Path(fields["FragmentPath"])
        files = (
            fragment,
            *(Path(p) for p in shlex.split(fields["DropInPaths"])),
            *self.spec.inputs,
        )
        fingerprints = tuple(_file_fingerprint(p) for p in files)
        _validate_launcher(self.spec.launcher, self.spec.executable)
        unit = {
            key: value
            for key, value in fields.items()
            if key not in _DYNAMIC_PROPERTIES
        }
        unit_hash = _digest(
            json.dumps(unit, sort_keys=True, separators=(",", ":")).encode()
        )
        gpu_raw = self.command(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ]
        )
        owners = _gpu_owners(gpu_raw, self.spec.gpu_uuids, pid)
        socket_raw = self.command(["ss", "-Htan", f"( sport = :{self.spec.port} )"])
        sockets = _socket_inodes(self.proc, pid, self.spec.port, socket_raw)
        again_pid, again_fields, again_object = self._unit()
        if (
            again_pid != pid
            or again_fields != fields
            or again_object != unit_object
            or _process(self.proc, pid, self.spec.executable, self.spec.config)
            != process
        ):
            raise RuntimeError("pair process or loaded unit changed during observation")
        return PairSnapshot(
            boot,
            process,
            unit_object,
            unit_hash,
            _digest(manager),
            fingerprints,
            owners,
            sockets,
        )

    def snapshot(self):
        """Obtain stable, idle observations; this does not authorize service stop."""
        before = self._observe()
        for index in range(3):
            raw = self.command(
                [
                    "nvidia-smi",
                    "-i",
                    ",".join(sorted(self.spec.gpu_uuids)),
                    "--query-gpu=uuid,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ]
            )
            _zero_utilization(raw, self.spec.gpu_uuids)
            if index < 2:
                self.sleep(1)
        after = self._observe()
        if before != after:
            raise RuntimeError("pair identity changed across idle observation")
        return after

    def verify(self, snapshot):
        """Re-observe and require exact identity; raises on any uncertainty."""
        if not isinstance(snapshot, PairSnapshot) or self.snapshot() != snapshot:
            raise RuntimeError("pair identity or idle state changed")
        return True
