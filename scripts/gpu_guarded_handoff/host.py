"""Fail-closed host checks for the existing RTX 2080 Ti serving instance.

These checks do not prove a verified router connection is idle: any established
client is rejected. This module is an operational tool for this host, not a
security boundary against same-user systemctl, signals, or a lease-file swap.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import time
import urllib.request

from . import scope

TARGET_GPU = "GPU-699004f8-e2f0-747e-5c02-30141e9d4b7c"
PAIR_GPUS = frozenset(
    {
        "GPU-704fef60-5280-9b88-fa3d-d11135caf576",
        "GPU-282e266a-24ac-6fa8-a16d-4281e52b4c0a",
    }
)
UNIT = "fifi-qwen36-2080ti.service"
PRIMARY = "fifi-qwen36-engine.service"
RUN = Path("/home/ben/fifi-scratch/a81ebdf26a2a/qwen36-2080ti")
START = RUN / "start.sh"
CONFIG = RUN / "qwen36-2080ti.json"
BUILD_ENV = RUN / "build-env.sh"
BINARY = RUN / "target" / "release" / "enginesrv"
KEY = RUN / "api-key"
ATTESTOR = Path(__file__).with_name("attested_launch.py").resolve()
# Only these reviewed launch scripts are supported. If an owner changes either,
# review all sourced dependencies and update both this policy and launcher.
APPROVED_START_SHA256 = (
    "67fce39d8ed2ec714da6f252e9bd0fd0e1dbc30dc6e0b83bf101b49f10c8000f"
)
APPROVED_BUILD_ENV_SHA256 = (
    "68a7ef8a7878646c44ab1a367474060e1800d7922ab0ef7b7d767289e0db56e8"
)
PORT = 18793
BUILDERS = frozenset(
    {"cargo", "cargo-clippy", "rustc", "nvcc", "cicc", "ptxas", "cc1plus"}
)
VOLATILE_ENV = frozenset(
    {
        b"SYSTEMD_EXEC_PID",
        b"INVOCATION_ID",
        b"JOURNAL_STREAM",
        b"MEMORY_PRESSURE_WATCH",
        b"MEMORY_PRESSURE_WRITE",
        b"MANAGERPIDFDID",
        b"LISTEN_PID",
        b"NOTIFY_SOCKET",
        b"WATCHDOG_PID",
        b"WATCHDOG_USEC",
        b"_",
        b"FIFI_HANDOFF_LAUNCH_PID",
        b"FIFI_HANDOFF_LAUNCH_START",
    }
)
ATTESTATION = {
    b"FIFI_HANDOFF_START_SHA256": START,
    b"FIFI_HANDOFF_CONFIG_SHA256": CONFIG,
    b"FIFI_HANDOFF_BUILD_ENV_SHA256": BUILD_ENV,
    b"FIFI_HANDOFF_BINARY_SHA256": BINARY,
    b"FIFI_HANDOFF_ATTESTOR_SHA256": ATTESTOR,
    b"FIFI_HANDOFF_PYTHON_SHA256": Path("/usr/bin/python3"),
    b"FIFI_HANDOFF_ENV_BINARY_SHA256": Path("/usr/bin/env"),
    b"FIFI_HANDOFF_SYSTEMCTL_SHA256": Path("/usr/bin/systemctl"),
    b"FIFI_HANDOFF_BUSCTL_SHA256": Path("/usr/bin/busctl"),
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inode(path):
    st = os.stat(path)
    return st.st_dev, st.st_ino


def process_environment(pid):
    result = {}
    for item in (Path("/proc") / str(pid) / "environ").read_bytes().split(b"\0"):
        if not item:
            continue
        key, separator, value = item.partition(b"=")
        if not separator or not key or key in result:
            raise RuntimeError("malformed or duplicate process environment key")
        result[key] = value
    return result


def environment_fingerprint(env):
    # Keep all values, including secrets, in memory only. Never put this digest
    # or its inputs into logs/receipts. Exclude manager-generated per-start IDs.
    return hashlib.sha256(
        b"\0".join(k + b"=" + env[k] for k in sorted(env) if k not in VOLATILE_ENV)
    ).digest()


def _fingerprint_lines(out, expected_keys=None):
    lines = out.splitlines()
    keys = [line.partition(b"=")[0] for line in lines if b"=" in line]
    if len(keys) != len(lines) or len(set(keys)) != len(keys):
        raise RuntimeError("cannot parse manager/unit environment")
    if expected_keys is not None and set(keys) != expected_keys:
        raise RuntimeError("incomplete unit environment metadata")
    return hashlib.sha256(b"\0".join(sorted(lines))).hexdigest()


def manager_fingerprint():
    out = subprocess.check_output(
        ["/usr/bin/systemctl", "--user", "show-environment"], timeout=5
    )
    return _fingerprint_lines(out)


def unit_environment_fingerprint(unit):
    keys = (
        b"Environment",
        b"PassEnvironment",
        b"UnsetEnvironment",
        b"FragmentPath",
        b"DropInPaths",
    )
    out = subprocess.check_output(
        [
            "/usr/bin/systemctl",
            "--user",
            "show",
            unit,
            *[arg for key in keys for arg in ("-p", key.decode())],
        ],
        timeout=5,
    )
    for line in out.splitlines():
        key, _, value = line.partition(b"=")
        if key in (b"Environment", b"PassEnvironment", b"UnsetEnvironment") and value:
            raise RuntimeError("unit environment overrides are unsupported")
    return _fingerprint_lines(out, set(keys))


def environment_files_empty(unit):
    encoded = "".join(
        c if c.isascii() and c.isalnum() else f"_{ord(c):02x}" for c in unit
    )
    path = "/org/freedesktop/systemd1/unit/" + encoded
    out = subprocess.check_output(
        [
            "/usr/bin/busctl",
            "--user",
            "get-property",
            "org.freedesktop.systemd1",
            path,
            "org.freedesktop.systemd1.Service",
            "EnvironmentFiles",
        ],
        timeout=5,
    )
    return out.strip() == b"a(sb) 0"


def attested_execstart(unit):
    if unit.get("NeedDaemonReload") != "no":
        return False
    try:
        argv = shlex.split(unit["ExecStart"].split("argv[]=", 1)[1].split(" ;", 1)[0])
        return (
            len(argv) == 2
            and os.path.samefile(argv[0], "/usr/bin/python3")
            and argv[1] == str(ATTESTOR)
        )
    except (OSError, ValueError, KeyError, IndexError):
        return False


def unit_metadata(unit):
    """Pin loaded unit configuration, excluding dynamic process status."""
    if unit.get("NeedDaemonReload") != "no":
        raise RuntimeError("unit configuration needs reload")
    fragment, dropins, command = (
        unit.get("FragmentPath"),
        unit.get("DropInPaths"),
        unit.get("ExecStart"),
    )
    if not fragment or dropins is None or not command or "argv[]=" not in command:
        raise RuntimeError("cannot determine loaded unit metadata")
    return {
        "fragment": fragment,
        "dropins": dropins,
        "command": command.split(" ; start_time=", 1)[0],
    }


def metadata_matches(unit, expected):
    try:
        return unit_metadata(unit) == expected
    except (RuntimeError, TypeError):
        return False


def proc_stat(pid):
    fields = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
    return int(fields[1]), int(fields[19])


def ancestors():
    found = set()
    pid = os.getpid()
    while pid > 1 and pid not in found:
        found.add(pid)
        try:
            pid, _ = proc_stat(pid)
        except (FileNotFoundError, ProcessLookupError):
            break
    return found


def assistant_state():
    """Snapshot Claude and Codex app-server owners without logging their arguments."""
    root = Path("/proc")
    ignored = ancestors()
    owners = {}
    candidates = []
    for p in root.iterdir():
        if not p.name.isdigit():
            continue
        try:
            parent, start = proc_stat(int(p.name))
            name = (p / "comm").read_text().strip()
            candidates.append((p, parent, start, name))
            argv = (p / "cmdline").read_bytes()
            if name == "claude" or (name == "codex" and b"app-server\0" in argv):
                owners[int(p.name)] = (
                    start,
                    os.readlink(p / "exe"),
                    inode(p / "exe"),
                    argv,
                )
        except (FileNotFoundError, ProcessLookupError):
            continue
    children = {}
    for p, parent, start, name in candidates:
        if parent not in owners or int(p.name) in ignored:
            continue
        try:
            argv = (p / "cmdline").read_bytes()
            exe = os.readlink(p / "exe")
            identity = (start, exe, inode(p / "exe"), argv)
            persistent = (
                name == "uv"
                or name.startswith("npm exec")
                or name in {"pyright-langser", "ssh-mcp", "serena", "codex-code-mode"}
                or (name == "python3" and b"/searxng-mcp.py\0" in argv)
            )
            children[(parent, int(p.name))] = (identity, persistent)
        except (FileNotFoundError, ProcessLookupError):
            continue
    return owners, children


def assistant_conflict(baseline, current):
    old_owners, old_children = baseline
    owners, children = current
    # An exiting Claude can orphan a still-running child. Treat disappearing,
    # newly started, reused or exec-changed owners as a conflict, too.
    if owners != old_owners:
        return True
    # A disappearing child may leave a detached grandchild behind. Reject any
    # addition, removal, PID reuse or exec/argument change until readmitted.
    if children != old_children:
        return True
    for _key, (_identity, persistent) in children.items():
        if not persistent:
            return True
    return False


class Host2080:
    def __init__(self, min_free_mib, restore_timeout=240):
        if min_free_mib < 2048:
            raise ValueError("at least 2048 MiB must be declared")
        self.min_free_mib = min_free_mib
        self.restore_timeout = restore_timeout
        self.snapshot_value = None
        self.assistant_baseline = None
        self.environment_digest = None
        self.key_digest = None
        self.manager_digest = None
        self.unit_environment_digest = None

    def state(self, unit):
        out = subprocess.check_output(
            [
                "/usr/bin/systemctl",
                "--user",
                "show",
                unit,
                "-p",
                "ActiveState",
                "-p",
                "MainPID",
                "-p",
                "Job",
                "-p",
                "FragmentPath",
                "-p",
                "DropInPaths",
                "-p",
                "ExecStart",
                "-p",
                "NeedDaemonReload",
            ],
            text=True,
            timeout=5,
        )
        return dict(line.split("=", 1) for line in out.splitlines())

    def live_identity(self, pid):
        _, start = proc_stat(pid)
        return start, os.readlink(f"/proc/{pid}/exe")

    def uses_config(self, pid):
        argv = (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
        indices = [i for i, value in enumerate(argv) if value == b"--config"]
        return (
            len(indices) == 1
            and indices[0] + 1 < len(argv)
            and argv[indices[0] + 1] == os.fsencode(str(CONFIG))
        )

    def environment_matches(self, pid):
        return (
            self.environment_digest is not None
            and environment_fingerprint(process_environment(pid))
            == self.environment_digest
        )

    def attestation_matches(self, pid, starttime):
        env = process_environment(pid)
        if (
            digest(START) != APPROVED_START_SHA256
            or digest(BUILD_ENV) != APPROVED_BUILD_ENV_SHA256
        ):
            return False
        try:
            bash = Path(os.fsdecode(env[b"FIFI_HANDOFF_BASH_PATH"]))
            bash_valid = (
                bash.is_absolute()
                and env[b"FIFI_HANDOFF_BASH_SHA256"] == digest(bash).encode()
            )
        except (KeyError, OSError, ValueError):
            return False
        return (
            env.get(b"FIFI_HANDOFF_LAUNCH_PID") == str(pid).encode()
            and env.get(b"FIFI_HANDOFF_LAUNCH_START") == str(starttime).encode()
            and env.get(b"FIFI_HANDOFF_MANAGER_ENV_SHA256")
            == manager_fingerprint().encode()
            and env.get(b"FIFI_HANDOFF_UNIT_ENV_SHA256")
            == unit_environment_fingerprint(UNIT).encode()
            and bash_valid
            and all(
                env.get(key) == digest(path).encode()
                for key, path in ATTESTATION.items()
            )
        )

    def processes(self):
        out = subprocess.check_output(
            [
                "/usr/bin/nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
        return [(r[0].strip(), int(r[1])) for r in csv.reader(io.StringIO(out))]

    def active_clients(self):
        out = subprocess.check_output(
            ["/usr/bin/ss", "-Htn", "state", "established", f"( sport = :{PORT} )"],
            text=True,
            timeout=5,
        )
        return bool(out.strip())

    def listening(self):
        out = subprocess.check_output(
            ["/usr/bin/ss", "-Hltn", f"( sport = :{PORT} )"], text=True, timeout=5
        )
        return bool(out.strip())

    def port_open(self):
        with socket.socket() as sock:
            sock.settimeout(1)
            return sock.connect_ex(("127.0.0.1", PORT)) == 0

    def quiet(self):
        out = subprocess.check_output(
            [
                "/usr/bin/nvidia-smi",
                "-i",
                ",".join([TARGET_GPU, *sorted(PAIR_GPUS)]),
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
        return len(out.splitlines()) == 3 and all(int(v) == 0 for v in out.splitlines())

    def device_free_mib(self):
        out = subprocess.check_output(
            [
                "/usr/bin/nvidia-smi",
                "-i",
                TARGET_GPU,
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
        return int(out.strip())

    def admission_memory_possible(self, snapshot):
        gpu = subprocess.check_output(
            [
                "/usr/bin/nvidia-smi",
                "-i",
                TARGET_GPU,
                "--query-gpu=memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
        rows = list(csv.reader(io.StringIO(gpu)))
        if len(rows) != 1 or len(rows[0]) != 2:
            raise RuntimeError("cannot determine device capacity")
        total, free = (int(v.strip()) for v in rows[0])
        apps = subprocess.check_output(
            [
                "/usr/bin/nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
        target = [
            r for r in csv.reader(io.StringIO(apps)) if r[0].strip() == TARGET_GPU
        ]
        if len(target) != 1 or int(target[0][1]) != snapshot["pid"]:
            raise RuntimeError("cannot establish sole service memory ownership")
        resident = int(target[0][2].strip())
        return (
            0 <= free <= total
            and 0 <= resident <= total
            and free + resident >= self.min_free_mib
        )

    def ready(self, snapshot):
        req = urllib.request.Request(
            f"http://127.0.0.1:{PORT}/v1/system",
            headers={"Authorization": "Bearer " + KEY.read_text().strip()},
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            engines = json.load(response)["engines"]
        return any(
            e.get("model") == snapshot["model"]
            and e.get("state") == "ready"
            and e.get("context_limit") == snapshot["context"]
            for e in engines
        )

    def competing(self, owner_scope=None):
        current = assistant_state()
        if self.assistant_baseline is None or assistant_conflict(
            self.assistant_baseline, current
        ):
            return True
        for p in Path("/proc").iterdir():
            if not p.name.isdigit():
                continue
            try:
                if (p / "comm").read_text().strip() in BUILDERS:
                    pid = int(p.name)
                    if owner_scope is None or not owner_scope.contains(pid):
                        return True
            except (FileNotFoundError, ProcessLookupError):
                continue
        return False

    def snapshot(self):
        self.assistant_baseline = assistant_state()
        if assistant_conflict(self.assistant_baseline, self.assistant_baseline):
            raise RuntimeError("other assistant work is active")
        unit, primary = self.state(UNIT), self.state(PRIMARY)
        for s in (unit, primary):
            if (
                s.get("ActiveState") != "active"
                or s.get("Job") != ""
                or int(s.get("MainPID", "0")) <= 0
            ):
                raise RuntimeError("both services must be settled and active")
        if not attested_execstart(unit) or primary.get("NeedDaemonReload") != "no":
            raise RuntimeError(
                "unit must use the pinned attested launcher and be reloaded"
            )
        pid, primary_pid = int(unit["MainPID"]), int(primary["MainPID"])
        starttime, exe = self.live_identity(pid)
        primary_starttime, primary_exe = self.live_identity(primary_pid)
        env = process_environment(pid)
        if (
            not self.uses_config(pid)
            or not os.path.samefile(exe, f"/proc/{pid}/exe")
            or not os.path.samefile(exe, BINARY)
            or not self.attestation_matches(pid, starttime)
        ):
            raise RuntimeError("live service executable/config differs from expected")
        self.environment_digest = environment_fingerprint(env)
        self.key_digest = digest(KEY)
        self.manager_digest = manager_fingerprint()
        self.unit_environment_digest = unit_environment_fingerprint(UNIT)
        if not environment_files_empty(UNIT) or not environment_files_empty(PRIMARY):
            raise RuntimeError("unit EnvironmentFiles are unsupported")
        cfg = json.loads(CONFIG.read_text())
        if cfg.get("listen") != f"127.0.0.1:{PORT}" or len(cfg.get("engines", [])) != 1:
            raise RuntimeError("unexpected 2080 service configuration")
        engine = cfg["engines"][0]
        unit_files = []
        for s in (unit, primary):
            unit_files.extend([s["FragmentPath"], *s.get("DropInPaths", "").split()])
        files = [
            str(START),
            str(CONFIG),
            str(BUILD_ENV),
            str(ATTESTOR),
            str(Path(exe).resolve(strict=True)),
            str(Path(primary_exe).resolve(strict=True)),
            *unit_files,
        ]
        snapshot = {
            "pid": pid,
            "starttime": starttime,
            "exe": exe,
            "primary_pid": primary_pid,
            "primary_starttime": primary_starttime,
            "primary_exe": primary_exe,
            "exe_inode": inode(f"/proc/{pid}/exe"),
            "primary_inode": inode(f"/proc/{primary_pid}/exe"),
            "unit_meta": unit_metadata(unit),
            "primary_meta": unit_metadata(primary),
            "files": {p: digest(p) for p in files},
            "model": engine["model"],
            "context": engine["qwen36"]["context_limit"],
        }
        self.snapshot_value = snapshot
        return snapshot

    def was_active(self, snapshot):
        return True

    def inspect_files(self, snapshot):
        return (
            self.key_digest is not None
            and digest(KEY) == self.key_digest
            and self.manager_digest is not None
            and manager_fingerprint() == self.manager_digest
            and unit_environment_fingerprint(UNIT) == self.unit_environment_digest
            and environment_files_empty(UNIT)
            and environment_files_empty(PRIMARY)
            and all(digest(p) == h for p, h in snapshot["files"].items())
        )

    def stable_primary(self, snapshot, records):
        primary = self.state(PRIMARY)
        pid = snapshot["primary_pid"]
        return (
            primary.get("ActiveState") == "active"
            and primary.get("Job") == ""
            and metadata_matches(primary, snapshot["primary_meta"])
            and int(primary.get("MainPID", "0")) == pid
            and self.live_identity(pid)
            == (snapshot["primary_starttime"], snapshot["primary_exe"])
            and inode(f"/proc/{pid}/exe") == snapshot["primary_inode"]
            and {g for g, p in records if g in PAIR_GPUS and p == pid} == PAIR_GPUS
            and all(p == pid for g, p in records if g in PAIR_GPUS)
        )

    def same_live(self, snapshot, records):
        state = self.state(UNIT)
        pid = snapshot["pid"]
        return (
            state.get("ActiveState") == "active"
            and state.get("Job") == ""
            and metadata_matches(state, snapshot["unit_meta"])
            and int(state.get("MainPID", "0")) == pid
            and self.live_identity(pid) == (snapshot["starttime"], snapshot["exe"])
            and self.uses_config(pid)
            and self.environment_matches(pid)
            and self.attestation_matches(pid, snapshot["starttime"])
            and attested_execstart(state)
            and inode(f"/proc/{pid}/exe") == snapshot["exe_inode"]
            and os.path.samefile(snapshot["exe"], f"/proc/{pid}/exe")
            and [(g, p) for g, p in records if g == TARGET_GPU] == [(TARGET_GPU, pid)]
        )

    def still_matches(self, snapshot):
        records = self.processes()
        return (
            self.inspect_files(snapshot)
            and self.same_live(snapshot, records)
            and self.stable_primary(snapshot, records)
            and not self.active_clients()
            and not self.competing()
            and self.ready(snapshot)
        )

    def admit(self, snapshot):
        scope.check_manager()
        for _ in range(3):
            if (
                not self.still_matches(snapshot)
                or not self.quiet()
                or not self.admission_memory_possible(snapshot)
            ):
                raise RuntimeError(
                    "service, clients, GPU, or competing work is not idle"
                )
            time.sleep(1)

    def stop(self, snapshot):
        subprocess.run(
            ["/usr/bin/systemctl", "--user", "stop", UNIT], check=True, timeout=45
        )

    def verify_stopped(self, snapshot):
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if (
                not any(g == TARGET_GPU for g, _ in self.processes())
                and not self.port_open()
                and not self.listening()
                and not self.active_clients()
            ):
                if self.device_free_mib() < self.min_free_mib:
                    raise RuntimeError("insufficient device memory")
                if (
                    not self.inspect_files(snapshot)
                    or not self.stable_primary(snapshot, self.processes())
                    or self.competing()
                    or not self.quiet()
                ):
                    raise RuntimeError("service files or pair identity changed")
                return
            time.sleep(1)
        raise RuntimeError("service did not release its GPU and listener")

    def restore(self, snapshot):
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            state = self.state(UNIT)
            if state.get("Job") or state.get("ActiveState") in (
                "deactivating",
                "activating",
                "reloading",
            ):
                time.sleep(1)
                continue
            if state.get("ActiveState") == "active":
                return
            if state.get("ActiveState") in ("inactive", "failed"):
                break
            raise RuntimeError("unexpected service state")
        else:
            raise RuntimeError("stop did not settle")
        if (
            not self.inspect_files(snapshot)
            or not metadata_matches(self.state(UNIT), snapshot["unit_meta"])
            or not self.stable_primary(snapshot, self.processes())
            or any(g == TARGET_GPU for g, _ in self.processes())
            or self.port_open()
            or self.listening()
            or self.active_clients()
        ):
            raise RuntimeError("cannot safely restore pinned service")
        subprocess.run(
            ["/usr/bin/systemctl", "--user", "start", "--no-block", UNIT],
            check=True,
            timeout=10,
        )

    def verify_restored(self, snapshot):
        deadline = time.monotonic() + self.restore_timeout
        while True:
            if not self.inspect_files(snapshot):
                return False
            unit = self.state(UNIT)
            pid = int(unit.get("MainPID", "0"))
            if (
                unit.get("ActiveState") == "active"
                and unit.get("Job") == ""
                and metadata_matches(unit, snapshot["unit_meta"])
                and pid > 0
            ):
                starttime, exe = self.live_identity(pid)
                records = self.processes()
                if (
                    os.path.samefile(exe, snapshot["exe"])
                    and os.path.samefile(snapshot["exe"], f"/proc/{pid}/exe")
                    and inode(f"/proc/{pid}/exe") == snapshot["exe_inode"]
                    and self.uses_config(pid)
                    and self.environment_matches(pid)
                    and self.attestation_matches(pid, starttime)
                    and attested_execstart(unit)
                    and [(g, p) for g, p in records if g == TARGET_GPU]
                    == [(TARGET_GPU, pid)]
                    and self.stable_primary(snapshot, records)
                    and self.ready(snapshot)
                ):
                    return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(2)

    def monitor(self, owner_scope):
        snapshot = self.snapshot_value
        records = self.processes()
        state = self.state(UNIT)
        if (
            not self.inspect_files(snapshot)
            or state.get("ActiveState") != "inactive"
            or state.get("Job") != ""
            or not metadata_matches(state, snapshot["unit_meta"])
            or self.competing(owner_scope)
            or not self.stable_primary(snapshot, records)
            or self.active_clients()
            or self.listening()
        ):
            raise RuntimeError(
                "competing work or service listener appeared during handoff"
            )
        for gpu, pid in records:
            if gpu == TARGET_GPU and not owner_scope.contains(pid):
                raise RuntimeError("foreign process on 2080")
        out = subprocess.check_output(
            [
                "/usr/bin/nvidia-smi",
                "-i",
                ",".join(sorted(PAIR_GPUS)),
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
        if len(out.splitlines()) != 2 or any(int(v) != 0 for v in out.splitlines()):
            raise RuntimeError("pair GPU utilization detected")
