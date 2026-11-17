"""Optional startup wrapper for the fixed 2080 service.

The owner must explicitly configure systemd to run this exact file; preparing it
changes no unit. Hashes bind pre-exec inputs to this process, but do not attest
model weights, dynamically loaded libraries, or a malicious same-user race.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys

RUN = Path("/home/ben/fifi-scratch/a81ebdf26a2a/qwen36-2080ti")
START = RUN / "start.sh"
CONFIG = RUN / "qwen36-2080ti.json"
BUILD_ENV = RUN / "build-env.sh"
BINARY = RUN / "target" / "release" / "enginesrv"
ATTESTOR = Path(__file__).resolve()
APPROVED_START_SHA256 = (
    "67fce39d8ed2ec714da6f252e9bd0fd0e1dbc30dc6e0b83bf101b49f10c8000f"
)
APPROVED_BUILD_ENV_SHA256 = (
    "68a7ef8a7878646c44ab1a367474060e1800d7922ab0ef7b7d767289e0db56e8"
)
INPUTS = {
    "FIFI_HANDOFF_START_SHA256": START,
    "FIFI_HANDOFF_CONFIG_SHA256": CONFIG,
    "FIFI_HANDOFF_BUILD_ENV_SHA256": BUILD_ENV,
    "FIFI_HANDOFF_BINARY_SHA256": BINARY,
    "FIFI_HANDOFF_ATTESTOR_SHA256": ATTESTOR,
    "FIFI_HANDOFF_PYTHON_SHA256": Path("/usr/bin/python3"),
    "FIFI_HANDOFF_ENV_BINARY_SHA256": Path("/usr/bin/env"),
    "FIFI_HANDOFF_SYSTEMCTL_SHA256": Path("/usr/bin/systemctl"),
    "FIFI_HANDOFF_BUSCTL_SHA256": Path("/usr/bin/busctl"),
}


def identity(path):
    with path.open("rb") as source:
        before = os.fstat(source.fileno())
        h = hashlib.sha256()
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
        after = os.fstat(source.fileno())
    current = os.stat(path)
    a = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    b = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    c = (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns)
    if a != b or a != c:
        raise RuntimeError("launcher input changed while hashing")
    return a, h.hexdigest()


def _fingerprint_lines(out, expected_keys=None):
    lines = out.splitlines()
    keys = [line.partition(b"=")[0] for line in lines if b"=" in line]
    if len(keys) != len(lines) or len(set(keys)) != len(keys):
        raise RuntimeError("cannot parse manager/unit environment")
    if expected_keys is not None and set(keys) != expected_keys:
        raise RuntimeError("incomplete unit environment metadata")
    return hashlib.sha256(b"\0".join(sorted(lines))).hexdigest()


def manager_fingerprint(out=None):
    if out is None:
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


def inherited_manager_matches(raw):
    """Reject manager changes between service fork and launcher receipt."""
    _fingerprint_lines(raw)
    for line in raw.splitlines():
        words = shlex.split(os.fsdecode(line))
        if len(words) != 1 or "=" not in words[0]:
            raise RuntimeError("unsupported manager environment syntax")
        key, value = words[0].split("=", 1)
        if os.environ.get(key) != value:
            return False
    return True


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


def main():
    if not os.path.samefile("/proc/self/exe", "/usr/bin/python3"):
        raise RuntimeError("expected system Python interpreter")
    bash_name = shutil.which("bash")
    if not bash_name:
        raise RuntimeError("bash not found in launch PATH")
    bash = Path(bash_name).resolve(strict=True)
    inputs = {**INPUTS, "FIFI_HANDOFF_BASH_SHA256": bash}
    identities = {key: identity(path) for key, path in inputs.items()}
    if (
        identities["FIFI_HANDOFF_START_SHA256"][1] != APPROVED_START_SHA256
        or identities["FIFI_HANDOFF_BUILD_ENV_SHA256"][1] != APPROVED_BUILD_ENV_SHA256
    ):
        raise RuntimeError("launcher or sourced environment is not reviewed")
    # Refuse file swaps during receipt creation rather than attest a mixture.
    for key, path in inputs.items():
        if identity(path) != identities[key]:
            raise RuntimeError("launcher input changed while creating receipt")
    unit = "fifi-qwen36-2080ti.service"
    if not environment_files_empty(unit):
        raise RuntimeError("unit EnvironmentFiles are unsupported")
    manager_state = subprocess.check_output(
        ["/usr/bin/systemctl", "--user", "show-environment"], timeout=5
    )
    if not inherited_manager_matches(manager_state):
        raise RuntimeError("manager environment changed after service fork")
    manager_digest = manager_fingerprint(manager_state)
    unit_digest = unit_environment_fingerprint(unit)
    # Detect a change while constructing the receipt; this is sampled, not a
    # lock on the user manager's environment.
    if (
        manager_fingerprint() != manager_digest
        or unit_environment_fingerprint(unit) != unit_digest
        or not environment_files_empty(unit)
    ):
        raise RuntimeError("manager or unit environment changed during receipt")
    env = os.environ.copy()
    env["FIFI_HANDOFF_MANAGER_ENV_SHA256"] = manager_digest
    env["FIFI_HANDOFF_UNIT_ENV_SHA256"] = unit_digest
    env.update({key: value[1] for key, value in identities.items()})
    env["FIFI_HANDOFF_BASH_PATH"] = str(bash)
    env["FIFI_HANDOFF_LAUNCH_PID"] = str(os.getpid())
    env["FIFI_HANDOFF_LAUNCH_START"] = (
        Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]
    )
    os.execve(str(START), [str(START)], env)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"attested launcher refused: {type(error).__name__}", file=sys.stderr)
        sys.exit(1)
