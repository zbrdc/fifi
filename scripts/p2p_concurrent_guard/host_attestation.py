"""Linux host-attestation primitives for a future joint supervisor.

No entry point or service action is provided here. Process snapshots and a
sealed companion do not prove exclusive access to a server, lease ownership,
request drain or admission. A live adapter must provide those separately.
"""

from dataclasses import dataclass
import fcntl
import hashlib
import os
from pathlib import Path
import stat


# The integer-clock companion revision independently reviewed by its owner.
COMPANION_SHA256 = "5c599696a6ad1a0b7ba3228b992fa57cb15ad9f55bb02036ca194ecf826abfbf"
GATE_SHA256 = "fd1250b97b6a0546d495be01d59e869719dbcb50a1326ef1c58c061096595184"
MAX_COMPANION_BYTES = 1 << 20


def digest_open(fd):
    st = os.fstat(fd)
    if not stat.S_ISREG(st.st_mode):
        raise ValueError("fingerprinted input must be a regular file")
    digest = hashlib.sha256()
    while chunk := os.read(fd, 1024 * 1024):
        digest.update(chunk)
    after = os.fstat(fd)
    if (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ValueError("fingerprinted input changed while reading")
    return digest.hexdigest()


def digest_file(path):
    """Hash a regular, no-follow file without recording contents."""
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        return digest_open(fd)
    finally:
        os.close(fd)


def _seal_pinned(path, expected):
    """Copy exactly the pinned companion to a sealed, non-writable memfd.

    The caller must pass this descriptor to the intended contained child and
    retain/close its own descriptor. A verified memfd does not prove that any
    child or systemd scope actually executed it.
    """
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("invalid expected digest")
    src = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    memfd = None
    try:
        st = os.fstat(src)
        if not stat.S_ISREG(st.st_mode) or not 0 < st.st_size <= MAX_COMPANION_BYTES:
            raise ValueError("companion must be a bounded regular file")
        content = bytearray()
        while chunk := os.read(src, min(65536, MAX_COMPANION_BYTES + 1 - len(content))):
            content.extend(chunk)
            if len(content) > MAX_COMPANION_BYTES:
                raise ValueError("companion exceeds size limit")
        after = os.fstat(src)
        if (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) or len(content) != st.st_size:
            raise ValueError("companion changed while reading")
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError("companion digest mismatch")
        memfd = os.memfd_create(
            "fifi-2080-companion", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING
        )
        view = memoryview(content)
        while view:
            written = os.write(memfd, view)
            if written <= 0:
                raise OSError("failed to copy companion")
            view = view[written:]
        seals = (
            fcntl.F_SEAL_WRITE
            | fcntl.F_SEAL_GROW
            | fcntl.F_SEAL_SHRINK
            | fcntl.F_SEAL_SEAL
        )
        fcntl.fcntl(memfd, fcntl.F_ADD_SEALS, seals)
        if fcntl.fcntl(memfd, fcntl.F_GET_SEALS) & seals != seals:
            raise OSError("companion memfd seals were not applied")
        os.lseek(memfd, 0, os.SEEK_SET)
        result, memfd = memfd, None
        return result
    finally:
        os.close(src)
        if memfd is not None:
            os.close(memfd)


def sealed_companion(path):
    """Seal the one explicitly pinned 2080 companion version."""
    if COMPANION_SHA256 is None:
        raise RuntimeError("final companion version has not been pinned")
    return _seal_pinned(path, COMPANION_SHA256)


def sealed_gate(path):
    """Seal the exact reviewed gate bytes as a separate immutable program."""
    return _seal_pinned(path, GATE_SHA256)


def process_start_time(stat_text, expected_pid):
    """Extract /proc starttime while accepting parentheses in comm."""
    closing = stat_text.rfind(")")
    opening = stat_text.find("(")
    if (
        opening < 1
        or closing <= opening
        or stat_text[:opening].strip() != str(expected_pid)
    ):
        raise ValueError("malformed process stat")
    fields = stat_text[closing + 1 :].split()
    if len(fields) <= 19 or not fields[19].isdigit():
        raise ValueError("missing process start time")
    return int(fields[19])


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    start_time: int
    executable: str
    executable_sha256: str
    cmdline_sha256: str
    environ_sha256: str
    cgroup_sha256: str
    config_sha256: tuple


def process_identity(pid, config_paths, *, proc_root=Path("/proc")):
    """Snapshot process and config inputs; a changed or missing value fails.

    This detects ordinary restarts and changes, not a same-UID adversary. Unit,
    listening sockets and GPU ownership must be verified by the caller.
    """
    if type(pid) is not int or pid < 1:
        raise ValueError("invalid process PID")
    config_paths = tuple(Path(path) for path in config_paths)
    if not config_paths or any(not path.is_absolute() for path in config_paths):
        raise ValueError("absolute process config inputs are required")
    if len(set(config_paths)) != len(config_paths):
        raise ValueError("duplicate process config input")
    root = Path(proc_root) / str(pid)
    before = process_start_time((root / "stat").read_text(), pid)
    executable = os.readlink(root / "exe")
    if not executable.startswith("/") or executable.endswith(" (deleted)"):
        raise ValueError("process executable is not a stable absolute path")
    # The proc link identifies the binary actually mapped by the process. A
    # replaced pathname alone must not make us hash a different executable.
    exe_fd = os.open(root / "exe", os.O_RDONLY | os.O_CLOEXEC)
    try:
        mapped = os.fstat(exe_fd)
        named = os.stat(executable, follow_symlinks=False)
        if (mapped.st_dev, mapped.st_ino) != (named.st_dev, named.st_ino):
            raise ValueError("mapped executable differs from its pathname")
        executable_sha256 = digest_open(exe_fd)
    finally:
        os.close(exe_fd)
    hashed = [
        hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ("cmdline", "environ", "cgroup")
    ]
    configs = tuple((str(Path(path)), digest_file(path)) for path in config_paths)
    after = process_start_time((root / "stat").read_text(), pid)
    if after != before or os.readlink(root / "exe") != executable:
        raise ValueError("process identity changed during snapshot")
    return ProcessIdentity(pid, before, executable, executable_sha256, *hashed, configs)
