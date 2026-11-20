"""A start gate for a reviewed companion, not a complete supervisor.

No credentials, service state, network, or GPU access occurs before the
parent's pipe signal. The parent must first own the canonical lease and attest
the child's PID/start time and scope. This gate does not attest its parent,
scope, peer processes, GPU state, server drain, or campaign admission.
"""

import argparse
from dataclasses import dataclass
import fcntl
import hashlib
import math
import os
from pathlib import Path
import re
import select
import stat
import sys
import time

LOCK = Path("/home/ben/fifi-qwen36/gpu.lock")
COMPANION_HASH = "5c599696a6ad1a0b7ba3228b992fa57cb15ad9f55bb02036ca194ecf826abfbf"
PYTHON = "/usr/bin/python3"
MAX_SCRIPT = 1 << 20
SCRATCH = Path("/home/ben/fifi-scratch")


def verify_lock(fd, path=LOCK):
    """Verify our open-file description has the exclusive canonical flock."""
    if type(fd) is not int or fd < 3:
        raise ValueError("invalid lease descriptor")
    inherited = os.fstat(fd)
    named = os.stat(path, follow_symlinks=False)
    identity = (inherited.st_dev, inherited.st_ino)
    if not stat.S_ISREG(inherited.st_mode) or identity != (named.st_dev, named.st_ino):
        raise ValueError("lease inode mismatch")

    def owned():
        text = Path(f"/proc/self/fdinfo/{fd}").read_text(encoding="ascii")
        locks = [line for line in text.splitlines() if line.startswith("lock:")]
        return len(locks) == 1 and bool(
            re.match(r"^lock:\s+\d+:\s+FLOCK\s+ADVISORY\s+WRITE\s+", locks[0])
        )

    if not owned():
        raise ValueError("inherited descriptor does not own the lease")
    probe = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        st = os.fstat(probe)
        if identity != (st.st_dev, st.st_ino):
            raise ValueError("lease changed")
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise ValueError("lease is not independently contended")
    finally:
        os.close(probe)
    after = os.stat(path, follow_symlinks=False)
    if identity != (after.st_dev, after.st_ino) or not owned():
        raise ValueError("lease changed during verification")


def verify_program(fd, expected=COMPANION_HASH):
    """Verify immutable bytes of the inherited sealed companion memfd."""
    if type(fd) is not int or fd < 3:
        raise ValueError("invalid program descriptor")
    st = os.fstat(fd)
    if not stat.S_ISREG(st.st_mode) or not 0 < st.st_size <= MAX_SCRIPT:
        raise ValueError("invalid program image")
    required = (
        fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL
    )
    if fcntl.fcntl(fd, fcntl.F_GET_SEALS) & required != required:
        raise ValueError("program image is not fully sealed")
    digest, offset = hashlib.sha256(), 0
    while offset < st.st_size:
        block = os.pread(fd, min(65536, st.st_size - offset), offset)
        if not block:
            raise ValueError("short program image")
        digest.update(block)
        offset += len(block)
    if digest.hexdigest() != expected:
        raise ValueError("program image digest mismatch")


def wait_for_parent(fd, timeout):
    """Wait for the parent's one-byte release; EOF or expiry refuses exec."""
    if type(fd) is not int or fd < 3 or not 0 < timeout <= 60:
        raise ValueError("invalid gate")
    if not stat.S_ISFIFO(os.fstat(fd).st_mode):
        raise ValueError("gate must be a pipe")
    if fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE != os.O_RDONLY:
        raise ValueError("gate must be read-only")
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("parent release timed out")
        readable, _, _ = select.select([fd], [], [], remaining)
        if readable:
            if os.read(fd, 1) != b"\x01":
                raise ValueError("parent release missing or invalid")
            return


@dataclass(frozen=True)
class CompanionOptions:
    """Validated fields; no raw URL, model, key, FD or opaque options."""

    campaign_root: str
    output: str
    max_requests: int
    duration_seconds: float
    request_timeout: float = 120.0
    max_tokens: int = 128
    drain_timeout: float = 120.0
    drain_samples: int = 3
    drain_interval: float = 1.0
    expect_substring: str | None = None

    def arguments(self):
        scratch = SCRATCH.resolve(strict=True)
        root = Path(self.campaign_root)
        if not root.is_absolute() or root.is_symlink():
            raise ValueError("campaign root must be a task-owned directory")
        root = root.resolve(strict=True)
        root_info = root.stat()
        if (
            not root.is_relative_to(scratch)
            or not stat.S_ISDIR(root_info.st_mode)
            or root_info.st_uid != os.geteuid()
            or root_info.st_mode & 0o077
        ):
            raise ValueError("campaign root must be private task scratch")
        output = Path(self.output)
        if not output.is_absolute() or not output.parent.resolve(
            strict=True
        ).is_relative_to(root):
            raise ValueError("companion output must be within task scratch")
        if output.exists() or output.is_symlink():
            raise ValueError("companion output already exists")
        for value, low, high in (
            (self.max_requests, 1, 10000),
            (self.max_tokens, 1, 2048),
            (self.drain_samples, 1, 20),
        ):
            if type(value) is not int or not low <= value <= high:
                raise ValueError("invalid bounded companion integer")
        for value, maximum in (
            (self.duration_seconds, 7200),
            (self.request_timeout, 600),
            (self.drain_timeout, 600),
            (self.drain_interval, 10),
        ):
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not 0 < value <= maximum
            ):
                raise ValueError("invalid bounded companion duration")
        args = [
            "--output",
            str(output),
            "--max-requests",
            str(self.max_requests),
            "--duration-seconds",
            str(self.duration_seconds),
            "--request-timeout",
            str(self.request_timeout),
            "--max-tokens",
            str(self.max_tokens),
            "--drain-timeout",
            str(self.drain_timeout),
            "--drain-samples",
            str(self.drain_samples),
            "--drain-interval",
            str(self.drain_interval),
        ]
        if self.expect_substring is not None:
            value = self.expect_substring
            if (
                not isinstance(value, str)
                or not value.strip()
                or "\0" in value
                or value.startswith("-")
                or len(value.encode("utf-8")) > 256
            ):
                raise ValueError("invalid public expected marker")
            args += ["--expect-substring", value]
        return args

    def path_identity(self):
        """Snapshot directory identities; this is not an openat-based fence."""
        root = Path(self.campaign_root).resolve(strict=True)
        parent = Path(self.output).parent.resolve(strict=True)
        return tuple(
            (st.st_dev, st.st_ino, st.st_uid, st.st_mode)
            for st in (root.stat(), parent.stat())
        )


def execute(gate_fd, lease_fd, program_fd, options, *, timeout=30):
    """Check, wait, recheck and exec the sealed companion without shell use."""
    if len({gate_fd, lease_fd, program_fd}) != 3:
        raise ValueError("gate, lease and program must be distinct descriptors")
    if not isinstance(options, CompanionOptions):
        raise ValueError("typed companion options required")
    companion_args = options.arguments()
    paths_before = options.path_identity()
    verify_lock(lease_fd)
    verify_program(program_fd)
    wait_for_parent(gate_fd, timeout)
    os.close(gate_fd)
    if options.arguments() != companion_args or options.path_identity() != paths_before:
        raise ValueError("companion options changed while awaiting release")
    verify_lock(lease_fd)
    verify_program(program_fd)
    os.set_inheritable(lease_fd, True)
    os.set_inheritable(program_fd, True)
    os.execv(
        PYTHON,
        [
            PYTHON,
            "-I",
            f"/proc/self/fd/{program_fd}",
            "--inherited-lease-fd",
            str(lease_fd),
            *companion_args,
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--gate-fd", type=int, required=True)
    parser.add_argument("--lease-fd", type=int, required=True)
    parser.add_argument("--program-fd", type=int, required=True)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-requests", type=int, required=True)
    parser.add_argument("--duration-seconds", type=float, required=True)
    parser.add_argument("--request-timeout", type=float, default=120)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--drain-timeout", type=float, default=120)
    parser.add_argument("--drain-samples", type=int, default=3)
    parser.add_argument("--drain-interval", type=float, default=1)
    parser.add_argument("--expect-substring")
    # argparse ordinarily accepts duplicate options and lets the last win.
    # In particular, a second descriptor must never override the trusted one.
    seen = set()
    for token in sys.argv[1:]:
        if token.startswith("--"):
            if "=" in token or token in seen:
                parser.error("duplicate or alternate option syntax")
            seen.add(token)
    args = parser.parse_args()
    options = CompanionOptions(
        args.campaign_root,
        args.output,
        args.max_requests,
        args.duration_seconds,
        args.request_timeout,
        args.max_tokens,
        args.drain_timeout,
        args.drain_samples,
        args.drain_interval,
        args.expect_substring,
    )
    execute(args.gate_fd, args.lease_fd, args.program_fd, options)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Never include secrets or arbitrary server responses in diagnostics.
        print(f"gated companion refused: {type(error).__name__}", file=sys.stderr)
        raise SystemExit(1)
