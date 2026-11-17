"""Cooperative GPU handoff state machine.

All service-specific checks/actions and scoped workload execution are supplied by
reviewed adapters. This module does not install or control any service by itself.
"""

from __future__ import annotations

from dataclasses import dataclass
import fcntl
import os
from pathlib import Path
import signal
import stat
from typing import Any, Callable, Protocol

SHARED_LOCK = Path("/home/ben/fifi-qwen36/gpu.lock")
CANONICAL_LOCK = Path("/home/ben/fifi-scratch/a81ebdf26a2a/qwen36-local/gpu.lock")


class Busy(RuntimeError):
    pass


class LeaseChanged(RuntimeError):
    pass


class Interrupted(BaseException):
    pass


class Service(Protocol):
    # These four inspection methods must be side-effect-free. A live adapter
    # must separately pin and verify unit, process, binary, config and clients.
    def snapshot(self) -> Any: ...
    def was_active(self, snapshot: Any) -> bool: ...
    def admit(self, snapshot: Any) -> None: ...
    def still_matches(self, snapshot: Any) -> bool: ...
    def stop(self, snapshot: Any) -> None: ...
    def verify_stopped(self, snapshot: Any) -> None: ...
    def restore(self, snapshot: Any) -> None: ...
    def verify_restored(self, snapshot: Any) -> bool: ...


class Workload(Protocol):
    def start(self) -> None: ...
    def wait(self, timeout: float, lease_check: Callable[[], None]) -> int: ...
    def cleanup_and_verify_empty(self) -> bool: ...


class Lease:
    """Exclusive nonblocking flock, pinned to an existing canonical inode."""

    def __init__(self, path: Path):
        self.logical_path = Path(os.path.abspath(path))
        self.path = path.resolve(strict=True)
        self.fd: int | None = None
        self.identity: tuple[int, int] | None = None

    def acquire(self) -> None:
        if self.fd is not None:
            raise Busy("lease instance already acquired")
        fd = os.open(self.path, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            opened = os.fstat(fd)
            if not stat.S_ISREG(opened.st_mode):
                raise ValueError("lock is not a regular file")
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise Busy("shared lease is held") from error
            current = os.stat(self.path, follow_symlinks=False)
            logical = os.stat(self.logical_path)
            identity = (opened.st_dev, opened.st_ino)
            if identity != (current.st_dev, current.st_ino) or identity != (
                logical.st_dev,
                logical.st_ino,
            ):
                raise LeaseChanged("lock path changed during acquisition")
            self.fd, self.identity = fd, identity
        except BaseException:
            os.close(fd)
            raise

    def verify(self) -> None:
        if self.fd is None or self.identity is None:
            raise LeaseChanged("lease not held")
        opened = os.fstat(self.fd)
        current = os.stat(self.path, follow_symlinks=False)
        logical = os.stat(self.logical_path)
        if (
            (opened.st_dev, opened.st_ino) != self.identity
            or (current.st_dev, current.st_ino) != self.identity
            or (logical.st_dev, logical.st_ino) != self.identity
        ):
            raise LeaseChanged("shared lock inode changed")
        fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def release(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


class SharedLease(Lease):
    """Pin the actual established host lock; caller cannot supply a root."""

    def __init__(self):
        if SHARED_LOCK.resolve(strict=True) != CANONICAL_LOCK.resolve(strict=True):
            raise LeaseChanged("shared lock alias differs from canonical host lock")
        super().__init__(SHARED_LOCK)

    def verify(self) -> None:
        super().verify()
        canonical = os.stat(CANONICAL_LOCK)
        if (canonical.st_dev, canonical.st_ino) != self.identity:
            raise LeaseChanged("canonical host lock inode changed")

    def acquire(self) -> None:
        super().acquire()
        try:
            self.verify()
        except BaseException:
            self.release()
            raise


@dataclass
class Outcome:
    result: str
    reason: str
    service_was_active: bool | None
    # None means this supervisor never took a service/workload action.
    restore_confirmed: bool | None
    workload_drained: bool
    # Non-None means retain this live object and its lock for manual recovery.
    retained_lease: Lease | None


def handoff(
    lease: Lease, service: Service, workload: Workload, timeout: float
) -> Outcome:
    """Run an injected contained workload. No service actions occur without lease.

    A failed/unknown cleanup or restoration returns a *held* lease. Callers must
    retain the returned Outcome and resolve recovery explicitly; never call
    Lease.release() merely because a handoff failed. This is a cooperative API,
    not a barrier to a separate direct systemctl caller or SIGKILL/power loss.
    """
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    lease.acquire()
    snapshot = None
    active: bool | None = None
    stop_requested = False
    workload_started = False
    drained = True
    restored: bool | None = None
    work_rc: int | None = None
    issue: BaseException | None = None
    previous: dict[int, Any] = {}

    def interrupted(signum: int, _frame: Any) -> None:
        raise Interrupted(f"interrupted by signal {signum}")

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.getsignal(sig)
            signal.signal(sig, interrupted)
        lease.verify()
        snapshot = service.snapshot()
        active = service.was_active(snapshot)
        if type(active) is not bool:
            raise ValueError("service active state must be an explicit bool")
        service.admit(snapshot)
        lease.verify()
        if not service.still_matches(snapshot):
            raise RuntimeError("service changed before handoff")
        if active:
            lease.verify()
            # Stop may be accepted even if the client times out; always restore.
            stop_requested = True
            service.stop(snapshot)
            service.verify_stopped(snapshot)
        lease.verify()
        workload_started = True  # start may partially succeed before raising
        workload.start()
        work_rc = workload.wait(timeout, lease.verify)
    except BaseException as error:
        issue = error
    finally:
        # Defer repeated termination signals until cleanup and restoration end.
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        if workload_started:
            try:
                drained = workload.cleanup_and_verify_empty() is True
            except BaseException as error:
                drained = False
                issue = issue or error
        if stop_requested and drained:
            try:
                lease.verify()
                service.restore(snapshot)
                lease.verify()
                restored = service.verify_restored(snapshot) is True
            except BaseException as error:
                issue = issue or error
                restored = False
        elif not stop_requested and workload_started:
            # Never start a service that was originally inactive; verify it
            # stayed inactive after the work, under the same pinned lease.
            try:
                lease.verify()
                restored = service.verify_restored(snapshot) is True
            except BaseException as error:
                issue = issue or error
                restored = False
        # Otherwise no service/workload action was taken: restore status is
        # deliberately None, not an assertion that an external owner is stable.
        for sig, handler in previous.items():
            signal.signal(sig, handler)

    if not drained or restored is False:
        detail = (
            "workload cleanup unconfirmed"
            if not drained
            else "service restoration unconfirmed"
        )
        return Outcome(
            "inconclusive", f"{detail}; {issue!r}", active, restored, drained, lease
        )
    lease.release()
    if issue is not None:
        return Outcome("failed", repr(issue), active, restored, drained, None)
    if work_rc != 0:
        return Outcome(
            "failed", f"workload exit {work_rc}", active, restored, drained, None
        )
    return Outcome(
        "passed",
        "workload complete and original state confirmed",
        active,
        restored,
        drained,
        None,
    )


def handoff_shared(service: Service, workload: Workload, timeout: float) -> Outcome:
    """Host entrypoint: always constructs the fixed-path shared lease.

    The injectable ``handoff`` is for tests only; live callers must use this
    no-lock-argument entrypoint.
    """
    return handoff(SharedLease(), service, workload, timeout)
