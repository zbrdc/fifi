"""Scoped command adapter for handoff.py; requires a user systemd manager."""

from __future__ import annotations

import subprocess
import time
from typing import Callable

from . import scope as scope_guard


class ScopedCommand:
    """Contain detached descendants in the existing identity-pinned scope.

    The supervisor retains its CLOEXEC lease fd; it is not inherited by the
    test, scope launcher, systemd manager, or any persistent serving service.
    A target-specific monitor must check foreign GPU/client/build activity.
    """

    def __init__(self, command: list[str], monitor: Callable[[], None], env=None):
        if not command:
            raise ValueError("empty workload")
        self.command, self.monitor = command, monitor
        self.env = env
        self.scope = scope_guard.Scope()
        self.child: subprocess.Popen | None = None

    def start(self) -> None:
        scope_guard.check_manager()
        self.child = subprocess.Popen(
            self.scope.command(self.command),
            start_new_session=True,
            close_fds=True,
            env=self.env,
        )
        self.scope.wait_started(self.child)

    def wait(self, timeout: float, lease_check: Callable[[], None]) -> int:
        assert self.child is not None
        deadline = time.monotonic() + timeout
        while self.child.poll() is None:
            if time.monotonic() >= deadline:
                raise TimeoutError("workload timed out")
            lease_check()
            self.monitor()
            time.sleep(1)
        lease_check()
        if self.scope.populated():
            raise RuntimeError("workload leader exited while descendants remain")
        return self.child.returncode

    def cleanup_and_verify_empty(self) -> bool:
        if self.child is None:
            return True
        # The reviewed scope cleanup pins unit/cgroup identity, holds until
        # empty even across setsid, and never signals an unrelated process.
        self.scope.cleanup(self.child)
        return self.child.poll() is not None and self.scope.populated() is False
