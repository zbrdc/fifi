"""Test-gated lifecycle sequencing for one fixed P2P user service.

This is not a live host backend. It intentionally accepts only explicit fake
backends/runners: no implementation yet proves full GPU-context ownership,
queue readiness, inactive-unit provenance, restoration inputs, or 2080 state.
The future coordinator must provide and independently review those verifiers
before removing this gate. No method releases the lease. If cleanup is
uncertain the owning process must keep the verified lease alive for recovery.

The state transitions follow the original pair reservation's stop/restore
shape with a stricter free-memory rule: at least max(8192 MiB, prior used +
1024 MiB) on each card and a feasible reported total. This sampled rule does
not guarantee a post-restore floor or account for fragmentation. Tests cannot
qualify systemd, GPU, HTTP or live behavior.
"""

from dataclasses import dataclass
import math
import time

from .observer import PairSnapshot

PAIR_UNIT = "fifi-qwen36-engine.service"


@dataclass(frozen=True)
class Admission:
    """Prior pair snapshot and measured VRAM used on the two exact GPUs."""

    unit: str
    snapshot: PairSnapshot
    used_mib: tuple
    total_mib: tuple
    authorized: bool

    def validate(self, expected_gpu_uuids):
        if self.unit != PAIR_UNIT or self.authorized is not True:
            raise ValueError("explicit pair admission required")
        if not isinstance(self.snapshot, PairSnapshot):
            raise ValueError("pinned pair snapshot required")
        owners = self.snapshot.gpu_owners
        if (
            not isinstance(owners, tuple)
            or len(owners) != 2
            or any(not isinstance(row, tuple) or len(row) != 2 for row in owners)
            or len({row[0] for row in owners}) != 2
            or any(row[1] != self.snapshot.process.pid for row in owners)
            or not isinstance(expected_gpu_uuids, tuple)
            or len(expected_gpu_uuids) != 2
            or set(expected_gpu_uuids) != {row[0] for row in owners}
        ):
            raise ValueError("invalid prior reported pair ownership")
        if not isinstance(self.used_mib, tuple) or len(self.used_mib) != 2:
            raise ValueError("prior pair memory measurements required")
        try:
            used = dict(self.used_mib)
        except (TypeError, ValueError) as error:
            raise ValueError("invalid prior pair memory") from error
        if (
            len(used) != 2
            or set(used) != {row[0] for row in owners}
            or any(type(value) is not int or value <= 0 for value in used.values())
        ):
            raise ValueError("prior pair memory does not match GPUs")
        if not isinstance(self.total_mib, tuple) or len(self.total_mib) != 2:
            raise ValueError("pair total-memory measurements required")
        try:
            total = dict(self.total_mib)
        except (TypeError, ValueError) as error:
            raise ValueError("invalid pair total-memory measurements") from error
        if (
            len(total) != 2
            or set(total) != set(used)
            or any(type(value) is not int or value <= 0 for value in total.values())
            or any(total[gpu] < used[gpu] + 1024 for gpu in used)
        ):
            raise ValueError("prior usage plus restoration floor exceeds GPU total")
        return used, total


@dataclass(frozen=True)
class UnitState:
    """One backend observation. Assertions must come from verified host data."""

    active_state: str
    main_pid: int
    job: str
    # These represent all pair contexts, not just `--query-compute-apps`.
    no_pair_contexts: bool
    no_pair_sockets: bool
    free_mib: tuple


class PairLifecycle:
    """Stop and restore only the admitted pair; never close or unlock a lease.

    `backend` and `runner` are test-only until real independent verifiers are
    implemented. Backend true assertions must be exact booleans and may not
    substitute sampled GPU utilization for queue or context ownership.
    Uncertain commands set phase before being issued so the caller can retain
    the lease and attempt audited recovery; SIGKILL/crash remain unguardable.
    Every lease/backend/runner method receives an absolute monotonic deadline
    and must enforce it internally. Post-call checks catch late returns but
    cannot interrupt an implementation that blocks indefinitely.
    """

    def __init__(
        self,
        admission,
        lease,
        backend,
        runner,
        *,
        expected_gpu_uuids,
        clock=time.monotonic,
        sleep=time.sleep,
    ):
        if getattr(backend, "test_backend_only", None) is not True:
            raise RuntimeError("live pair lifecycle backend is not implemented")
        if getattr(runner, "test_runner_only", None) is not True:
            raise RuntimeError("live systemd runner is not implemented")
        if not isinstance(admission, Admission):
            raise ValueError("pair admission required")
        self.used, self.total = admission.validate(expected_gpu_uuids)
        self.admission, self.lease = admission, lease
        self.backend, self.runner, self.clock, self.sleep = (
            backend,
            runner,
            clock,
            sleep,
        )
        self.phase = "ready"
        self.start_requested = False

    def _remaining(self, deadline):
        now = self.clock()
        if type(now) not in (int, float) or not math.isfinite(now):
            raise RuntimeError("invalid monotonic clock")
        remaining = deadline - now
        if not math.isfinite(remaining) or remaining <= 0:
            raise TimeoutError("pair lifecycle deadline exceeded")
        return remaining

    def _call(self, deadline, operation, *args, **kwargs):
        # Each implementation must enforce this absolute deadline internally;
        # the post-check cannot interrupt an implementation that blocks forever.
        self._remaining(deadline)
        result = operation(*args, deadline=deadline, **kwargs)
        self._remaining(deadline)
        return result

    def _guard(self, deadline):
        if self._call(deadline, self.lease.verify) is not True:
            raise RuntimeError("canonical lease not verified")
        if self._call(deadline, self.backend.secondary_unchanged) is not True:
            raise RuntimeError("secondary identity or clients changed")

    def _observe(self, deadline):
        self._guard(deadline)
        value = self._call(deadline, self.backend.observe_pair)
        if not isinstance(value, UnitState):
            raise RuntimeError("unverified pair state")
        if (
            value.active_state
            not in (
                "active",
                "inactive",
                "failed",
                "activating",
                "deactivating",
                "reloading",
            )
            or type(value.main_pid) is not int
            or value.main_pid < 0
            or (value.active_state == "active" and value.main_pid == 0)
            or not isinstance(value.job, str)
            or type(value.no_pair_contexts) is not bool
            or type(value.no_pair_sockets) is not bool
            or not isinstance(value.free_mib, tuple)
        ):
            raise RuntimeError("invalid pair state")
        return value

    def _stopped(self, view):
        if view.active_state not in ("inactive", "failed") or view.job:
            return False
        if (
            view.main_pid != 0
            or view.no_pair_contexts is not True
            or view.no_pair_sockets is not True
        ):
            raise RuntimeError("stopped pair ownership or sockets are not clear")
        try:
            free = dict(view.free_mib)
        except (TypeError, ValueError) as error:
            raise RuntimeError("invalid pair free-memory data") from error
        if (
            len(view.free_mib) != 2
            or len(free) != 2
            or set(free) != set(self.used)
            or any(type(value) is not int or value < 0 for value in free.values())
            or any(free[gpu] > self.total[gpu] for gpu in free)
        ):
            raise RuntimeError("pair free-memory data incomplete")
        if any(free[gpu] < max(8192, used + 1024) for gpu, used in self.used.items()):
            raise RuntimeError("pair lacks restoration memory margin")
        return True

    def stop(self, *, timeout=120):
        if (
            self.phase != "ready"
            or type(timeout) not in (int, float)
            or not 0 < timeout <= 600
        ):
            raise ValueError("invalid stop state or timeout")
        now = self.clock()
        if type(now) not in (int, float) or not math.isfinite(now):
            raise RuntimeError("invalid monotonic clock")
        deadline = now + timeout
        self._guard(deadline)
        if (
            self._call(
                deadline, self.backend.current_pair_matches, self.admission.snapshot
            )
            is not True
        ):
            raise RuntimeError("pinned pair identity or idle observation changed")
        if self._call(deadline, self.backend.queue_ready_and_idle) is not True:
            raise RuntimeError("pair readiness or queue unverified")
        # Repeat critical checks after readiness, immediately before stop.
        self._guard(deadline)
        if (
            self._call(
                deadline, self.backend.current_pair_matches, self.admission.snapshot
            )
            is not True
        ):
            raise RuntimeError("pair changed before stop")
        self.phase = "stop_attempted"
        try:
            result = self._call(
                deadline,
                self.runner.run,
                ["systemctl", "--user", "stop", PAIR_UNIT],
                timeout=min(45, self._remaining(deadline)),
            )
            if type(result) is not int or result != 0:
                raise RuntimeError("pair stop command did not report success")
            while True:
                view = self._observe(deadline)
                if self._stopped(view):
                    self.phase = "stopped"
                    return
                self.sleep(min(1, self._remaining(deadline)))
        except BaseException:
            self.phase = "uncertain"
            raise

    def restore(self, *, timeout=240):
        if self.phase not in (
            "ready",
            "stop_attempted",
            "stopped",
            "uncertain",
            "start_attempted",
        ):
            raise ValueError("no pair handoff to restore")
        if type(timeout) not in (int, float) or not 0 < timeout <= 600:
            raise ValueError("invalid restore timeout")
        no_stop_issued = self.phase == "ready"
        now = self.clock()
        if type(now) not in (int, float) or not math.isfinite(now):
            raise RuntimeError("invalid monotonic clock")
        deadline = now + timeout
        try:
            while True:
                view = self._observe(deadline)
                if view.active_state == "active" and not view.job:
                    original = view.main_pid == self.admission.snapshot.process.pid
                    if not original and not self.start_requested:
                        raise RuntimeError("pair process replaced before our start")
                    if (
                        not original
                        and self._call(
                            deadline,
                            self.backend.started_pair_from_attempt,
                            self.admission.snapshot,
                        )
                        is not True
                    ):
                        raise RuntimeError("replacement start provenance unverified")
                    if (
                        self._call(
                            deadline,
                            self.backend.restored_pair_matches,
                            self.admission.snapshot,
                            allow_replacement=not original,
                        )
                        is not True
                    ):
                        raise RuntimeError(
                            "restored pair identity or provenance unverified"
                        )
                    if (
                        self._call(deadline, self.backend.queue_ready_and_idle)
                        is not True
                    ):
                        raise RuntimeError(
                            "restored pair readiness or queue unverified"
                        )
                    self._guard(deadline)
                    if (
                        not original
                        and self._call(
                            deadline,
                            self.backend.started_pair_from_attempt,
                            self.admission.snapshot,
                        )
                        is not True
                    ):
                        raise RuntimeError("replacement start provenance changed")
                    if (
                        self._call(
                            deadline,
                            self.backend.restored_pair_matches,
                            self.admission.snapshot,
                            allow_replacement=not original,
                        )
                        is not True
                    ):
                        raise RuntimeError(
                            "restored pair changed after readiness check"
                        )
                    self.phase = "restored"
                    return
                if self._stopped(view):
                    if no_stop_issued:
                        raise RuntimeError("pair stopped without our handoff")
                    if (
                        self._call(
                            deadline,
                            self.backend.restore_inputs_match,
                            self.admission.snapshot,
                        )
                        is not True
                    ):
                        raise RuntimeError(
                            "inactive unit or restoration inputs changed"
                        )
                    self._guard(deadline)
                    if not self._stopped(self._observe(deadline)):
                        raise RuntimeError("pair changed before restoration start")
                    if (
                        self._call(
                            deadline,
                            self.backend.restore_inputs_match,
                            self.admission.snapshot,
                        )
                        is not True
                    ):
                        raise RuntimeError("restoration inputs changed before start")
                    # Avoid multiple starts after a partial start attempt.
                    if self.start_requested:
                        raise RuntimeError("prior start outcome is uncertain")
                    self.start_requested = True
                    self.phase = "start_attempted"
                    result = self._call(
                        deadline,
                        self.runner.run,
                        ["systemctl", "--user", "start", "--no-block", PAIR_UNIT],
                        timeout=min(10, self._remaining(deadline)),
                    )
                    if type(result) is not int or result != 0:
                        raise RuntimeError("pair start command did not report success")
                self.sleep(min(1, self._remaining(deadline)))
        except BaseException:
            self.phase = "uncertain"
            raise
