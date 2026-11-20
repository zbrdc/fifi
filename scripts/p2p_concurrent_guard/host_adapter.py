"""Fail-closed host-adapter sequencing for joint P2P/2080 experiments.

This adapter has NO real service backend or CLI. Its backend is a test double
until the historical pair runner is rebound and reviewed. A fake's assertions
cannot attest real systemd scopes, sockets, GPUs, receipts, or server drain.
"""

from dataclasses import dataclass
import os
from pathlib import Path
import re

from .host_attestation import sealed_companion, sealed_gate

GPU_2080 = "GPU-699004f8-e2f0-747e-5c02-30141e9d4b7c"
COMPANION = Path(
    "/home/ben/fifi-scratch/07dda740c23e/2080-concurrent-companion/companion.py"
)
GATE = Path(__file__).with_name("gated_companion.py")
HASH = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class SecondaryIdentity:
    pid: int
    start_ticks: int
    cgroup: str
    executable_sha256: str
    unit_sha256: str
    environment_sha256: str
    config_sha256: str
    gpu_uuid: str
    gpu_owners: tuple

    def valid(self):
        return (
            type(self.pid) is int
            and self.pid > 0
            and type(self.start_ticks) is int
            and self.start_ticks > 0
            and isinstance(self.cgroup, str)
            and self.cgroup.startswith("/")
            and all(
                isinstance(value, str) and HASH.fullmatch(value)
                for value in (
                    self.executable_sha256,
                    self.unit_sha256,
                    self.environment_sha256,
                    self.config_sha256,
                )
            )
            and self.gpu_uuid == GPU_2080
            and self.gpu_owners == (self.pid,)
        )


class HostAdapter:
    """Composition against an injected backend; no backend is shipped.

    The backend must implement all operations invoked here and prove process,
    scope, socket, receipt and service facts independently. It must retain
    unverified scopes and the lease on uncertain cleanup. Tests exercise the
    ordering with fakes only; this class is not a live admission authority.
    """

    def __init__(self, backend, lease):
        if getattr(backend, "test_backend_only", None) is not True:
            raise RuntimeError("real host backend is not implemented or admitted")
        self.backend, self.lease = backend, lease
        self.secondary = None
        self.client = None
        self.pair = None
        self.client_spawn_attempted = False
        self.pair_spawn_attempted = False
        self.program_fd = self.gate_fd = self.write_fd = None

    def acquire(self):
        self.lease.acquire()

    def verify_lease(self):
        if self.lease.verify() is not True:
            raise RuntimeError("lease attestation failed")

    def snapshot_and_admit(self):
        if self.backend.snapshot_pair_and_admit() is not True:
            raise RuntimeError("pair admission failed")
        identity = self.backend.capture_secondary()
        if not isinstance(identity, SecondaryIdentity) or not identity.valid():
            raise RuntimeError("secondary identity incomplete")
        self.secondary = identity
        self.verify_secondary()

    def verify_secondary(self):
        if self.secondary is None or self.backend.capture_secondary() != self.secondary:
            raise RuntimeError("secondary service identity changed")
        if self.backend.check_secondary_clients(self.client) is not True:
            raise RuntimeError("secondary clients not attested")

    def stop_pair(self):
        self.backend.stop_pair()

    def verify_pair_stopped(self):
        if self.backend.verify_pair_stopped() is not True:
            raise RuntimeError("pair stop not verified")

    def start_pair_scope(self):
        self.pair_spawn_attempted = True
        self.pair = self.backend.start_pair_scope(self.lease.fd)
        if self.backend.verify_pair_scope(self.pair) is not True:
            raise RuntimeError("pair scope not attested")

    def start_client_scope(self):
        self.verify_lease()
        self.verify_secondary()
        self.program_fd = sealed_companion(COMPANION)
        self.gate_fd = sealed_gate(GATE)
        read_fd, self.write_fd = os.pipe2(os.O_CLOEXEC)
        try:
            self.client_spawn_attempted = True
            self.client = self.backend.start_gated_companion(
                read_fd, self.lease.fd, self.program_fd, self.gate_fd
            )
        finally:
            os.close(read_fd)
        # The backend must attest exact child PID/start time/cgroup, held
        # descriptors, the sealed gate and waiting state BEFORE release.
        if (
            self.backend.verify_gate_waiting(
                self.client, self.lease.fd, self.program_fd, self.gate_fd
            )
            is not True
        ):
            raise RuntimeError("gated client not attested")
        self.verify_secondary()
        self.verify_lease()
        if os.write(self.write_fd, b"\x01") != 1:
            raise RuntimeError("gated client release failed")
        os.close(self.write_fd)
        self.write_fd = None
        if self.backend.verify_companion_exec(self.client) is not True:
            raise RuntimeError("companion exec not attested")

    def wait_with_monitor(self):
        # Backend must verify producer/hash/boot/clock bindings and completed
        # whole-request receipts; no GPU/preload phase inference is allowed.
        return self.backend.wait_with_monitor(
            self.pair, self.client, self.verify_secondary, self.verify_lease
        )

    def drain_client_scope(self):
        if self.write_fd is not None:
            os.close(self.write_fd)
            self.write_fd = None
        if self.client is None:
            # A launch can partially succeed before returning its handle. Until
            # independently recoverable, do not pretend an unknown scope is empty.
            return not self.client_spawn_attempted
        # Forced termination or unverified server-side drain must return False.
        return self.backend.drain_client_scope(self.client) is True

    def drain_pair_scope(self):
        if self.pair is None:
            return not self.pair_spawn_attempted
        return self.backend.drain_pair_scope(self.pair) is True

    def restore_pair(self):
        self.backend.restore_pair()

    def verify_pair_restored(self):
        return self.backend.verify_pair_restored() is True

    def verify_secondary_idle(self):
        return self.backend.verify_secondary_idle() is True

    def release(self):
        for name in ("gate_fd", "program_fd"):
            fd = getattr(self, name)
            if fd is not None:
                os.close(fd)
                setattr(self, name, None)
        self.lease.close()
        # An inherited duplicate can keep the flock alive after our close.
        # A competing new owner is likewise not distinguishable from that
        # case here: report uncertainty rather than claim a verified release.
        if self.lease.verify_released() is not True:
            raise RuntimeError("lease release is not independently verified")
