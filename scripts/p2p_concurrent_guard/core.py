"""Fail-closed state machine for a future cooperative concurrent campaign.

This module has no host adapter or CLI and cannot start a real campaign. Its
injected adapter must pin identities, own the established lease and provide
contained scopes. An experiment is not safe merely because this code accepts it.
"""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Interval:
    start_ns: int
    end_ns: int

    def valid(self) -> bool:
        return (
            type(self.start_ns) is int
            and type(self.end_ns) is int
            and 0 <= self.start_ns < self.end_ns
        )


def intervals_overlap(pair: Interval, secondary: Interval) -> bool:
    return (
        pair.valid()
        and secondary.valid()
        and max(pair.start_ns, secondary.start_ns) < min(pair.end_ns, secondary.end_ns)
    )


class Adapter(Protocol):
    """Host operations for a future adapter, not an implementation.

    Verification methods must raise on uncertainty; release may return only
    after release is verified. Restore must handle a partially successful stop.
    The supervisor must keep this object and its lease alive for manual recovery
    when a result is inconclusive. None of these obligations is checked here.
    """

    def acquire(self) -> None: ...
    def verify_lease(self) -> None: ...
    def release(self) -> None: ...
    def snapshot_and_admit(self) -> None: ...
    def verify_secondary(self) -> None: ...
    def stop_pair(self) -> None: ...
    def verify_pair_stopped(self) -> None: ...
    def start_pair_scope(self) -> None: ...
    def start_client_scope(self) -> None: ...
    def wait_with_monitor(self) -> tuple[Interval, Interval]: ...
    def drain_client_scope(self) -> bool: ...
    def drain_pair_scope(self) -> bool: ...
    def restore_pair(self) -> None: ...
    def verify_pair_restored(self) -> bool: ...
    def verify_secondary_idle(self) -> bool: ...


@dataclass(frozen=True)
class Outcome:
    status: str
    label: str
    # None means ownership is unknown, including when release raised after
    # possibly releasing. The caller must not treat it as proven held or free.
    lease_retained: bool | None
    reason: str


def run(adapter: Adapter, *, concurrent: bool = False) -> Outcome:
    """Run only an explicitly opted-in concurrent request with injected host actions.

    A scope is marked attempted *before* start because start may partially
    succeed. Unresolved cleanup, identity, idle, restoration, or lease checks
    prevent an intentional release; actual ownership remains unknown until an
    external recovery check verifies it.
    """
    if concurrent is not True:
        return Outcome("disabled", "concurrent", False, "explicit opt-in required")
    adapter.acquire()  # Failure here is not proof a lease was acquired.
    pair_stop_attempted = pair_started = client_started = False
    failure = None
    try:
        adapter.verify_lease()
        adapter.snapshot_and_admit()
        adapter.verify_secondary()
        adapter.verify_lease()
        pair_stop_attempted = True
        adapter.stop_pair()
        adapter.verify_pair_stopped()
        adapter.verify_secondary()
        adapter.verify_lease()
        pair_started = True
        adapter.start_pair_scope()
        client_started = True
        adapter.start_client_scope()
        pair, secondary = adapter.wait_with_monitor()
        adapter.verify_secondary()
        adapter.verify_lease()
        if not intervals_overlap(pair, secondary):
            raise RuntimeError("no verified request overlap")
    except BaseException as error:
        # Never put exception text in receipts: adapters can see secrets.
        failure = type(error).__name__

    uncertain = False
    if client_started:
        try:
            uncertain |= adapter.drain_client_scope() is not True
        except BaseException:
            uncertain = True
    if pair_started:
        try:
            uncertain |= adapter.drain_pair_scope() is not True
        except BaseException:
            uncertain = True
    if not uncertain and pair_stop_attempted:
        try:
            adapter.verify_lease()
            adapter.verify_secondary()
            adapter.restore_pair()
            uncertain |= adapter.verify_pair_restored() is not True
        except BaseException:
            uncertain = True
    if not uncertain and client_started:
        try:
            adapter.verify_secondary()
            uncertain |= adapter.verify_secondary_idle() is not True
        except BaseException:
            uncertain = True
    if not uncertain:
        try:
            adapter.verify_lease()
        except BaseException:
            uncertain = True
    if uncertain:
        return Outcome(
            "inconclusive", "concurrent", None, "cleanup or restoration unverified"
        )
    try:
        adapter.release()
    except BaseException:
        return Outcome("inconclusive", "concurrent", None, "lease release unverified")
    if failure:
        return Outcome("failed", "concurrent", False, failure)
    return Outcome("passed", "concurrent", False, "verified overlap and cleanup")
