#!/usr/bin/env python3
"""Bounded M1/M1b storage-I/O harness; it does not parse GGUF or claim cold I/O."""

import argparse
import concurrent.futures
import ctypes
import errno
import fcntl
import json
import mmap
import os
import platform
import random
import resource
import stat
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

REPORT_SCHEMA = "storage-io/1"
INPUT_SCHEMA = "storage-io-input/1"
G1_REPORT_SHA256 = "b88e9bedd85fb781b3cb78c195e42bc18e4fbe029012d340554caa518cf98e34"
SHARDS = {
    "shard1": {
        "name": "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf",
        "bytes": 47_039_860_096,
        "prefix_sha256": "4eed74c765f93972a79d05d369ff43ecd507cdcbf2e08e040838bf39ecc4dcfc",
    },
    "shard2": {
        "name": "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf",
        "bytes": 28_800_138_432,
        "prefix_sha256": "1b390074f502373dc462cbe454f842588f6d174d3eb88e010b5f8819f10cd197",
    },
}
NGRAM = {"data_offset": 192, "rows": 320_001_536, "row_bytes": 90,
         "data_bytes": 28_800_138_240}
CONCURRENCY = tuple(range(1, 17))
REPETITIONS = 3
DEFAULT_SEED = 20261002
MAX_RANGE_BYTES = 8 * 1024**2
MAX_REQUESTED_BYTES = 16 * 1024**3
MAX_SECONDS = 600
# Confirmed by qualification/storage-io's compiled macOS SDK ABI probe.
DARWIN_F_RDADVISE = 44


class InputError(ValueError):
    """The supplied checked metadata or range is invalid."""


class ReadProgressError(OSError):
    """A positioned read failed after some byte requests or completions."""

    def __init__(self, error, requested_bytes, completed_bytes, syscalls):
        code = getattr(error, "errno", None) or errno.EIO
        super().__init__(code, str(error))
        self.requested_bytes = requested_bytes
        self.completed_bytes = completed_bytes
        self.syscalls = syscalls


class ShortReadError(ReadProgressError):
    """A positioned read ended before its checked range."""

    def __init__(self, error, message=None, requested_bytes=0, completed_bytes=0, syscalls=0):
        cause = error if isinstance(error, BaseException) else OSError(error, message)
        super().__init__(cause, requested_bytes, completed_bytes, syscalls)


class BudgetError(RuntimeError):
    """The bounded run would exceed its byte budget."""

    def __init__(self, message):
        super().__init__(message)
        self.requested_bytes = 0
        self.completed_bytes = 0
        self.syscalls = 0


class UnsupportedVariant(RuntimeError):
    """The operating system or file system rejected a requested variant."""


class DeadlineExpired(RuntimeError):
    """The bounded run reached its work-submission deadline."""

    def __init__(self, message, requested_bytes=0, completed_bytes=0, syscalls=0):
        super().__init__(message)
        self.requested_bytes = requested_bytes
        self.completed_bytes = completed_bytes
        self.syscalls = syscalls


class ReadBudget:
    def __init__(self, limit, parent=None):
        self.limit = limit
        self.parent = parent
        self.requested = 0
        self._lock = threading.Lock()

    def child(self):
        return ReadBudget(self.limit, parent=self)

    def reserve(self, amount):
        with self._lock:
            if amount < 0 or self.requested + amount > self.limit:
                raise BudgetError("aggregate requested-byte ceiling exceeded")
            if self.parent is not None:
                self.parent.reserve(amount)
            self.requested += amount


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise InputError(f"{name} must be an integer >= {minimum}")
    return value


def checked_bounds(offset, length, file_bytes, name="range"):
    offset = _integer(offset, f"{name}.offset")
    length = _integer(length, f"{name}.length", 1)
    file_bytes = _integer(file_bytes, "file_bytes")
    end = offset + length
    if end > file_bytes:
        raise InputError(f"{name} exceeds file bounds")
    return {"offset": offset, "length": length, "end": end}


def checked_range(offset, length, file_bytes, name="range"):
    checked = checked_bounds(offset, length, file_bytes, name)
    if checked["length"] > MAX_RANGE_BYTES:
        raise InputError(f"{name} exceeds the {MAX_RANGE_BYTES}-byte read limit")
    return checked


def page_span(offset, length, page_bytes):
    _integer(offset, "offset")
    _integer(length, "length", 1)
    page_bytes = _integer(page_bytes, "page_bytes", 1)
    pages = ((offset % page_bytes) + length + page_bytes - 1) // page_bytes
    return {"offset_mod_page": offset % page_bytes, "pages_spanned": pages,
            "page_span_bytes": pages * page_bytes}


def deterministic_sample(values, count, seed):
    count = _integer(count, "sample_count")
    ordered = sorted(values, key=lambda value: str(value))
    if count > len(ordered):
        raise InputError("sample count exceeds available checked records")
    return random.Random(seed).sample(ordered, count)


def _is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_manifest(data):
    """Validate supplied metadata. The harness never parses checkpoint bytes."""
    if not isinstance(data, dict) or data.get("schema") != INPUT_SCHEMA:
        raise InputError(f"manifest schema must be {INPUT_SCHEMA}")
    if data.get("g1_report_sha256") != G1_REPORT_SHA256:
        raise InputError("manifest must cite the checked G1 header report")
    shards = data.get("shards")
    if not isinstance(shards, dict) or set(shards) != set(SHARDS):
        raise InputError("manifest must identify exactly the two pinned IQ3_XXS shards")
    for role, pinned in SHARDS.items():
        shard = shards[role]
        if not isinstance(shard, dict) or shard.get("name") != pinned["name"]:
            raise InputError(f"{role} is not the pinned IQ3_XXS shard")
        if shard.get("bytes") != pinned["bytes"] or shard.get("prefix_sha256") != pinned["prefix_sha256"]:
            raise InputError(f"{role} differs from checked G1 size or header-prefix identity")
        if not isinstance(shard.get("path"), str) or not shard["path"]:
            raise InputError(f"{role} requires an explicit artifact path")
        full_hash = shard.get("full_sha256")
        receipt = shard.get("hash_receipt")
        if not _is_sha256(full_hash) or not isinstance(receipt, dict):
            raise InputError(f"{role} requires an approved full-file hash receipt")
        if receipt.get("method") != "one_full_pass_sha256" or receipt.get("sha256") != full_hash:
            raise InputError(f"{role} full-file hash receipt is inconsistent")
        if receipt.get("bytes") != pinned["bytes"]:
            raise InputError(f"{role} hash receipt byte count is inconsistent")
        _integer(receipt.get("mtime_ns"), f"{role}.hash_receipt.mtime_ns")
        _integer(receipt.get("inode"), f"{role}.hash_receipt.inode")

    metadata = data.get("expert_metadata")
    if not isinstance(metadata, dict) or metadata.get("source") != "inference_engine::weights::GgufHeader":
        raise InputError("expert extents must come from the existing checked GgufHeader")
    if metadata.get("shard1_prefix_sha256") != SHARDS["shard1"]["prefix_sha256"]:
        raise InputError("expert extents do not identify the checked shard-1 header")
    if metadata.get("ngram") != NGRAM:
        raise InputError("n-gram row geometry differs from the checked IQ4_NL tensor")

    extents = metadata.get("expert_tensor_extents")
    if not isinstance(extents, list) or not extents:
        raise InputError("checked expert tensor extents are required")
    extent_by_name = {}
    for extent in extents:
        if not isinstance(extent, dict):
            raise InputError("each expert tensor extent must be an object")
        name = extent.get("tensor")
        if not isinstance(name, str) or not name.startswith("blk.") or "_exps.weight" not in name:
            raise InputError("expert tensor extent must name a routed expert tensor")
        if name in extent_by_name:
            raise InputError("expert tensor extent names must be unique")
        checked = checked_bounds(extent.get("offset"), extent.get("length"), SHARDS["shard1"]["bytes"], name)
        extent_by_name[name] = checked

    groups = metadata.get("groups")
    if not isinstance(groups, list) or len(groups) < max(CONCURRENCY):
        raise InputError("at least 16 checked expert records are required for concurrency 1..16")
    ids = set()
    for group in groups:
        if not isinstance(group, dict) or not isinstance(group.get("id"), str) or not group["id"]:
            raise InputError("each expert record needs a nonempty id")
        if group["id"] in ids:
            raise InputError("expert record ids must be unique")
        ids.add(group["id"])
        ranges = group.get("ranges")
        if not isinstance(ranges, list) or len(ranges) != 3:
            raise InputError(f"expert {group['id']} must have three checked projection ranges")
        names = set()
        group_bytes = 0
        for item in ranges:
            if not isinstance(item, dict):
                raise InputError("each expert projection range must be an object")
            name = item.get("tensor")
            if name not in extent_by_name or name in names:
                raise InputError(f"expert {group['id']} must use three distinct checked tensors")
            names.add(name)
            checked = checked_range(item.get("offset"), item.get("length"), SHARDS["shard1"]["bytes"],
                                   f"expert {group['id']} projection")
            extent = extent_by_name[name]
            if checked["offset"] < extent["offset"] or checked["end"] > extent["end"]:
                raise InputError(f"expert {group['id']} range falls outside its checked tensor extent")
            group_bytes += checked["length"]
        if group_bytes > MAX_RANGE_BYTES:
            raise InputError(f"expert {group['id']} exceeds the contiguous-record range limit")
    groups.sort(key=lambda group: group["id"])
    return data


def open_checked(shard):
    path = Path(shard["path"])
    receipt = shard["hash_receipt"]
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise InputError(f"cannot stat the explicit artifact path: {exc}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise InputError("artifact path must be a regular file, not a symlink")
    if info.st_size != shard["bytes"]:
        raise InputError("artifact file size differs from checked metadata")
    if info.st_mtime_ns != receipt["mtime_ns"] or info.st_ino != receipt["inode"]:
        raise InputError("artifact stat identity differs from its approved hash receipt")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise InputError(f"cannot open artifact read-only: {exc}") from exc
    actual = os.fstat(fd)
    if (not stat.S_ISREG(actual.st_mode) or actual.st_size != shard["bytes"]
            or actual.st_ino != receipt["inode"] or actual.st_mtime_ns != receipt["mtime_ns"]):
        os.close(fd)
        raise InputError("opened artifact identity changed after validation")
    return fd


def pread_range(fd, offset, length, file_bytes, budget, read_fn=os.pread, deadline=None):
    checked = checked_range(offset, length, file_bytes)
    current = checked["offset"]
    remaining = checked["length"]
    requested = 0
    completed = 0
    calls = 0
    while remaining:
        if deadline is not None and time.monotonic() >= deadline:
            raise DeadlineExpired("measurement deadline reached before the next pread",
                                  requested, completed, calls)
        ask = remaining
        try:
            budget.reserve(ask)
        except BudgetError as exc:
            exc.requested_bytes = requested
            exc.completed_bytes = completed
            exc.syscalls = calls
            raise
        requested += ask
        calls += 1
        try:
            chunk = read_fn(fd, ask, current)
        except OSError as exc:
            raise ReadProgressError(exc, requested, completed, calls) from exc
        if not chunk:
            raise ShortReadError(errno.EIO, f"short read after {completed} of {length} bytes",
                                 requested, completed, calls)
        if len(chunk) > ask:
            error = OSError(errno.EIO, "pread returned more bytes than requested")
            raise ReadProgressError(error, requested, completed, calls)
        current += len(chunk)
        remaining -= len(chunk)
        completed += len(chunk)
    return {"requested_bytes": requested, "completed_bytes": completed, "syscalls": calls,
            "mapped_virtual_bytes": 0}


class _Radvisory(ctypes.Structure):
    _fields_ = [("ra_offset", ctypes.c_int64), ("ra_count", ctypes.c_int)]


def _radvisory_abi_supported():
    return (ctypes.sizeof(_Radvisory) == 16 and _Radvisory.ra_offset.offset == 0
            and _Radvisory.ra_count.offset == 8 and ctypes.sizeof(ctypes.c_int64) == 8
            and ctypes.sizeof(ctypes.c_int) == 4)


def rdadvise(fd, offset, length):
    if sys.platform != "darwin" or not _radvisory_abi_supported():
        raise UnsupportedVariant("F_RDADVISE or the verified Darwin radvisory ABI is unavailable")
    request = _Radvisory(offset, length)
    raw = ctypes.string_at(ctypes.addressof(request), ctypes.sizeof(request))
    try:
        fcntl.fcntl(fd, DARWIN_F_RDADVISE, raw)
    except OSError as exc:
        raise UnsupportedVariant(f"F_RDADVISE was rejected: {exc}") from exc


def mapped_range(fd, offset, length, file_bytes, page_bytes, budget, willneed=False):
    checked = checked_range(offset, length, file_bytes)
    span = page_span(offset, length, page_bytes)
    aligned_offset = offset - span["offset_mod_page"]
    mapped_length = span["offset_mod_page"] + length
    if aligned_offset + mapped_length > file_bytes:
        raise InputError("aligned mmap window exceeds file bounds")
    budget.reserve(length)
    with os.fdopen(os.dup(fd), "rb", closefd=True) as stream:
        try:
            view = mmap.mmap(stream.fileno(), length=mapped_length, access=mmap.ACCESS_READ,
                             offset=aligned_offset)
        except (OSError, ValueError) as exc:
            raise UnsupportedVariant(f"read-only mmap failed: {exc}") from exc
        try:
            advice_ns = 0
            if willneed:
                advice = getattr(mmap, "MADV_WILLNEED", None)
                if advice is None or not hasattr(view, "madvise"):
                    raise UnsupportedVariant("MADV_WILLNEED is unavailable")
                before = time.perf_counter_ns()
                try:
                    view.madvise(advice, 0, mapped_length)
                except (OSError, ValueError) as exc:
                    raise UnsupportedVariant(f"MADV_WILLNEED was rejected: {exc}") from exc
                advice_ns = time.perf_counter_ns() - before
            first = span["offset_mod_page"]
            touched = view[first:first + checked["length"]]
            if len(touched) != checked["length"]:
                raise ShortReadError(errno.EIO, "mapped range ended before its checked length")
            return {"requested_bytes": length, "completed_bytes": length, "syscalls": 0,
                    "mapped_virtual_bytes": span["page_span_bytes"],
                    "pages_spanned": span["pages_spanned"], "advice_ns": advice_ns}
        finally:
            view.close()


def _variants():
    return [
        {"access": "pread", "cache_treatment": "default_uncontrolled", "advice": "none"},
        {"access": "pread", "cache_treatment": "F_NOCACHE_requested", "advice": "none"},
        {"access": "pread", "cache_treatment": "default_uncontrolled", "advice": "F_RDADVISE"},
        {"access": "mmap", "cache_treatment": "default_uncontrolled", "advice": "none"},
        {"access": "mmap", "cache_treatment": "F_NOCACHE_requested", "advice": "none"},
        {"access": "mmap", "cache_treatment": "default_uncontrolled", "advice": "MADV_WILLNEED"},
        {"access": "mmap", "cache_treatment": "default_uncontrolled", "advice": "F_RDADVISE"},
    ]


def variant_status(variant, system=None, fd_constants=None, map_constants=None):
    system = sys.platform if system is None else system
    fd_constants = vars(fcntl) if fd_constants is None else fd_constants
    map_constants = vars(mmap) if map_constants is None else map_constants
    if system != "darwin":
        return "unsupported", "real storage qualification requires macOS"
    if variant["cache_treatment"] == "F_NOCACHE_requested" and "F_NOCACHE" not in fd_constants:
        return "unsupported", "F_NOCACHE is not exposed by this Python build"
    if variant["advice"] == "F_RDADVISE" and (DARWIN_F_RDADVISE != 44 or not _radvisory_abi_supported()):
        return "unsupported", "F_RDADVISE or the verified Darwin radvisory ABI is unavailable"
    if variant["advice"] == "MADV_WILLNEED" and "MADV_WILLNEED" not in map_constants:
        return "unsupported", "MADV_WILLNEED is not exposed by this Python build"
    return "candidate", None


def _extent_by_name(manifest):
    return {item["tensor"]: item for item in manifest["expert_metadata"]["expert_tensor_extents"]}


def sample_tasks(manifest, workload, repetition, concurrency, seed):
    groups = manifest["expert_metadata"]["groups"]
    rng = random.Random(seed + repetition * 1009 + concurrency)
    if workload == "ngram_row90":
        rows = rng.sample(range(NGRAM["rows"]), concurrency)
        return [{"id": f"row:{row}", "ranges": [{"offset": NGRAM["data_offset"] + row * NGRAM["row_bytes"],
                 "length": NGRAM["row_bytes"], "tensor": "per_layer_token_embd.weight"}],
                 "useful_bytes": NGRAM["row_bytes"]} for row in rows]

    chosen = random.Random(seed + repetition * 1009 + concurrency).sample(groups, concurrency)
    extents = _extent_by_name(manifest)
    tasks = []
    for index, group in enumerate(chosen):
        ranges = group["ranges"]
        useful = sum(item["length"] for item in ranges)
        if workload == "expert_three_projection_reads":
            task_ranges = ranges
        elif workload == "expert_contiguous_byte_control":
            projection = ranges[(seed + repetition + concurrency + index) % len(ranges)]
            extent = extents[projection["tensor"]]
            if useful > extent["length"]:
                raise InputError(f"expert {group['id']} is larger than its projection tensor extent")
            start_limit = extent["offset"] + extent["length"] - useful
            offset_rng = random.Random(seed + repetition * 65537 + concurrency * 257 + index)
            offset = offset_rng.randint(extent["offset"], start_limit)
            task_ranges = [{"tensor": projection["tensor"], "offset": offset, "length": useful}]
        else:
            raise InputError(f"unknown workload {workload}")
        tasks.append({"id": group["id"], "ranges": task_ranges, "useful_bytes": useful})
    return tasks


def _summary(values):
    ordered = sorted(values)
    def percentile(q):
        return ordered[max(0, min(len(ordered) - 1, int((len(ordered) - 1) * q + 0.999999)))]
    return {"min_ns": ordered[0], "p50_ns": percentile(.50), "p95_ns": percentile(.95),
            "p99_ns": percentile(.99), "max_ns": ordered[-1], "samples": len(ordered)}


def _resource_snapshot(path):
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "machine": platform.platform(), "cpu_count": os.cpu_count(),
              "page_bytes": os.sysconf("SC_PAGE_SIZE"), "minor_faults": usage.ru_minflt,
              "major_faults": usage.ru_majflt, "user_cpu_seconds": usage.ru_utime,
              "system_cpu_seconds": usage.ru_stime, "max_rss_platform_units": usage.ru_maxrss,
              "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None}
    try:
        volume = os.statvfs(path)
        result["volume_free_bytes"] = volume.f_bavail * volume.f_frsize
    except OSError as exc:
        result["volume_error"] = str(exc)
    for key, command in (("swap", ["sysctl", "-n", "vm.swapusage"]), ("vm_stat", ["vm_stat"])):
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=3, check=False)
            result[key] = completed.stdout.strip()[:2000] if completed.returncode == 0 else None
            if completed.returncode:
                result[key + "_error"] = completed.stderr.strip()[:300]
        except (OSError, subprocess.TimeoutExpired) as exc:
            result[key] = None
            result[key + "_error"] = f"{type(exc).__name__}: {exc}"
    return result


def _failure_status(exc):
    if isinstance(exc, UnsupportedVariant):
        return "unsupported"
    if isinstance(exc, DeadlineExpired):
        return "deadline"
    return "failed"


def _failed_sample(task, exc):
    recorded = getattr(exc, "sample_record", None)
    if isinstance(recorded, dict):
        sample = dict(recorded)
        sample.setdefault("sample_id", task["id"])
        sample.setdefault("useful_bytes", task["useful_bytes"])
        if sample.get("status") == "complete":
            sample["status"] = _failure_status(exc)
        elif sample.get("status") not in ("partial", "unsupported", "deadline", "failed"):
            sample["status"] = _failure_status(exc)
        sample.setdefault("failure", {"type": type(exc).__name__, "message": str(exc)})
        sample["future_failure"] = {"type": type(exc).__name__, "message": str(exc)}
        return sample
    requested = getattr(exc, "requested_bytes", 0)
    completed = getattr(exc, "completed_bytes", 0)
    return {"sample_id": task["id"], "status": _failure_status(exc),
            "useful_bytes": task["useful_bytes"], "requested_bytes": requested,
            "completed_bytes": completed, "syscalls": getattr(exc, "syscalls", 0),
            "mapped_virtual_bytes": 0, "pages_spanned": 0, "latency_ns": None,
            "read_amplification": requested / task["useful_bytes"],
            "cache_setup_ns": 0, "advice_ns": 0, "ranges": [],
            "failure": {"type": type(exc).__name__, "message": str(exc)}}


def _run_sample(task, variant, shard, page_bytes, budget, deadline=None):
    sample_budget = budget.child()
    sample = {"sample_id": task["id"], "status": "partial",
              "useful_bytes": task["useful_bytes"], "requested_bytes": 0,
              "completed_bytes": 0, "syscalls": 0, "mapped_virtual_bytes": 0,
              "pages_spanned": 0, "latency_ns": None, "read_amplification": 0.0,
              "cache_setup_ns": 0, "advice_ns": 0, "ranges": []}
    fd = None
    latency_start = None
    try:
        fd = open_checked(shard)
        if variant["cache_treatment"] == "F_NOCACHE_requested":
            command = getattr(fcntl, "F_NOCACHE", None)
            if command is None:
                raise UnsupportedVariant("F_NOCACHE is unavailable")
            setup_start = time.perf_counter_ns()
            try:
                fcntl.fcntl(fd, command, 1)
            except OSError as exc:
                raise UnsupportedVariant(f"F_NOCACHE was rejected: {exc}") from exc
            finally:
                sample["cache_setup_ns"] += time.perf_counter_ns() - setup_start
        latency_start = time.perf_counter_ns()
        for item in task["ranges"]:
            if deadline is not None and time.monotonic() >= deadline:
                raise DeadlineExpired("measurement deadline reached before the next range")
            checked = checked_range(item["offset"], item["length"], shard["bytes"])
            span = page_span(checked["offset"], checked["length"], page_bytes)
            requested_before = sample_budget.requested
            range_advice_ns = 0
            try:
                if variant["advice"] == "F_RDADVISE":
                    advice_start = time.perf_counter_ns()
                    try:
                        rdadvise(fd, checked["offset"], checked["length"])
                    finally:
                        range_advice_ns += time.perf_counter_ns() - advice_start
                if variant["access"] == "pread":
                    read = pread_range(fd, checked["offset"], checked["length"],
                                       shard["bytes"], sample_budget, deadline=deadline)
                else:
                    read = mapped_range(fd, checked["offset"], checked["length"],
                                        shard["bytes"], page_bytes, sample_budget,
                                        variant["advice"] == "MADV_WILLNEED")
            except Exception as exc:
                requested = sample_budget.requested - requested_before
                completed = getattr(exc, "completed_bytes", 0)
                syscalls = getattr(exc, "syscalls", 0)
                sample["completed_bytes"] += completed
                sample["syscalls"] += syscalls
                sample["pages_spanned"] += span["pages_spanned"]
                sample["advice_ns"] += range_advice_ns
                sample["ranges"].append({"tensor": item["tensor"],
                                         "offset": checked["offset"],
                                         "length": checked["length"], **span,
                                         "requested_bytes": requested,
                                         "completed_bytes": completed,
                                         "syscalls": syscalls,
                                         "status": _failure_status(exc),
                                         "failure": {"type": type(exc).__name__,
                                                     "message": str(exc)}})
                raise
            sample["requested_bytes"] += read["requested_bytes"]
            sample["completed_bytes"] += read["completed_bytes"]
            sample["syscalls"] += read["syscalls"]
            sample["mapped_virtual_bytes"] += read["mapped_virtual_bytes"]
            sample["pages_spanned"] += read.get("pages_spanned", span["pages_spanned"])
            sample["advice_ns"] += range_advice_ns + read.get("advice_ns", 0)
            sample["ranges"].append({"tensor": item["tensor"],
                                     "offset": checked["offset"],
                                     "length": checked["length"], **span,
                                     "requested_bytes": read["requested_bytes"],
                                     "completed_bytes": read["completed_bytes"],
                                     "syscalls": read["syscalls"], "status": "complete"})
        sample["status"] = "complete"
    except Exception as exc:
        sample["status"] = _failure_status(exc)
        sample["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError as exc:
                sample["cleanup_failure"] = {"type": type(exc).__name__, "message": str(exc)}
                if sample["status"] == "complete":
                    sample["status"] = "failed"
                    sample["failure"] = sample.pop("cleanup_failure")
    sample["requested_bytes"] = sample_budget.requested
    sample["read_amplification"] = sample["requested_bytes"] / task["useful_bytes"]
    if latency_start is not None:
        sample["latency_ns"] = time.perf_counter_ns() - latency_start
    return sample


def _planned_bytes(manifest, variants, seed):
    total = 0
    workloads = ("ngram_row90", "expert_three_projection_reads", "expert_contiguous_byte_control")
    for _ in variants:
        for workload in workloads:
            for concurrency in CONCURRENCY:
                for repetition in range(REPETITIONS):
                    tasks = sample_tasks(manifest, workload, repetition, concurrency, seed)
                    total += sum(item["length"] for task in tasks for item in task["ranges"])
    return total


def run_measurement(args):
    if sys.platform != "darwin":
        raise InputError("real M1/M1b measurement is permitted only on the target Mac")
    if not args.reservation_id:
        raise InputError("record the parent-approved exclusive reservation with --reservation-id")
    scratch_value = os.environ.get("TMPDIR")
    if not scratch_value:
        raise InputError("TMPDIR must point to the assigned task scratch")
    scratch = Path(scratch_value).resolve()
    output = Path(args.output).resolve()
    if not output.is_relative_to(scratch):
        raise InputError("measurement output must remain inside task scratch")
    data = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    manifest = validate_manifest(data)
    for shard in manifest["shards"].values():
        fd = open_checked(shard)
        os.close(fd)

    variants = _variants()
    supported = []
    unsupported = []
    for variant in variants:
        status, reason = variant_status(variant)
        item = {**variant, "status": status, "reason": reason}
        (supported if status == "candidate" else unsupported).append(item)
    planned = _planned_bytes(manifest, supported, args.seed)
    if planned > MAX_REQUESTED_BYTES:
        raise BudgetError(f"planned requested bytes {planned} exceed {MAX_REQUESTED_BYTES}")
    page_bytes = os.sysconf("SC_PAGE_SIZE")
    budget = ReadBudget(MAX_REQUESTED_BYTES)
    start = time.monotonic()
    deadline = start + args.max_seconds
    before_resources = _resource_snapshot(manifest["shards"]["shard1"]["path"])
    results = []
    failed_variants = []
    matrix_complete = True
    deadline_expired = False
    stop_for_deadline = False
    workloads = ("ngram_row90", "expert_three_projection_reads",
                 "expert_contiguous_byte_control")
    for variant in supported:
        variant_rows = []
        variant_failed = False
        for workload in workloads:
            if variant_failed or stop_for_deadline:
                break
            for concurrency in CONCURRENCY:
                repetitions = []
                all_latencies = []
                repetition_failed = False
                for repetition in range(REPETITIONS):
                    if time.monotonic() >= deadline:
                        deadline_expired = True
                        matrix_complete = False
                        stop_for_deadline = True
                        repetitions.append({"index": repetition, "status": "deadline",
                            "scheduled_samples": concurrency, "submitted_samples": 0,
                            "returned_samples": 0, "recorded_samples": 0,
                            "fully_completed_samples": 0, "unsent_samples": concurrency,
                            "failed_submission_samples": 0,
                            "batch_elapsed_ns": 0, "samples": []})
                        break
                    tasks = sample_tasks(manifest, workload, repetition, concurrency, args.seed)
                    batch_start = time.perf_counter_ns()
                    samples = []
                    futures = {}
                    submission_failures = []
                    submitted = 0
                    unsent = 0
                    shard = manifest["shards"]["shard2" if workload == "ngram_row90" else "shard1"]
                    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                        for task in tasks:
                            if time.monotonic() >= deadline:
                                deadline_expired = True
                                matrix_complete = False
                                stop_for_deadline = True
                                unsent = len(tasks) - submitted - len(submission_failures)
                                break
                            try:
                                future = pool.submit(_run_sample, task, variant, shard,
                                                     page_bytes, budget, deadline)
                            except Exception as exc:
                                submission_failures.append(_failed_sample(task, exc))
                                unsent = len(tasks) - submitted - len(submission_failures)
                                break
                            futures[future] = task
                            submitted += 1
                        for future in concurrent.futures.as_completed(futures):
                            task = futures[future]
                            try:
                                samples.append(future.result())
                            except Exception as exc:
                                samples.append(_failed_sample(task, exc))
                    samples.extend(submission_failures)
                    batch_ns = time.perf_counter_ns() - batch_start
                    samples.sort(key=lambda item: item["sample_id"])
                    all_latencies.extend(item["latency_ns"] for item in samples
                                         if item["latency_ns"] is not None)
                    complete_samples = sum(item["status"] == "complete" for item in samples)
                    unsupported_samples = [item for item in samples if item["status"] == "unsupported"]
                    failed_samples = [item for item in samples if item["status"] == "failed"]
                    deadline_samples = [item for item in samples if item["status"] == "deadline"]
                    if deadline_samples:
                        deadline_expired = True
                        matrix_complete = False
                        stop_for_deadline = True
                    if time.monotonic() >= deadline:
                        deadline_expired = True
                    repetition_status = ("complete" if complete_samples == len(tasks)
                                         and submitted == len(tasks) and not submission_failures
                                         else "deadline" if deadline_samples or stop_for_deadline
                                         else "partial")
                    repetitions.append({"index": repetition, "status": repetition_status,
                                        "scheduled_samples": len(tasks),
                                        "submitted_samples": submitted,
                                        "returned_samples": len(samples) - len(submission_failures),
                                        "recorded_samples": len(samples),
                                        "fully_completed_samples": complete_samples,
                                        "unsent_samples": unsent,
                                        "failed_submission_samples": len(submission_failures),
                                        "batch_elapsed_ns": batch_ns, "samples": samples})
                    if repetition_status != "complete":
                        matrix_complete = False
                        repetition_failed = True
                        if unsupported_samples:
                            unsupported.append({**variant, "status": "unsupported-at-runtime",
                                "reason": unsupported_samples[0]["failure"]["message"],
                                "sample_ids": [item["sample_id"] for item in unsupported_samples]})
                            variant_failed = True
                        if failed_samples:
                            failed_variants.append({**variant, "status": "failed-at-runtime",
                                "sample_ids": [item["sample_id"] for item in failed_samples],
                                "reasons": [item["failure"]["message"] for item in failed_samples]})
                            variant_failed = True
                        if not deadline_samples and not stop_for_deadline and not variant_failed:
                            failed_variants.append({**variant, "status": "incomplete-at-runtime",
                                "sample_ids": [item["sample_id"] for item in samples
                                               if item["status"] != "complete"],
                                "reasons": ["one or more samples were not submitted"]})
                            variant_failed = True
                        break
                totals = [sample for item in repetitions for sample in item["samples"]]
                useful = sum(item["useful_bytes"] for item in totals)
                requested = sum(item["requested_bytes"] for item in totals)
                completed = sum(item["completed_bytes"] for item in totals)
                row_complete = (len(repetitions) == REPETITIONS
                                and all(item["status"] == "complete" for item in repetitions))
                variant_rows.append({"variant": variant, "workload": workload,
                    "concurrency": concurrency, "status": "complete" if row_complete else "partial",
                    "repetitions_requested": REPETITIONS,
                    "repetitions_attempted": len(repetitions),
                    "repetitions_not_attempted": REPETITIONS - len(repetitions),
                    "repetitions_completed": sum(item["status"] == "complete" for item in repetitions),
                    "latency_distribution_ns": _summary(all_latencies) if all_latencies else None,
                    "useful_bytes": useful, "requested_bytes": requested, "completed_bytes": completed,
                    "read_amplification": requested / useful if useful else None,
                    "physical_io_bytes": None,
                    "physical_io_note": "not measured; logical requests and mapped pages are not physical I/O",
                    "cache_state": "uncontrolled; no cold or warm claim",
                    "mapped_virtual_bytes": sum(item["mapped_virtual_bytes"] for item in totals),
                    "pages_spanned": sum(item["pages_spanned"] for item in totals),
                    "syscalls": sum(item["syscalls"] for item in totals),
                    "repetitions": repetitions})
                if repetition_failed or stop_for_deadline:
                    break
            if variant_failed or stop_for_deadline:
                break
        results.extend(variant_rows)
        if stop_for_deadline:
            break

    after_resources = _resource_snapshot(manifest["shards"]["shard1"]["path"])
    finished_at = time.monotonic()
    deadline_expired = deadline_expired or finished_at >= deadline
    report = {"schema": REPORT_SCHEMA, "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "reservation_id": args.reservation_id,
        "source_commit": _git_head(Path(__file__).resolve().parents[2]), "command": sys.argv,
        "machine": {"uname": platform.uname()._asdict(), "python_version": platform.python_version(),
                    "page_bytes": page_bytes},
        "artifact_identity": {role: {"name": shard["name"], "bytes": shard["bytes"],
             "full_sha256": shard["full_sha256"], "hash_receipt": shard["hash_receipt"]}
             for role, shard in manifest["shards"].items()},
        "metadata_source": {"parser": manifest["expert_metadata"]["source"],
            "g1_report_sha256": manifest["g1_report_sha256"],
            "shard1_prefix_sha256": manifest["expert_metadata"]["shard1_prefix_sha256"]},
        "limits": {"aggregate_requested_bytes": MAX_REQUESTED_BYTES, "max_range_bytes": MAX_RANGE_BYTES,
                   "max_runtime_seconds": args.max_seconds, "repetitions": REPETITIONS,
                   "concurrency": list(CONCURRENCY)},
        "seed": args.seed, "planned_requested_bytes": planned,
        "requested_bytes_actual": budget.requested,
        "unattributed_requested_bytes": budget.requested - sum(
            sample["requested_bytes"] for row in results for repetition in row["repetitions"]
            for sample in repetition["samples"]),
        "cache_policy": "No cache purge. F_NOCACHE and advice are recorded as requested, not proof of cold storage.",
        "initial_resource_snapshot": before_resources, "results": results,
        "unsupported_variants": unsupported, "failed_variants": failed_variants,
        "final_resource_snapshot": after_resources,
        "elapsed_seconds": finished_at - start,
        "deadline_expired": deadline_expired,
        "deadline_note": "The runtime limit is not a hard wall-clock guarantee. The runner stops new submissions after it expires, drains already-owned work, and then takes the final snapshot.",
        "complete_matrix": matrix_complete,
        "status": "complete" if matrix_complete else "bounded_partial"}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, sort_keys=True, indent=2)
        handle.write("\n")
    return report


def _git_head(root):
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                                text=True, timeout=3, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _expect_error(call, fragment):
    try:
        call()
    except (InputError, ShortReadError, BudgetError) as exc:
        assert fragment in str(exc), (fragment, str(exc))
        return
    raise AssertionError(f"expected an error containing {fragment!r}")


def _synthetic_manifest():
    extents = []
    groups = []
    tensor_names = [f"blk.0.ffn_{projection}_exps.weight" for projection in ("gate", "up", "down")]
    for projection_index, tensor in enumerate(tensor_names):
        extents.append({"tensor": tensor, "offset": 1_024 + projection_index * 200_000,
                        "length": 100_000})
    for group_index in range(16):
        ranges = []
        for projection_index, (tensor, length) in enumerate(zip(
                tensor_names, (64 + group_index, 96 + group_index, 128 + group_index))):
            base = extents[projection_index]["offset"] + 10 + group_index * 256
            ranges.append({"tensor": tensor, "offset": base, "length": length})
        groups.append({"id": f"layer0-expert{group_index:02d}", "ranges": ranges})
    shards = {}
    for index, (role, pinned) in enumerate(SHARDS.items()):
        digest = ("a" if index == 0 else "b") * 64
        shards[role] = {**pinned, "path": f"/synthetic/{role}.gguf", "full_sha256": digest,
                        "hash_receipt": {"method": "one_full_pass_sha256", "sha256": digest,
                            "bytes": pinned["bytes"], "mtime_ns": 1, "inode": index + 1}}
    return {"schema": INPUT_SCHEMA, "g1_report_sha256": G1_REPORT_SHA256, "shards": shards,
        "expert_metadata": {"source": "inference_engine::weights::GgufHeader",
            "shard1_prefix_sha256": SHARDS["shard1"]["prefix_sha256"], "ngram": NGRAM,
            "expert_tensor_extents": extents, "groups": groups}}


def self_test():
    scratch = os.environ.get("TMPDIR")
    if not scratch or not Path(scratch).is_dir():
        raise InputError("TMPDIR must name the existing task scratch")
    assert REPORT_SCHEMA == "storage-io/1"
    # A 20-byte range starting ten bytes before a page boundary spans two pages.
    assert page_span(16_374, 20, 16_384) == {
        "offset_mod_page": 16_374, "pages_spanned": 2, "page_span_bytes": 32_768}
    _expect_error(lambda: checked_range(999, 2, 1_000), "file bounds")
    _expect_error(lambda: checked_range(-1, 1, 10), "offset")
    _expect_error(lambda: checked_range(0, True, 10), "length")
    values = [f"expert-{index:02d}" for index in range(20)]
    assert deterministic_sample(values, 4, DEFAULT_SEED) == deterministic_sample(values, 4, DEFAULT_SEED)
    _expect_error(lambda: deterministic_sample(values, 21, 1), "exceeds available")
    budget = ReadBudget(4)
    budget.reserve(3)
    _expect_error(lambda: budget.reserve(2), "ceiling")
    assert budget.requested == 3

    reads = []
    def partial_reader(_fd, size, offset):
        reads.append((size, offset))
        return b"abcde"[offset:offset + min(size, 2)]
    with tempfile.TemporaryDirectory(prefix="storage-io-selftest-", dir=scratch) as directory:
        path = Path(directory) / "synthetic.bin"
        payload = bytes((index * 37 + 11) % 251 for index in range(64 * 1024))
        path.write_bytes(payload)
        fd = os.open(path, os.O_RDONLY)
        try:
            result = pread_range(fd, 0, 5, len(payload), ReadBudget(20), partial_reader)
            assert result["requested_bytes"] == 9 and result["completed_bytes"] == 5
            assert result["syscalls"] == 3
            _expect_error(lambda: pread_range(fd, 0, 5, len(payload), ReadBudget(20),
                                              lambda _fd, _size, _offset: b""), "short read")
            row = {"id": "row:7", "ranges": [{"tensor": "ngram", "offset": 123, "length": 90}],
                   "useful_bytes": 90}
            single = pread_range(fd, 123, 90, len(payload), ReadBudget(100))
            assert single["completed_bytes"] == row["useful_bytes"]
            group = {"id": "g0", "ranges": [
                {"tensor": "gate", "offset": 500, "length": 64},
                {"tensor": "up", "offset": 700, "length": 96},
                {"tensor": "down", "offset": 900, "length": 128}]}
            split = [pread_range(fd, item["offset"], item["length"], len(payload), ReadBudget(300))
                     for item in group["ranges"]]
            whole_bytes = sum(item["length"] for item in group["ranges"])
            whole = pread_range(fd, 1_000, whole_bytes, len(payload), ReadBudget(whole_bytes))
            assert sum(item["completed_bytes"] for item in split) == whole["completed_bytes"] == 288
            mapping_result = mapped_range(fd, 16_374, 20, len(payload), 16_384, ReadBudget(20))
            assert mapping_result["completed_bytes"] == 20
            assert mapping_result["pages_spanned"] == 2
            optional = {}
            if sys.platform == "darwin":
                try:
                    rdadvise(fd, 123, 90)
                    optional["F_RDADVISE"] = "passed on synthetic file"
                except UnsupportedVariant as exc:
                    optional["F_RDADVISE"] = f"unsupported: {exc}"
                try:
                    advised = mapped_range(fd, 123, 90, len(payload), 16_384,
                                           ReadBudget(90), willneed=True)
                    optional["MADV_WILLNEED"] = "passed on synthetic file"
                    assert advised["completed_bytes"] == 90
                except UnsupportedVariant as exc:
                    optional["MADV_WILLNEED"] = f"unsupported: {exc}"
                nocache = getattr(fcntl, "F_NOCACHE", None)
                if nocache is None:
                    optional["F_NOCACHE"] = "unsupported: not exposed by Python"
                else:
                    try:
                        fcntl.fcntl(fd, nocache, 1)
                        optional["F_NOCACHE"] = "passed on synthetic file"
                    except OSError as exc:
                        optional["F_NOCACHE"] = f"unsupported: {exc}"
            else:
                optional = {"platform": "non-macOS; Darwin variants not executed"}
        finally:
            os.close(fd)

    manifest = _synthetic_manifest()
    validate_manifest(manifest)
    split_tasks = sample_tasks(manifest, "expert_three_projection_reads", 0, 1, DEFAULT_SEED)
    one_tasks = sample_tasks(manifest, "expert_contiguous_byte_control", 0, 1, DEFAULT_SEED)
    assert split_tasks == sample_tasks(manifest, "expert_three_projection_reads", 0, 1, DEFAULT_SEED)
    expected = sum(item["length"] for item in split_tasks[0]["ranges"])
    assert expected == sum(item["length"] for item in one_tasks[0]["ranges"])
    assert expected != 1_746_133
    changed = json.loads(json.dumps(manifest))
    changed["expert_metadata"]["groups"][0]["ranges"][0]["offset"] = SHARDS["shard1"]["bytes"]
    _expect_error(lambda: validate_manifest(changed), "file bounds")
    assert variant_status(_variants()[0], system="linux", fd_constants={}, map_constants={})[0] == "unsupported"
    assert variant_status(_variants()[1], system="darwin", fd_constants={}, map_constants={})[0] == "unsupported"
    assert variant_status(_variants()[5], system="darwin", fd_constants={}, map_constants={})[0] == "unsupported"
    assert _radvisory_abi_supported()
    assert DARWIN_F_RDADVISE == 44
    assert MAX_REQUESTED_BYTES == 16 * 1024**3 and MAX_RANGE_BYTES == 8 * 1024**2
    return optional


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--manifest")
    parser.add_argument("--output")
    parser.add_argument("--reservation-id")
    parser.add_argument("--repetitions", type=int, default=REPETITIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-seconds", type=int, default=MAX_SECONDS)
    args = parser.parse_args(argv)
    if args.self_test:
        if args.manifest or args.output or args.reservation_id:
            parser.error("--self-test cannot be combined with measurement inputs")
        return args
    if not args.manifest or not args.output:
        parser.error("--manifest and --output are required unless --self-test is used")
    if not args.reservation_id:
        parser.error("--reservation-id is required after the parent grants an exclusive measurement reservation")
    if args.repetitions != REPETITIONS:
        parser.error(f"--repetitions must equal the fixed value {REPETITIONS}")
    if not 1 <= args.max_seconds <= MAX_SECONDS:
        parser.error(f"--max-seconds must be between 1 and {MAX_SECONDS}")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    return args


def main(argv=None):
    try:
        args = parse_args(sys.argv[1:] if argv is None else argv)
        if args.self_test:
            optional = self_test()
            print("storage-io self-test: PASS")
            print(json.dumps({"synthetic_optional_api_checks": optional}, sort_keys=True))
            return 0
        report = run_measurement(args)
        print(json.dumps({"status": report["status"], "rows": len(report["results"]),
                          "requested_bytes_actual": report["requested_bytes_actual"],
                          "output": str(Path(args.output).resolve())}, sort_keys=True))
        return 0 if report["complete_matrix"] else 3
    except (InputError, BudgetError, OSError, json.JSONDecodeError) as exc:
        print(f"storage-io: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
