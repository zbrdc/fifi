"""Read-only NVML process and memory sampling for exactly two pinned UUIDs.

The caller supplies an already loaded, independently trusted NVML library;
this module does not load a driver or run a CLI on import. Every required
query, including MPS, must be supported and succeed. No unsupported/permission
error is interpreted as an empty process list. This is sampled evidence, not a
fence against new contexts, a complete proof of all driver allocations, or a
replacement for service, queue, socket, lease and scope verification.

The ctypes layouts and v3 entry points match the local Linux nvml.h; ABI and
library provenance must be reviewed on the deployment host before use.
"""

from dataclasses import dataclass
import ctypes
import re
import time
import uuid as uuidlib

_SUCCESS = 0
_INSUFFICIENT_SIZE = 7
_MAX_PROCESSES = 4096
_MAX_RETRIES = 3
_MIB = 1024 * 1024
_UUID = re.compile(
    r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\Z"
)


class NvmlInconclusive(RuntimeError):
    """A required NVML fact is unavailable or inconsistent."""


class _Memory(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


class _Process(ctypes.Structure):
    _fields_ = [
        ("pid", ctypes.c_uint),
        ("used_gpu_memory", ctypes.c_ulonglong),
        ("gpu_instance_id", ctypes.c_uint),
        ("compute_instance_id", ctypes.c_uint),
    ]


@dataclass(frozen=True)
class DeviceSample:
    uuid: str
    compute_pids: tuple[int, ...]
    graphics_pids: tuple[int, ...]
    mps_pids: tuple[int, ...]
    total_bytes: int
    free_bytes: int
    used_bytes: int


@dataclass(frozen=True)
class PairSample:
    boot_id: str
    started_ns: int
    finished_ns: int
    devices: tuple[DeviceSample, DeviceSample]


def _ok(code):
    if type(code) is not int or code != _SUCCESS:
        raise NvmlInconclusive("NVML query failed or is unsupported")


def _boot_id(value):
    if not isinstance(value, str):
        raise NvmlInconclusive("missing boot identity")
    try:
        if str(uuidlib.UUID(value)) != value:
            raise NvmlInconclusive("invalid boot identity")
    except (ValueError, AttributeError) as error:
        raise NvmlInconclusive("invalid boot identity") from error
    return value


def _uuid_pair(uuids):
    if (
        not isinstance(uuids, tuple)
        or len(uuids) != 2
        or len(set(uuids)) != 2
        or any(
            not isinstance(value, str) or not _UUID.fullmatch(value) for value in uuids
        )
    ):
        raise ValueError("two distinct canonical GPU UUIDs required")
    return uuids


class NvmlReader:
    """Use an injected NVML ABI object; never implicitly load a library."""

    def __init__(self, library, uuids, *, boot_id_reader, clock_ns=time.monotonic_ns):
        self.uuids = _uuid_pair(uuids)
        self.clock_ns = clock_ns
        self.boot_id_reader = boot_id_reader
        self.library = library
        declarations = {
            "nvmlInit_v2": [],
            "nvmlShutdown": [],
            "nvmlDeviceGetHandleByUUID": [
                ctypes.c_char_p,
                ctypes.POINTER(ctypes.c_void_p),
            ],
            "nvmlDeviceGetUUID": [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_char),
                ctypes.c_uint,
            ],
            "nvmlDeviceGetMemoryInfo": [ctypes.c_void_p, ctypes.POINTER(_Memory)],
            "nvmlDeviceGetComputeRunningProcesses_v3": [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint),
                ctypes.POINTER(_Process),
            ],
            "nvmlDeviceGetGraphicsRunningProcesses_v3": [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint),
                ctypes.POINTER(_Process),
            ],
            "nvmlDeviceGetMPSComputeRunningProcesses_v3": [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint),
                ctypes.POINTER(_Process),
            ],
        }
        for name, arguments in declarations.items():
            try:
                fn = getattr(library, name)
                fn.argtypes = arguments
                fn.restype = ctypes.c_int
            except (AttributeError, TypeError) as error:
                raise NvmlInconclusive(
                    "required NVML ABI entry point unavailable"
                ) from error

    def _processes(self, fn, handle):
        count = ctypes.c_uint(0)
        code = fn(handle, ctypes.byref(count), None)
        if code == _SUCCESS and count.value == 0:
            return ()
        if code != _INSUFFICIENT_SIZE or not 1 <= count.value <= _MAX_PROCESSES:
            raise NvmlInconclusive("NVML process-list size unavailable")
        needed = count.value
        for _ in range(_MAX_RETRIES):
            capacity = min(_MAX_PROCESSES, needed + 8)
            array = (_Process * capacity)()
            size = ctypes.c_uint(capacity)
            code = fn(handle, ctypes.byref(size), array)
            if code == _SUCCESS:
                if size.value > capacity:
                    raise NvmlInconclusive("NVML returned more processes than capacity")
                pids = tuple(int(array[index].pid) for index in range(size.value))
                if any(pid <= 0 for pid in pids) or len(set(pids)) != len(pids):
                    raise NvmlInconclusive("NVML process identities invalid")
                return tuple(sorted(pids))
            if (
                code != _INSUFFICIENT_SIZE
                or size.value <= capacity
                or size.value > _MAX_PROCESSES
            ):
                raise NvmlInconclusive("NVML process list is unavailable or unstable")
            needed = size.value
        raise NvmlInconclusive("NVML process list kept growing")

    def _device(self, uuid, handle):
        actual = ctypes.create_string_buffer(96)
        _ok(self.library.nvmlDeviceGetUUID(handle, actual, len(actual)))
        try:
            if actual.value.decode("ascii") != uuid:
                raise NvmlInconclusive(
                    "NVML handle UUID does not match requested device"
                )
        except UnicodeError as error:
            raise NvmlInconclusive("NVML UUID encoding invalid") from error
        compute = self._processes(
            self.library.nvmlDeviceGetComputeRunningProcesses_v3, handle
        )
        graphics = self._processes(
            self.library.nvmlDeviceGetGraphicsRunningProcesses_v3, handle
        )
        mps = self._processes(
            self.library.nvmlDeviceGetMPSComputeRunningProcesses_v3, handle
        )
        memory = _Memory()
        _ok(self.library.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory)))
        if (
            memory.total == 0
            or memory.free > memory.total
            or memory.used > memory.total
            or memory.free + memory.used != memory.total
        ):
            raise NvmlInconclusive("NVML memory accounting inconsistent")
        return DeviceSample(
            uuid, compute, graphics, mps, memory.total, memory.free, memory.used
        )

    def sample(self):
        """Return one complete bounded snapshot or raise; never claim exclusion."""
        try:
            return self._sample()
        except NvmlInconclusive:
            raise
        except Exception:
            # Do not expose paths or library-provided text in errors.
            raise NvmlInconclusive("NVML binding or host observation failed") from None

    def _sample(self):
        before_boot = _boot_id(self.boot_id_reader())
        started = self.clock_ns()
        if type(started) is not int or started < 0:
            raise NvmlInconclusive("invalid sample clock")
        _ok(self.library.nvmlInit_v2())
        try:
            devices = []
            handles = set()
            for uuid in self.uuids:
                handle = ctypes.c_void_p()
                _ok(
                    self.library.nvmlDeviceGetHandleByUUID(
                        uuid.encode("ascii"), ctypes.byref(handle)
                    )
                )
                if handle.value is None or handle.value in handles:
                    raise NvmlInconclusive("NVML handles missing or aliased")
                handles.add(handle.value)
                devices.append(self._device(uuid, handle))
            finished = self.clock_ns()
            if type(finished) is not int or finished < started:
                raise NvmlInconclusive("invalid sample clock")
            if _boot_id(self.boot_id_reader()) != before_boot:
                raise NvmlInconclusive("boot identity changed during GPU sample")
            return PairSample(before_boot, started, finished, tuple(devices))
        finally:
            _ok(self.library.nvmlShutdown())


def _checked_sample(sample):
    if (
        not isinstance(sample, PairSample)
        or type(sample.started_ns) is not int
        or type(sample.finished_ns) is not int
        or sample.started_ns < 0
        or sample.finished_ns < sample.started_ns
        or not isinstance(sample.devices, tuple)
        or len(sample.devices) != 2
        or any(not isinstance(item, DeviceSample) for item in sample.devices)
    ):
        raise NvmlInconclusive("invalid GPU sample")
    _boot_id(sample.boot_id)
    try:
        _uuid_pair(tuple(item.uuid for item in sample.devices))
    except (ValueError, TypeError) as error:
        raise NvmlInconclusive("invalid sampled UUIDs") from error
    for item in sample.devices:
        for pids in (item.compute_pids, item.graphics_pids, item.mps_pids):
            if (
                not isinstance(pids, tuple)
                or any(type(pid) is not int or pid <= 0 for pid in pids)
                or tuple(sorted(set(pids))) != pids
            ):
                raise NvmlInconclusive("invalid sampled process IDs")
        if (
            any(
                type(value) is not int
                for value in (item.total_bytes, item.free_bytes, item.used_bytes)
            )
            or item.total_bytes <= 0
            or item.free_bytes < 0
            or item.used_bytes < 0
            or item.free_bytes + item.used_bytes != item.total_bytes
        ):
            raise NvmlInconclusive("invalid sampled memory")
    return sample


def active_pair_matches(sample, uuids, pid, expected_boot_id):
    """Check listed PIDs and boot; caller must attest PID start ticks separately."""
    expected = _uuid_pair(uuids)
    if type(pid) is not int or pid <= 0:
        raise NvmlInconclusive("invalid active-pair input")
    _checked_sample(sample)
    if sample.boot_id != _boot_id(expected_boot_id):
        raise NvmlInconclusive("GPU sample does not match expected boot")
    if tuple(device.uuid for device in sample.devices) != expected:
        raise NvmlInconclusive("sample device UUIDs differ")
    for device in sample.devices:
        if (
            device.compute_pids != (pid,)
            or any(p != pid for p in device.graphics_pids)
            or device.mps_pids
        ):
            raise NvmlInconclusive("unexpected recorded pair GPU processes")
    return True


def stopped_pair_has_headroom(
    current, baseline, expected_uuids, expected_pid, expected_boot_id
):
    """Require empty recorded process classes and sampled 1-GiB restoration floor."""
    _checked_sample(current)
    active_pair_matches(baseline, expected_uuids, expected_pid, expected_boot_id)
    if current.boot_id != baseline.boot_id:
        raise NvmlInconclusive("GPU samples were captured across different boots")
    if any(device.used_bytes <= 0 for device in baseline.devices):
        raise NvmlInconclusive("baseline pair usage is missing")
    if current.started_ns < baseline.finished_ns:
        raise NvmlInconclusive("GPU samples overlap or are out of order")
    before, after = baseline.devices, current.devices
    if (
        len(before) != 2
        or len(after) != 2
        or tuple(x.uuid for x in before) != tuple(x.uuid for x in after)
    ):
        raise NvmlInconclusive("GPU sample identities differ")
    for old, new in zip(before, after, strict=True):
        if (
            old.total_bytes != new.total_bytes
            or new.compute_pids
            or new.graphics_pids
            or new.mps_pids
        ):
            raise NvmlInconclusive("pair GPU totals or process sets changed")
        threshold = max(8192 * _MIB, old.used_bytes + 1024 * _MIB)
        if threshold > new.total_bytes or new.free_bytes < threshold:
            raise NvmlInconclusive("sampled GPU restoration headroom insufficient")
    return True
