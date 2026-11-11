#!/usr/bin/env python3
"""Sample a Linux process and NVIDIA GPUs while a separately controlled benchmark runs.

This observer does not start a workload, reserve devices, or verify its build.
Run it in an admitted benchmark window and compare benchmark results with the
observer off and on before using the timings as performance evidence.

Example:
  python3 scripts/bench_host_gpu_resources.py --pid 1234 --gpu GPU-... \
      --gpu GPU-... --interval 2 --output /path/to/scratch/resources.jsonl

CPU rates are cores consumed, not a percentage of all logical CPUs. GPU PCIe
RX/TX values are NVML's 20 ms samples in KB/s and are not continuous byte
counts. GPU utilization is a sampled busy percentage, not compute efficiency.
See https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
import threading
import time


@dataclass(frozen=True)
class CpuTicks:
    start: int
    user: int
    system: int
    name: str = ''


@dataclass(frozen=True)
class CpuSnapshot:
    process: CpuTicks
    threads: dict[int, CpuTicks]
    system_total: int
    system_idle: int
    monotonic_ns: int


def parse_proc_stat(text: str) -> CpuTicks:
    """Parse /proc/PID/stat without splitting a parenthesized command name."""
    start, end = text.find('('), text.rfind(')')
    if start < 1 or end <= start or not text[:start].strip().isdigit():
        raise ValueError('malformed process stat')
    fields = text[end + 1:].split()
    if len(fields) < 20:
        raise ValueError('incomplete process stat')
    user, system, birth = int(fields[11]), int(fields[12]), int(fields[19])
    if min(user, system, birth) < 0:
        raise ValueError('negative process stat counter')
    return CpuTicks(start=birth, user=user, system=system, name=text[start + 1:end])


def parse_system_stat(text: str) -> tuple[int, int]:
    fields = text.splitlines()[0].split()
    if len(fields) < 6 or fields[0] != 'cpu':
        raise ValueError('missing aggregate CPU stat')
    values = [int(value) for value in fields[1:9]]
    if any(value < 0 for value in values):
        raise ValueError('negative aggregate CPU counter')
    # guest and guest_nice are already included in user and nice.
    total = sum(values)
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return total, idle


def read_cpu_snapshot(proc: Path, pid: int, monotonic_ns: int) -> CpuSnapshot:
    process = parse_proc_stat((proc / str(pid) / 'stat').read_text())
    threads: dict[int, CpuTicks] = {}
    for thread in (proc / str(pid) / 'task').iterdir():
        if not thread.name.isdigit():
            continue
        try:
            threads[int(thread.name)] = parse_proc_stat((thread / 'stat').read_text())
        except (FileNotFoundError, ProcessLookupError):
            # Threads may exit between listing and reading their counters.
            continue
    total, idle = parse_system_stat((proc / 'stat').read_text())
    return CpuSnapshot(process, threads, total, idle, monotonic_ns)


def cpu_delta(previous: CpuSnapshot, current: CpuSnapshot, ticks_per_second: int) -> dict:
    if previous.process.start != current.process.start:
        raise ValueError('process identity changed')
    seconds = (current.monotonic_ns - previous.monotonic_ns) / 1e9
    user = current.process.user - previous.process.user
    system = current.process.system - previous.process.system
    if seconds <= 0 or ticks_per_second <= 0 or min(user, system) < 0:
        raise ValueError('invalid CPU interval or counters')
    scale = ticks_per_second * seconds
    candidates = []
    for tid, now in current.threads.items():
        before = previous.threads.get(tid)
        if before is None or before.start != now.start:
            continue
        ticks = now.user + now.system - before.user - before.system
        if ticks >= 0:
            candidates.append((ticks / scale, tid, now.name))
    busiest = max(candidates, default=None)
    total = current.system_total - previous.system_total
    idle = current.system_idle - previous.system_idle
    busy = (total - idle) / total if total > 0 and 0 <= idle <= total else None
    return {'interval_seconds': seconds, 'process_user_cores': user / scale,
            'process_system_cores': system / scale, 'process_total_cores': (user + system) / scale,
            'busiest_thread_cores': busiest[0] if busiest else None,
            'busiest_thread_id': busiest[1] if busiest else None,
            'busiest_thread_name': busiest[2] if busiest else None,
            'matched_threads': len(candidates), 'system_busy_fraction': busy}


class NvmlUtilization(ctypes.Structure):
    _fields_ = [('gpu', ctypes.c_uint), ('memory', ctypes.c_uint)]


class NvmlMemory(ctypes.Structure):
    _fields_ = [('total', ctypes.c_ulonglong), ('free', ctypes.c_ulonglong),
                ('used', ctypes.c_ulonglong)]


class Nvml:
    def __init__(self, uuids: list[str], library: str = 'libnvidia-ml.so.1'):
        self.lib = ctypes.CDLL(library)
        void_p, uint_p = ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)
        signatures = {
            'nvmlInit_v2': [], 'nvmlShutdown': [],
            'nvmlDeviceGetHandleByUUID': [ctypes.c_char_p, ctypes.POINTER(void_p)],
            'nvmlDeviceGetUtilizationRates': [void_p, ctypes.POINTER(NvmlUtilization)],
            'nvmlDeviceGetMemoryInfo': [void_p, ctypes.POINTER(NvmlMemory)],
            'nvmlDeviceGetPcieThroughput': [void_p, ctypes.c_int, uint_p],
            'nvmlDeviceGetCurrPcieLinkGeneration': [void_p, uint_p],
            'nvmlDeviceGetCurrPcieLinkWidth': [void_p, uint_p],
        }
        for name, args in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = args, ctypes.c_int
        status = self.lib.nvmlInit_v2()
        if status:
            raise RuntimeError(f'NVML initialization failed: {status}')
        self.handles = {}
        try:
            for uuid in uuids:
                handle = void_p()
                status = self.lib.nvmlDeviceGetHandleByUUID(uuid.encode(), ctypes.byref(handle))
                if status:
                    raise RuntimeError(f'NVML could not resolve {uuid}: {status}')
                self.handles[uuid] = handle
        except Exception:
            self.close()
            raise

    def close(self):
        self.lib.nvmlShutdown()

    def sample(self, uuid: str) -> dict:
        handle, errors = self.handles[uuid], {}
        result = {'uuid': uuid, 'started_monotonic_ns': time.monotonic_ns()}
        utilization = NvmlUtilization()
        status = self.lib.nvmlDeviceGetUtilizationRates(handle, ctypes.byref(utilization))
        result.update({'gpu_busy_percent': utilization.gpu if not status else None,
                       'gpu_memory_busy_percent': utilization.memory if not status else None})
        if status:
            errors['utilization'] = status
        memory = NvmlMemory()
        status = self.lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory))
        result.update({'memory_total_bytes': memory.total if not status else None,
                       'memory_free_bytes': memory.free if not status else None,
                       'memory_used_bytes': memory.used if not status else None})
        if status:
            errors['memory'] = status
        for key, function, extra in (
            ('pcie_tx_kb_per_s', self.lib.nvmlDeviceGetPcieThroughput, (0,)),
            ('pcie_rx_kb_per_s', self.lib.nvmlDeviceGetPcieThroughput, (1,)),
            ('pcie_generation', self.lib.nvmlDeviceGetCurrPcieLinkGeneration, ()),
            ('pcie_width', self.lib.nvmlDeviceGetCurrPcieLinkWidth, ()),
        ):
            value = ctypes.c_uint()
            query_started = time.monotonic_ns()
            status = function(handle, *extra, ctypes.byref(value))
            query_finished = time.monotonic_ns()
            result[key] = value.value if not status else None
            if key in ('pcie_tx_kb_per_s', 'pcie_rx_kb_per_s'):
                result[key + '_started_monotonic_ns'] = query_started
                result[key + '_finished_monotonic_ns'] = query_finished
            if status:
                errors[key] = status
        result.update({'finished_monotonic_ns': time.monotonic_ns(), 'errors': errors})
        return result


def write_event(stream, event: dict) -> None:
    stream.write(json.dumps(event, allow_nan=False) + '\n')
    stream.flush()


def run(pid: int, uuids: list[str], interval: float, samples: int, output: Path) -> int:
    if (pid <= 0 or not math.isfinite(interval) or interval < 0.25 or samples < 0
            or not uuids or len(set(uuids)) != len(uuids)
            or any(not re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', uuid)
                   for uuid in uuids)):
        raise ValueError('invalid process, GPU list, interval, or sample count')
    ticks = os.sysconf('SC_CLK_TCK')
    initial = read_cpu_snapshot(Path('/proc'), pid, time.monotonic_ns())
    try:
        affinity_count = len(os.sched_getaffinity(pid))
    except OSError:
        affinity_count = None
    nvml = Nvml(uuids)
    stop = threading.Event()
    previous_handlers = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        with output.open('x') as stream:
            write_event(stream, {'type': 'metadata', 'pid': pid, 'process_start_ticks': initial.process.start,
                'gpu_uuids': uuids, 'interval_seconds': interval, 'clock_ticks_per_second': ticks,
                'process_affinity_cpu_count': affinity_count, 'host_logical_cpus': os.cpu_count(),
                'pcie_note': 'Each value is an independent 20 ms NVML sample in KB/s.',
                'cpu_note': 'CPU rates are cores and exclude child processes; busiest thread includes only threads '
                            'present in consecutive samples.',
                'limitations': 'No workload admission, device isolation or build identity is established.'})
            previous, count, deadline = initial, 0, time.monotonic()
            while not stop.is_set() and (samples == 0 or count < samples):
                wall_start, self_cpu_start = time.monotonic_ns(), time.process_time_ns()
                try:
                    current = read_cpu_snapshot(Path('/proc'), pid, wall_start)
                    if current.process.start != initial.process.start:
                        raise ValueError('process identity changed')
                    cpu = cpu_delta(previous, current, ticks) if count else None
                except (FileNotFoundError, ProcessLookupError):
                    write_event(stream, {'type': 'end', 'reason': 'process_disappeared', 'samples': count})
                    return 1
                except ValueError as error:
                    write_event(stream, {'type': 'end', 'reason': 'invalid_cpu_sample',
                                         'error': str(error), 'samples': count})
                    return 1
                gpu = [nvml.sample(uuid) for uuid in uuids]
                wall_end, self_cpu_end = time.monotonic_ns(), time.process_time_ns()
                write_event(stream, {'type': 'sample', 'utc_ns': time.time_ns(),
                    'started_monotonic_ns': wall_start, 'finished_monotonic_ns': wall_end,
                    'observer_wall_ms': (wall_end - wall_start) / 1e6,
                    'observer_cpu_ms': (self_cpu_end - self_cpu_start) / 1e6,
                    'cpu': cpu, 'gpus': gpu})
                previous, count = current, count + 1
                if samples and count >= samples:
                    break
                deadline += interval
                stop.wait(max(0.0, deadline - time.monotonic()))
            write_event(stream, {'type': 'end', 'reason': 'signal' if stop.is_set() else 'sample_limit',
                                 'samples': count})
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        nvml.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid', required=True, type=int)
    parser.add_argument('--gpu', required=True, action='append', help='GPU UUID; repeat for each device')
    parser.add_argument('--interval', type=float, default=2.0, help='sampling interval in seconds, at least 0.25')
    parser.add_argument('--samples', type=int, default=0, help='stop after this many samples; 0 means until signalled')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        return run(args.pid, args.gpu, args.interval, args.samples, args.output)
    except (OSError, ValueError, RuntimeError, AttributeError) as error:
        print(f'resource observer failed: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
