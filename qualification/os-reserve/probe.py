"""Phase 0 OS_RESERVE probe: allocate MLX arrays in 512 MiB steps and watch for swap start.

Residency uses the process physical footprint (macOS `phys_footprint`), because
MLX Metal buffers do not enter RSS. RSS stays in each row as a recorded column only.
"""

import gc
import json
import os
import platform
import re
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone

BLOCK_BYTES = 512 * 1024 * 1024
BLOCK_ELEMS = BLOCK_BYTES // 4
MAX_STEPS = 128
SWAP_THRESHOLD_BYTES = 8 * 1024 * 1024
SWAPOUTS_THRESHOLD_PAGES = 512
STEP_SETTLE_SECONDS = 2.0
RESIDENCY_FRACTION = 0.5
CUMULATIVE_FRACTION = 0.9
CALIBRATION_MIN_BYTES = 256 * 1024 * 1024
REPORT_SCHEMA = "p0a05-os-reserve/4"
CALIBRATION_SCHEMA = "p0a05-os-reserve-calibration/1"
RUSAGE_INFO_V2 = 2
RUSAGE_WORDS = 64
RUSAGE_V2_RESIDENT_INDEX = 8
RUSAGE_V2_FOOTPRINT_INDEX = 9
STABILITY_INTERVAL_SECONDS = 5.0
STABILITY_SAMPLES = 4
STABILITY_MAX_POLLS = 24

VM_KEY_CANDIDATES = {
    "pages_free": ("Pages free",),
    "pages_active": ("Pages active",),
    "pages_inactive": ("Pages inactive",),
    "pages_wired": ("Pages wired down",),
    "pages_compressed": ("Pages stored in compressor", "Pages occupied by compressor"),
    "swapins": ("Swapins", "Pages swapins"),
    "swapouts": ("Swapouts", "Pages swapouts"),
}

METAL_INFO_CANDIDATES = {
    "memory_size_bytes": ("memory_size", "memorySize", "device_memory_size"),
    "max_recommended_working_set_bytes": (
        "max_recommended_working_set_size",
        "max_working_set_size",
        "recommendedMaxWorkingSetSize",
    ),
    "max_buffer_length_bytes": ("max_buffer_length", "maxBufferLength"),
}

SWAP_FACTORS = {"": 1, "K": 1024, "M": 1024 ** 2, "G": 1024 ** 3, "T": 1024 ** 4}


def sysctl(name):
    result = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def capture_swap():
    raw = sysctl("vm.swapusage")
    fields = {"raw": raw}
    for name in ("total", "used", "free"):
        match = re.search(name + r"\s*=\s*([0-9.]+)([KMGT]?)", raw)
        if match is None:
            fields[name + "_bytes"] = None
        else:
            factor = SWAP_FACTORS[match.group(2)]
            fields[name + "_bytes"] = int(round(float(match.group(1)) * factor))
    return fields


def capture_vm_stat():
    result = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True)
    counts = {}
    page_size = None
    header = re.search(r"page size of (\d+) bytes", result.stdout)
    if header is not None:
        page_size = int(header.group(1))
    for line in result.stdout.splitlines():
        match = re.match(r'^"?(.*?)"?: +(\d+)\.$', line)
        if match is not None:
            counts[match.group(1)] = int(match.group(2))
    return page_size, counts


def vm_fields(counts):
    fields = {}
    for out_name, candidates in VM_KEY_CANDIDATES.items():
        fields[out_name] = next((counts[key] for key in candidates if key in counts), None)
    return fields


def sample_swap():
    swap = capture_swap()
    page_size, counts = capture_vm_stat()
    swap["vm_page_size_bytes"] = page_size
    return swap, vm_fields(counts)


def stable_pair(previous, current):
    previous_swap, previous_fields = previous
    swap, fields = current
    if previous_swap["used_bytes"] is None or swap["used_bytes"] is None:
        return False
    if abs(swap["used_bytes"] - previous_swap["used_bytes"]) > SWAP_THRESHOLD_BYTES:
        return False
    previous_outs = previous_fields.get("swapouts")
    outs = fields.get("swapouts")
    if previous_outs is None or outs is None:
        return False
    return outs - previous_outs <= SWAPOUTS_THRESHOLD_PAGES


def wait_for_stable_baseline():
    previous = sample_swap()
    stable = 0
    polls = 0
    for polls in range(1, STABILITY_MAX_POLLS + 1):
        time.sleep(STABILITY_INTERVAL_SECONDS)
        current = sample_swap()
        stable = stable + 1 if stable_pair(previous, current) else 0
        previous = current
        if stable >= STABILITY_SAMPLES - 1:
            return current[0], True, polls
    return current[0], False, polls


def max_rss_bytes():
    # macOS reports ru_maxrss in bytes; Linux reports it in KiB.
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw if sys.platform == "darwin" else raw * 1024


def current_rss_bytes():
    result = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True, check=True
    )
    return int(result.stdout.strip()) * 1024


class FootprintError(RuntimeError):
    """The process footprint cannot be read. The run is invalid."""


def read_rusage_v2_words(platform_name=None):
    platform_name = sys.platform if platform_name is None else platform_name
    if platform_name != "darwin":
        raise FootprintError("phys_footprint needs macOS, platform is " + platform_name)
    try:
        import ctypes

        lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
        buf = (ctypes.c_uint64 * RUSAGE_WORDS)()
        status = lib.proc_pid_rusage(os.getpid(), RUSAGE_INFO_V2, ctypes.byref(buf))
    except Exception as exc:
        raise FootprintError("proc_pid_rusage call failed: " + type(exc).__name__) from exc
    if status != 0:
        raise FootprintError("proc_pid_rusage returned " + str(status))
    return list(buf)


def rusage_counters():
    """Return (phys_footprint, ri_resident_size) from one rusage read."""
    words = read_rusage_v2_words()
    if words[RUSAGE_V2_FOOTPRINT_INDEX] <= 0:
        raise FootprintError("proc_pid_rusage reported a zero footprint")
    return int(words[RUSAGE_V2_FOOTPRINT_INDEX]), int(words[RUSAGE_V2_RESIDENT_INDEX])


def phys_footprint_bytes():
    return rusage_counters()[0]


def mlx_call(mx, name):
    fn = getattr(mx, name, None)
    if not callable(fn):
        fn = getattr(getattr(mx, "metal", None), name, None)
    return fn() if callable(fn) else None


def mlx_api_resolution(mx):
    """Record where each cleanup call resolves. A null means the call is a no-op."""
    found = {}
    for name in ("clear_cache", "reset_peak_memory", "get_peak_memory"):
        if callable(getattr(mx, name, None)):
            found[name] = "mx"
        elif callable(getattr(getattr(mx, "metal", None), name, None)):
            found[name] = "mx.metal"
        else:
            found[name] = None
    return found


def mlx_int(mx, name):
    value = mlx_call(mx, name)
    return int(value) if value is not None else None


def coerce_number(value):
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return value


def metal_limits(mx):
    limits = {"available": False, "resolved_names": {}}
    metal = getattr(mx, "metal", None)
    if metal is None:
        limits["error"] = "mlx.core.metal attribute missing"
        return limits
    try:
        limits["available"] = bool(metal.is_available())
    except Exception:
        limits["available"] = True
    info_name = next(
        (name for name in ("device_info", "get_device_info") if callable(getattr(metal, name, None))),
        None,
    )
    limits["resolved_names"]["device_info"] = info_name
    raw = {}
    if info_name is not None:
        try:
            raw = getattr(metal, info_name)()
        except Exception as exc:
            limits["error"] = "device_info call failed: " + type(exc).__name__
    limits["raw_device_info"] = raw
    for out_name, candidates in METAL_INFO_CANDIDATES.items():
        for key in candidates:
            if isinstance(raw, dict) and key in raw:
                limits[out_name] = coerce_number(raw[key])
                limits["resolved_names"][out_name] = key
                break
    return limits


def step_residency(previous, baseline, row):
    """Return (step growth, cumulative growth, resident_ok) from footprint rows.

    Both checks must pass: at least 50% of the block per step and at least 90%
    of the cumulative allocated bytes since the baseline row.
    """
    step_growth = row["phys_footprint_bytes"] - previous["phys_footprint_bytes"]
    cumulative_growth = row["phys_footprint_bytes"] - baseline["phys_footprint_bytes"]
    step_ok = step_growth >= RESIDENCY_FRACTION * BLOCK_BYTES
    cumulative_ok = cumulative_growth >= CUMULATIVE_FRACTION * row["cumulative_bytes"]
    return step_growth, cumulative_growth, step_ok and cumulative_ok


def capture_row(step, cumulative_bytes):
    swap = capture_swap()
    page_size, counts = capture_vm_stat()
    footprint, resident = rusage_counters()
    row = {
        "step": step,
        "block_bytes": BLOCK_BYTES,
        "cumulative_bytes": cumulative_bytes,
        "swap_used_bytes": swap["used_bytes"],
        "vm_page_size_bytes": page_size,
        "phys_footprint_bytes": footprint,
        "ri_resident_size_bytes": resident,
        "current_rss_bytes": current_rss_bytes(),
        "max_rss_bytes": max_rss_bytes(),
    }
    row.update(vm_fields(counts))
    return row


def print_progress(row):
    swap_mib = (row["swap_used_bytes"] or 0) / 1024 ** 2
    print(
        f"step {row['step']:3d}"
        f" cumulative {row['cumulative_bytes'] / 1024 ** 3:6.2f} GiB"
        f" swap {swap_mib:8.1f} MiB"
        f" footprint {row['phys_footprint_bytes'] / 1024 ** 3:6.2f} GiB"
        f" rss {row['current_rss_bytes'] / 1024 ** 3:6.2f} GiB"
        f" resident_ok {row['resident_ok']}"
    )


def swap_started(row):
    used_delta = row["swap_used_delta_bytes"]
    outs_delta = row["swapouts_delta_pages"]
    if used_delta is not None and used_delta > SWAP_THRESHOLD_BYTES:
        return True
    return outs_delta is not None and outs_delta > SWAPOUTS_THRESHOLD_PAGES


def apply_deltas(row, baseline, previous_swapouts):
    used = row["swap_used_bytes"]
    row["swap_used_delta_bytes"] = used - baseline["used_bytes"] if used is not None else None
    outs = row["swapouts"]
    if outs is not None and previous_swapouts is not None:
        row["swapouts_delta_pages"] = outs - previous_swapouts
    else:
        row["swapouts_delta_pages"] = None
    return outs if outs is not None else previous_swapouts


def run_steps(mx, baseline, physical_bytes):
    arrays = []
    steps = []
    stop_reason = "steps_done"
    previous_swapouts = baseline["swapouts_pages"]
    previous = baseline["sample"]
    for step in range(1, MAX_STEPS + 1):
        if step * BLOCK_BYTES > physical_bytes:
            stop_reason = "physical_cap"
            break
        try:
            # Random data defeats the compressor, so pressure is real.
            block = mx.random.uniform(shape=(BLOCK_ELEMS,), dtype=mx.float32)
            mx.eval(block)
            peak = mlx_int(mx, "get_peak_memory")
            mlx_call(mx, "reset_peak_memory")
            mlx_call(mx, "clear_cache")
            arrays.append(block)
        except Exception as exc:
            stop_reason = "allocation_raised:" + type(exc).__name__
            break
        time.sleep(STEP_SETTLE_SECONDS)
        try:
            row = capture_row(step, step * BLOCK_BYTES)
        except FootprintError as exc:
            print("footprint unavailable: " + str(exc), file=sys.stderr)
            stop_reason = "invalid_footprint_unavailable"
            break
        row["mlx_peak_memory_bytes"] = peak
        row["mlx_active_memory_bytes"] = mlx_int(mx, "get_active_memory")
        row["mlx_cache_memory_bytes"] = mlx_int(mx, "get_cache_memory")
        previous_swapouts = apply_deltas(row, baseline, previous_swapouts)
        if row["swap_used_delta_bytes"] is None or row["swapouts_delta_pages"] is None:
            row["resident_ok"] = False
            steps.append(row)
            stop_reason = "invalid_swap_counters_unavailable"
            break
        growth, cumulative_growth, ok = step_residency(previous, baseline["sample"], row)
        row["resident_growth_bytes"] = growth
        row["footprint_cumulative_growth_bytes"] = cumulative_growth
        row["resident_ok"] = ok
        previous = row
        steps.append(row)
        print_progress(row)
        if swap_started(row):
            stop_reason = "swap_start"
            break
        if not row["resident_ok"]:
            stop_reason = "invalid_not_resident"
            break
    return arrays, steps, stop_reason


def largest_no_swap_bytes(steps):
    largest = 0
    for row in steps:
        if not swap_started(row) and row.get("resident_ok", False):
            largest = row["cumulative_bytes"]
    return largest


def validity(stop_reason, largest, physical_bytes):
    """Return (valid, invalid_reason). Only a swap start with resident steps gives a measurement."""
    if stop_reason == "invalid_footprint_unavailable":
        return False, "footprint_unavailable"
    if stop_reason == "invalid_not_resident":
        return False, "allocated_bytes_not_resident"
    if largest > physical_bytes:
        return False, "largest_no_swap_exceeds_physical"
    if stop_reason != "swap_start":
        return False, "swap_start_not_observed:" + stop_reason
    if largest <= 0:
        return False, "no_resident_step_before_swap"
    return True, None


def build_report(mx, context, largest):
    valid, invalid_reason = validity(context["stop_reason"], largest, context["physical_bytes"])
    reserve = context["physical_bytes"] - largest if valid else None
    return {
        "schema": REPORT_SCHEMA,
        "valid": valid,
        "invalid_reason": invalid_reason,
        "residency_fraction": RESIDENCY_FRACTION,
        "cumulative_fraction": CUMULATIVE_FRACTION,
        "residency_counter": "phys_footprint",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "machine": context["machine"],
        "physical_bytes": context["physical_bytes"],
        "metal_limits": context["metal_limits"],
        "swap_baseline": context["swap_baseline"],
        "stop_reason": context["stop_reason"],
        "steps": context["steps"],
        "largest_no_swap_cumulative_bytes": largest,
        "os_reserve_bytes": reserve,
        "os_reserve_gib": reserve / 1024 ** 3 if reserve is not None else None,
        "python_version": platform.python_version(),
        "mlx_version": getattr(mx, "__version__", "unknown"),
        "mlx_api": mlx_api_resolution(mx),
    }


def fp_row(footprint, cumulative, rss=1024 ** 3, comp=0):
    return {"phys_footprint_bytes": footprint, "cumulative_bytes": cumulative,
            "current_rss_bytes": rss, "pages_compressed": comp}


def simulate(footprint_steps, rss=1024 ** 3, comp_steps=None):
    """Judge a series of footprints like run_steps. Return (rows, stop reason)."""
    base = fp_row(1024 ** 3, 0, rss)
    previous = base
    rows = []
    for index, footprint in enumerate(footprint_steps, start=1):
        comp = comp_steps[index - 1] if comp_steps else 0
        row = fp_row(footprint, index * BLOCK_BYTES, rss, comp)
        _step, _cum, ok = step_residency(previous, base, row)
        row["resident_ok"] = ok
        row["swap_used_delta_bytes"] = 0
        row["swapouts_delta_pages"] = 0
        rows.append(row)
        previous = row
        if not ok:
            return rows, "invalid_not_resident"
    return rows, "swap_start"


def self_test():
    gib = 1024 ** 3
    grow = [gib + n * BLOCK_BYTES for n in range(1, 5)]
    # RSS flat while the footprint grows: valid.
    rows, stop = simulate(grow, rss=gib)
    assert stop == "swap_start" and all(r["resident_ok"] for r in rows)
    assert largest_no_swap_bytes(rows) == 4 * BLOCK_BYTES
    assert validity(stop, largest_no_swap_bytes(rows), 36 * gib) == (True, None)
    # Footprint flat while RSS grows: invalid.
    rows, stop = simulate([gib] * 4)
    assert stop == "invalid_not_resident" and not rows[0]["resident_ok"]
    assert validity(stop, 0, 36 * gib)[0] is False
    assert (False, "allocated_bytes_not_resident") == validity(stop, 0, 36 * gib)
    # Compressor growth with a flat footprint earns no credit.
    rows, stop = simulate([gib] * 3, comp_steps=[BLOCK_BYTES // 16384 * n for n in (1, 2, 3)])
    assert stop == "invalid_not_resident"
    # Whole-run start allowance does not exist: the first step needs its own growth.
    rows, stop = simulate([gib + BLOCK_BYTES // 4])
    assert stop == "invalid_not_resident"
    # 50% per step passes the step check but fails the 90% cumulative check.
    half = [gib + n * BLOCK_BYTES // 2 for n in range(1, 4)]
    base = fp_row(gib, 0)
    step, cum, ok = step_residency(base, base, fp_row(half[0], BLOCK_BYTES))
    assert step == BLOCK_BYTES // 2 and not ok
    rows, stop = simulate(half)
    assert stop == "invalid_not_resident"
    # Near the 90% cumulative boundary at step 9 (step growth stays above 50%).
    prev9 = fp_row(gib + int(7.55 * BLOCK_BYTES), 8 * BLOCK_BYTES)
    assert not step_residency(prev9, base, fp_row(gib + int(8.09 * BLOCK_BYTES), 9 * BLOCK_BYTES))[2]
    assert step_residency(prev9, base, fp_row(gib + int(8.11 * BLOCK_BYTES), 9 * BLOCK_BYTES))[2]
    # 95% growth per block stays valid.
    near = [gib + int(n * BLOCK_BYTES * 0.95) for n in range(1, 6)]
    assert simulate(near)[1] == "swap_start"
    # The footprint reader fails clearly off macOS.
    try:
        read_rusage_v2_words("linux")
        raise AssertionError("expected FootprintError")
    except FootprintError:
        pass
    assert validity("invalid_footprint_unavailable", 0, 36 * gib) == (False, "footprint_unavailable")
    phys = 36 * gib
    assert validity("swap_start", 20 * gib, phys) == (True, None)
    assert validity("invalid_not_resident", 0, phys)[0] is False
    assert validity("steps_done", 64 * gib, phys)[1] == "largest_no_swap_exceeds_physical"
    assert validity("steps_done", 20 * gib, phys)[0] is False
    assert validity("swap_start", 0, phys)[0] is False
    assert validity("physical_cap", 20 * gib, phys) == (False, "swap_start_not_observed:physical_cap")
    assert validity("invalid_swap_counters_unavailable", 20 * gib, phys)[0] is False
    rows = [{"cumulative_bytes": 1, "resident_ok": True, "swap_used_delta_bytes": 0, "swapouts_delta_pages": 0},
            {"cumulative_bytes": 2, "resident_ok": False, "swap_used_delta_bytes": 0, "swapouts_delta_pages": 0}]
    assert largest_no_swap_bytes(rows) == 1
    print("self-test ok")
    return 0


def warm_up(mx):
    """Initialize Metal so device setup does not count as block residency."""
    mx.eval(mx.zeros(16))
    mlx_call(mx, "clear_cache")


def calibrate(mx, output_path):
    """Allocate one block and require at least 256 MiB of footprint growth."""
    warm_up(mx)
    footprint_before, resident_before = rusage_counters()
    rss_before = current_rss_bytes()
    free_before = vm_fields(capture_vm_stat()[1])["pages_free"]
    block = mx.random.uniform(shape=(BLOCK_ELEMS,), dtype=mx.float32)
    mx.eval(block)
    peak = mlx_int(mx, "get_peak_memory")
    mlx_call(mx, "clear_cache")
    time.sleep(STEP_SETTLE_SECONDS)
    footprint_after, resident_after = rusage_counters()
    footprint_growth = footprint_after - footprint_before
    free_after = vm_fields(capture_vm_stat()[1])["pages_free"]
    free_delta = free_before - free_after if None not in (free_before, free_after) else None
    result = {
        "schema": CALIBRATION_SCHEMA,
        "valid": footprint_growth >= CALIBRATION_MIN_BYTES,
        "block_bytes": BLOCK_BYTES,
        "min_footprint_growth_bytes": CALIBRATION_MIN_BYTES,
        "footprint_growth_bytes": footprint_growth,
        "ri_resident_size_growth_bytes": resident_after - resident_before,
        "rss_growth_bytes": current_rss_bytes() - rss_before,
        "free_pages_decrease": free_delta,
        "mlx_peak_memory_bytes": peak,
        "mlx_active_memory_bytes": mlx_int(mx, "get_active_memory"),
        "mlx_cache_memory_bytes": mlx_int(mx, "get_cache_memory"),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "mlx_version": getattr(mx, "__version__", "unknown"),
        "mlx_api": mlx_api_resolution(mx),
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result))
    del block
    return 0 if result["valid"] else 3


def main():
    if len(sys.argv) == 2 and sys.argv[1] == "--self-test":
        return self_test()
    if len(sys.argv) == 3 and sys.argv[1] == "--calibrate":
        import mlx.core as mx

        try:
            return calibrate(mx, sys.argv[2])
        except FootprintError as exc:
            print("footprint unavailable: " + str(exc), file=sys.stderr)
            return 1
    if len(sys.argv) != 2:
        print("usage: probe.py <output.json> | --calibrate <output.json> | --self-test", file=sys.stderr)
        return 2
    import mlx.core as mx

    footprint_error = None
    try:
        warm_up(mx)
        phys_footprint_bytes()
    except FootprintError as exc:
        footprint_error = str(exc)
        print("footprint unavailable: " + footprint_error, file=sys.stderr)

    physical_bytes = int(sysctl("hw.memsize"))
    swap_baseline, stable, polls = wait_for_stable_baseline()
    _page_size, vm_counts = capture_vm_stat()
    swap_baseline["pre_vm_stat"] = vm_counts
    swap_baseline["pre_max_rss_bytes"] = max_rss_bytes()
    swap_baseline["stability_polls"] = polls
    swap_baseline["stable"] = stable
    baseline_used = swap_baseline["used_bytes"]
    if baseline_used is None:
        print("vm.swapusage parse failed", file=sys.stderr)
        return 1
    arrays = []
    steps = []
    stop_reason = "steps_done"
    if footprint_error is not None:
        stop_reason = "invalid_footprint_unavailable"
    elif stable:
        try:
            sample = capture_row(0, 0)
        except FootprintError as exc:
            print("footprint unavailable: " + str(exc), file=sys.stderr)
            sample = None
            stop_reason = "invalid_footprint_unavailable"
        if sample is not None:
            baseline = {
                "used_bytes": baseline_used,
                "swapouts_pages": vm_fields(vm_counts)["swapouts"],
                "sample": sample,
            }
            arrays, steps, stop_reason = run_steps(mx, baseline, physical_bytes)
    else:
        stop_reason = "unstable_baseline"
    largest = largest_no_swap_bytes(steps)
    context = {
        "machine": sysctl("hw.model"),
        "physical_bytes": physical_bytes,
        "metal_limits": metal_limits(mx),
        "swap_baseline": swap_baseline,
        "stop_reason": stop_reason,
        "steps": steps,
    }
    report = build_report(mx, context, largest)
    with open(sys.argv[1], "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
        handle.write("\n")
    arrays.clear()
    gc.collect()
    reserve = report["os_reserve_bytes"]
    reserve_text = f"{reserve / 1024 ** 3:.2f} GiB" if reserve is not None else "none"
    print(
        f"stop {stop_reason} valid {report['valid']} reason {report['invalid_reason']}"
        f" largest_no_swap {largest / 1024 ** 3:.2f} GiB"
        f" os_reserve {reserve_text}"
    )
    return 0 if report["valid"] else 4


if __name__ == "__main__":
    sys.exit(main())
