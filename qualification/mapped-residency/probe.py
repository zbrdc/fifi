"""Phase 1 mapped-residency probe: map the stepping-stone shards read-only and measure residency, working set, and reclaim under owned pressure."""

import argparse
import gc
import json
import mmap
import os
import platform
import random
import re
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone

BLOCK_BYTES = 512 * 1024 * 1024
BLOCK_ELEMS = BLOCK_BYTES // 4
TOUCH_STEP_BYTES = 512 * 1024 * 1024
WINDOW_BYTES = 16 * 1024 * 1024
WINDOWS_PER_STEP = 64
STEADY_STEPS = 8
STEADY_SEED = 20260928
MAX_PRESSURE_STEPS = 128
SWAP_THRESHOLD_BYTES = 8 * 1024 * 1024
SWAPOUTS_THRESHOLD_PAGES = 512
STEP_SETTLE_SECONDS = 2.0
STABILITY_INTERVAL_SECONDS = 5.0
STABILITY_SAMPLES = 4
STABILITY_MAX_POLLS = 24
RESIDENCY_FRACTION = 0.5
CUMULATIVE_FRACTION = 0.9
RUSAGE_INFO_V2 = 2
RUSAGE_WORDS = 64
RUSAGE_V2_RESIDENT_INDEX = 8
RUSAGE_V2_FOOTPRINT_INDEX = 9
REPORT_SCHEMA = "p1-mapped-residency/5"
PHASE_ORDER = ("steady", "pressure", "sweep")
SWEEP_FIRST_ORDER = ("steady", "sweep", "pressure")
ORDERS = {"default": PHASE_ORDER, "sweep-first": SWEEP_FIRST_ORDER}

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
    current = previous
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


class FootprintError(RuntimeError):
    """The process footprint cannot be read. The pressure run is invalid."""


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
    """Return (footprint, ri_resident_size) from one rusage read."""
    words = read_rusage_v2_words()
    footprint = words[RUSAGE_V2_FOOTPRINT_INDEX]
    if footprint <= 0:
        raise FootprintError("proc_pid_rusage reported a zero footprint")
    return int(footprint), int(words[RUSAGE_V2_RESIDENT_INDEX])


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


def warm_up(mx):
    """Initialize Metal so device setup does not count as block residency."""
    mx.eval(mx.zeros(16))
    mlx_call(mx, "clear_cache")


def current_rss_bytes():
    result = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True, check=True
    )
    return int(result.stdout.strip()) * 1024


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


def swap_started(row):
    used_delta = row["swap_used_delta_bytes"]
    outs_delta = row["swapouts_delta_pages"]
    if used_delta is not None and used_delta > SWAP_THRESHOLD_BYTES:
        return True
    return outs_delta is not None and outs_delta > SWAPOUTS_THRESHOLD_PAGES


def apply_deltas(row, state):
    used = row["swap_used_bytes"]
    base = state["baseline"]["used_bytes"]
    row["swap_used_delta_bytes"] = used - base if used is not None and base is not None else None
    outs = row["swapouts"]
    previous = state["previous_swapouts"]
    row["swapouts_delta_pages"] = outs - previous if outs is not None and previous is not None else None
    if outs is not None:
        state["previous_swapouts"] = outs


def counters_or_none(required):
    try:
        return rusage_counters()
    except FootprintError:
        if required:
            raise
        return None, None


def capture_row(phase, step, cumulative_bytes, state, require_footprint=False):
    swap = capture_swap()
    page_size, counts = capture_vm_stat()
    footprint, resident_size = counters_or_none(require_footprint)
    row = {
        "phase": phase,
        "step": step,
        "cumulative_bytes": cumulative_bytes,
        "swap_used_bytes": swap["used_bytes"],
        "vm_page_size_bytes": page_size,
        "phys_footprint_bytes": footprint,
        "ri_resident_size_bytes": resident_size,
        "current_rss_bytes": current_rss_bytes(),
        "max_rss_bytes": max_rss_bytes(),
    }
    row.update(vm_fields(counts))
    apply_deltas(row, state)
    return row


def print_progress(row):
    swap_mib = (row["swap_used_bytes"] or 0) / 1024 ** 2
    print(
        f"{row['phase']:8s} step {row['step']:3d}"
        f" cumulative {row['cumulative_bytes'] / 1024 ** 3:6.2f} GiB"
        f" swap {swap_mib:8.1f} MiB"
        f" footprint {(row.get('phys_footprint_bytes') or 0) / 1024 ** 3:6.2f} GiB"
        f" rss {row['current_rss_bytes'] / 1024 ** 3:6.2f} GiB"
    )


def touch_span(mm, start, length):
    # One byte read per page is enough to fault the page in.
    touched = 0
    end = start + length
    pos = start
    while pos < end and pos < len(mm):
        mm[pos]
        touched += 1
        pos += mmap.PAGESIZE
    return touched


def discover_shards(checkpoint_dir):
    names = sorted(name for name in os.listdir(checkpoint_dir) if re.match(r"model-.*\.safetensors$", name))
    if not names:
        raise SystemExit("no model-*.safetensors files in " + checkpoint_dir)
    return [(name, os.stat(os.path.join(checkpoint_dir, name)).st_size) for name in names]


def map_shards(checkpoint_dir, shards):
    mappings = []
    for name, _size in shards:
        handle = open(os.path.join(checkpoint_dir, name), "rb")
        mappings.append((name, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)))
    return mappings


def sweep_rows(mappings, state):
    rows = []
    cumulative = 0
    step = 0
    for _name, mm in mappings:
        size = len(mm)
        pos = 0
        while pos < size:
            take = min(TOUCH_STEP_BYTES, size - pos)
            touch_span(mm, pos, take)
            pos += take
            cumulative += take
            step += 1
            time.sleep(STEP_SETTLE_SECONDS)
            row = capture_row("sweep", step, cumulative, state)
            rows.append(row)
            print_progress(row)
            if swap_started(row):
                return rows, "swap_start"
    return rows, "steps_done"


def steady_rows(mappings, mode, madvise_errors, state):
    if mode == "random":
        for name, mm in mappings:
            try:
                mm.madvise(mmap.MADV_RANDOM)
            except Exception as exc:
                madvise_errors.append(name + ": " + type(exc).__name__)
    rng = random.Random(STEADY_SEED)
    rows = []
    for step in range(1, STEADY_STEPS + 1):
        for _ in range(WINDOWS_PER_STEP):
            _name, mm = mappings[rng.randrange(len(mappings))]
            max_start = max(0, len(mm) - WINDOW_BYTES)
            start = rng.randrange(0, max_start // mmap.PAGESIZE + 1) * mmap.PAGESIZE
            touch_span(mm, start, min(WINDOW_BYTES, len(mm) - start))
        time.sleep(STEP_SETTLE_SECONDS)
        cumulative = step * WINDOWS_PER_STEP * WINDOW_BYTES
        row = capture_row("steady", step, cumulative, state)
        rows.append(row)
        print_progress(row)
        if swap_started(row):
            return rows, "swap_start"
    return rows, "steps_done"


def pressure_resident_ok(start, previous, row):
    """Check that the owned blocks became resident, by process footprint.

    Each step needs footprint growth of at least 50% of the block. The growth
    since the phase start also needs at least 90% of the cumulative bytes.
    Compressor growth and RSS give no credit, and no start allowance applies.
    """
    step_growth = row["phys_footprint_bytes"] - previous["phys_footprint_bytes"]
    total_growth = row["phys_footprint_bytes"] - start["phys_footprint_bytes"]
    return (
        step_growth >= RESIDENCY_FRACTION * BLOCK_BYTES
        and total_growth >= CUMULATIVE_FRACTION * row["cumulative_bytes"]
    )


def pressure_validity(stop_reason, largest, physical_bytes):
    """Return (valid, invalid_reason) for the pressure phase."""
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


def pressure_rows(mx, state, physical_bytes):
    arrays = []
    state["pressure_arrays"] = arrays
    rows = []
    stop_reason = "steps_done"
    try:
        start = capture_row("pressure", 0, 0, state, require_footprint=True)
    except FootprintError as exc:
        print("footprint unavailable: " + str(exc), file=sys.stderr)
        return rows, "invalid_footprint_unavailable"
    state["pressure_start_rss_bytes"] = start["current_rss_bytes"]
    previous = start
    for step in range(1, MAX_PRESSURE_STEPS + 1):
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
            row = capture_row("pressure", step, step * BLOCK_BYTES, state, require_footprint=True)
        except FootprintError as exc:
            print("footprint unavailable: " + str(exc), file=sys.stderr)
            stop_reason = "invalid_footprint_unavailable"
            break
        if row["swap_used_delta_bytes"] is None or row["swapouts_delta_pages"] is None:
            row["resident_ok"] = False
            rows.append(row)
            stop_reason = "invalid_swap_counters_unavailable"
            break
        row["mlx_peak_memory_bytes"] = peak
        row["mlx_active_memory_bytes"] = mlx_int(mx, "get_active_memory")
        row["mlx_cache_memory_bytes"] = mlx_int(mx, "get_cache_memory")
        row["resident_growth_bytes"] = row["phys_footprint_bytes"] - previous["phys_footprint_bytes"]
        row["footprint_cumulative_growth_bytes"] = row["phys_footprint_bytes"] - start["phys_footprint_bytes"]
        row["resident_ok"] = pressure_resident_ok(start, previous, row)
        previous = row
        rows.append(row)
        print_progress(row)
        if swap_started(row):
            stop_reason = "swap_start"
            break
        if not row["resident_ok"]:
            stop_reason = "invalid_not_resident"
            break
    return rows, stop_reason


def last_rss(rows):
    return rows[-1]["current_rss_bytes"] if rows else None


def pressure_skip_reason(order, phases):
    """Return why pressure must not run in the sweep-first order, or None."""
    if order != "sweep-first":
        return None
    sweep = phases.get("sweep")
    if sweep is None or sweep["stop_reason"] == "not_requested":
        return "not_run_sweep_not_requested"
    if sweep["stop_reason"] != "steps_done":
        return "not_run_sweep_incomplete:" + sweep["stop_reason"]
    return None


def pressure_rss_summary(pressure):
    """Return (start, end, min) process RSS for the pressure phase, or Nones."""
    start = pressure.get("start_rss_bytes")
    values = [row["current_rss_bytes"] for row in pressure["rows"]]
    if start is not None:
        values.insert(0, start)
    if not values:
        return None, None, None
    return values[0], values[-1], min(values)


def build_report(args, context, phases, derived):
    return {
        "schema": REPORT_SCHEMA,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "machine": context["machine"],
        "physical_bytes": context["physical_bytes"],
        "metal_limits": context["metal_limits"],
        "swap_baseline": context["swap_baseline"],
        "checkpoint_dir": args.checkpoint_dir,
        "shards": context["shards"],
        "mapped_total_bytes": context["mapped_total_bytes"],
        "madvise": args.madvise,
        "madvise_errors": context["madvise_errors"],
        "order": context["order"],
        "phases_requested": context["phases_requested"],
        "phases": phases,
        "derived": derived,
        "python_version": platform.python_version(),
        "mlx_version": context["mlx_version"],
        "mlx_api": context["mlx_api"],
    }


def self_test():
    tmpdir = os.environ.get("TMPDIR")
    if not tmpdir:
        sys.exit("set TMPDIR to the task scratch directory")
    path = os.path.join(tmpdir, "mapped-residency-selftest.bin")
    size = 4 * mmap.PAGESIZE + 123
    with open(path, "wb") as handle:
        handle.write(b"\x00" * size)
    with open(path, "rb") as handle:
        mm = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        assert touch_span(mm, 0, size) == (size + mmap.PAGESIZE - 1) // mmap.PAGESIZE
        assert touch_span(mm, 0, 0) == 0
        assert touch_span(mm, size, 10) == 0
        mm.close()
    os.unlink(path)
    state = {"baseline": {"used_bytes": 100}, "previous_swapouts": 10}
    row = {"swap_used_bytes": 100 + SWAP_THRESHOLD_BYTES + 1, "swapouts": 10}
    apply_deltas(row, state)
    assert swap_started(row)
    row = {"swap_used_bytes": 100, "swapouts": 10 + SWAPOUTS_THRESHOLD_PAGES + 1}
    apply_deltas(row, state)
    assert swap_started(row)
    row = {"swap_used_bytes": 100, "swapouts": state["previous_swapouts"]}
    apply_deltas(row, state)
    assert not swap_started(row)
    gib = 1024 ** 3
    page = 16384
    start = {"phys_footprint_bytes": gib}

    def mk(step, footprint, rss=gib, comp=0):
        return {"step": step, "cumulative_bytes": step * BLOCK_BYTES, "phys_footprint_bytes": footprint,
                "current_rss_bytes": rss, "pages_compressed": comp}

    # RSS flat, footprint grows: ok.
    assert pressure_resident_ok(start, start, mk(1, gib + BLOCK_BYTES, rss=gib))
    # Footprint flat, RSS grows: invalid.
    assert not pressure_resident_ok(start, start, mk(1, gib, rss=gib + BLOCK_BYTES))
    # Compressor growth earns no credit.
    assert not pressure_resident_ok(start, start, mk(1, gib, comp=BLOCK_BYTES // page))
    # No start allowance: a flat footprint at a later step fails.
    assert not pressure_resident_ok(start, mk(8, gib + 8 * BLOCK_BYTES), mk(9, gib + 8 * BLOCK_BYTES))
    # Late steps still need footprint growth: 3 blocks gained by step 9 fails.
    assert not pressure_resident_ok(start, mk(8, gib + 3 * BLOCK_BYTES), mk(9, gib + 3 * BLOCK_BYTES))
    # Step growth passes at 60% but cumulative gain below 90% fails.
    assert not pressure_resident_ok(start, mk(8, gib + 2 * BLOCK_BYTES), mk(9, gib + 2 * BLOCK_BYTES + int(0.6 * BLOCK_BYTES)))
    # 60% per step passes the step check, but 5.4 blocks by step 9 is below 90%.
    assert not pressure_resident_ok(start, mk(8, gib + int(4.8 * BLOCK_BYTES)), mk(9, gib + int(5.4 * BLOCK_BYTES)))
    # Near the 90% cumulative boundary at step 9 (step growth stays above 50%).
    prev = mk(8, gib + int(7.55 * BLOCK_BYTES))
    assert not pressure_resident_ok(start, prev, mk(9, gib + int(8.09 * BLOCK_BYTES)))
    assert pressure_resident_ok(start, prev, mk(9, gib + int(8.11 * BLOCK_BYTES)))
    # The MLX API record names each resolved call or null.
    class FakeMx:
        class metal:
            reset_peak_memory = staticmethod(lambda: None)
        clear_cache = staticmethod(lambda: None)
    assert mlx_api_resolution(FakeMx) == {
        "clear_cache": "mx", "reset_peak_memory": "mx.metal", "get_peak_memory": None}
    try:
        read_rusage_v2_words("linux")
        raise AssertionError("expected FootprintError")
    except FootprintError:
        pass
    assert pressure_validity("invalid_footprint_unavailable", 0, 36 * gib) == (False, "footprint_unavailable")
    phys = 36 * gib
    assert pressure_validity("swap_start", 20 * gib, phys) == (True, None)
    assert pressure_validity("invalid_not_resident", 0, phys)[0] is False
    assert pressure_validity("steps_done", 64 * gib, phys)[1] == "largest_no_swap_exceeds_physical"
    assert pressure_validity("steps_done", 20 * gib, phys)[0] is False
    assert pressure_validity("physical_cap", 20 * gib, phys) == (False, "swap_start_not_observed:physical_cap")
    assert pressure_validity("invalid_swap_counters_unavailable", 0, phys)[0] is False
    rows = [{"cumulative_bytes": 1, "resident_ok": True, "swap_used_delta_bytes": 0, "swapouts_delta_pages": 0},
            {"cumulative_bytes": 2, "resident_ok": False, "swap_used_delta_bytes": 0, "swapouts_delta_pages": 0}]
    assert largest_no_swap_bytes(rows) == 1
    assert REPORT_SCHEMA == "p1-mapped-residency/5"
    assert ORDERS["default"] == ("steady", "pressure", "sweep")
    assert ORDERS["sweep-first"] == ("steady", "sweep", "pressure")
    assert pressure_skip_reason("default", {"sweep": {"stop_reason": "swap_start"}}) is None
    assert pressure_skip_reason("sweep-first", {"sweep": {"stop_reason": "steps_done"}}) is None
    assert pressure_skip_reason("sweep-first", {"sweep": {"stop_reason": "swap_start"}}) == \
        "not_run_sweep_incomplete:swap_start"
    assert pressure_skip_reason("sweep-first", {"sweep": {"stop_reason": "not_requested"}}) == \
        "not_run_sweep_not_requested"
    assert run_order_selftest() == 0
    ns = argparse.Namespace
    assert madvise_needs_steady(ns(madvise="random"), {"sweep", "pressure"})
    assert not madvise_needs_steady(ns(madvise="random"), {"steady", "sweep"})
    assert not madvise_needs_steady(ns(madvise="normal"), {"sweep"})
    ctx = {k: None for k in ("machine", "physical_bytes", "metal_limits", "swap_baseline", "shards",
                             "mapped_total_bytes", "madvise_errors", "mlx_version", "mlx_api")}
    ctx.update(order="sweep-first", phases_requested="steady,sweep,pressure")
    rep = build_report(ns(checkpoint_dir="x", madvise="normal"), ctx, {}, {})
    assert rep["order"] == "sweep-first" and rep["phases_requested"] == "steady,sweep,pressure"
    print("self-test ok")
    return 0


def run_order_selftest():
    """Drive run_phases with stub runners to check order, skip, and derived keys."""
    global reset_phase_baseline, phase_runners
    saved = reset_phase_baseline, phase_runners
    calls = []
    sweep_reason = {"value": "steps_done"}

    def stub_runners(ctx):
        def pressure():
            ctx["state"]["pressure_start_rss_bytes"] = 900
            return [{"current_rss_bytes": 800, "cumulative_bytes": 1, "resident_ok": True,
                     "swap_used_delta_bytes": 0, "swapouts_delta_pages": 0},
                    {"current_rss_bytes": 850, "cumulative_bytes": 2, "resident_ok": True,
                     "swap_used_delta_bytes": 0, "swapouts_delta_pages": 0}], "swap_start"
        return {
            "steady": lambda: (calls.append("steady") or [], "steps_done"),
            "sweep": lambda: (calls.append("sweep") or [], sweep_reason["value"]),
            "pressure": lambda: (calls.append("pressure"), pressure())[1],
        }

    reset_phase_baseline = lambda state: None
    phase_runners = stub_runners

    class StubMx:
        pass

    def ctx(order, wanted):
        return {"mx": StubMx, "state": {}, "wanted": set(wanted), "order": order}

    try:
        shards = [("a", 10)]
        both = ("steady", "sweep", "pressure")
        phases = run_phases(ctx("sweep-first", both))
        assert calls == ["steady", "sweep", "pressure"], calls
        d = derive(shards, phases, 1 << 40, "sweep-first", set(both))
        assert d["phase_order"] == ["steady", "sweep", "pressure"]
        assert (d["pressure_start_rss_bytes"], d["pressure_end_rss_bytes"], d["pressure_min_rss_bytes"]) == (900, 850, 800)
        assert d["pressure_valid"] is True
        calls.clear()
        run_phases(ctx("default", both))
        assert calls == ["steady", "pressure", "sweep"], calls
        calls.clear()
        d = derive(shards, run_phases(ctx("default", ("sweep", "pressure"))), 1 << 40, "default", {"sweep", "pressure"})
        assert calls == ["pressure", "sweep"] and d["phase_order"] == ["pressure", "sweep"]
        calls.clear()
        sweep_reason["value"] = "swap_start"
        phases = run_phases(ctx("sweep-first", ("sweep", "pressure")))
        assert calls == ["sweep"], calls
        assert phases["pressure"] == {"stop_reason": "not_run_sweep_incomplete:swap_start", "rows": []}
        d = derive(shards, phases, 1 << 40, "sweep-first", {"sweep", "pressure"})
        assert d["pressure_valid"] is False
        assert d["pressure_invalid_reason"] == "not_run_sweep_incomplete:swap_start"
        assert d["pressure_start_rss_bytes"] is None and d["pressure_min_rss_bytes"] is None
        calls.clear()
        phases = run_phases(ctx("sweep-first", ("pressure",)))
        assert calls == [] and phases["pressure"]["stop_reason"] == "not_run_sweep_not_requested"
    finally:
        reset_phase_baseline, phase_runners = saved
    return 0


def reset_phase_baseline(state):
    # Swap deltas are measured against the start of each phase so one phase
    # cannot trip the next phase's threshold.
    swap = capture_swap()
    _page_size, counts = capture_vm_stat()
    outs = vm_fields(counts)["swapouts"]
    state["baseline"]["used_bytes"] = swap["used_bytes"]
    state["baseline"]["swapouts_pages"] = outs
    state["previous_swapouts"] = outs


def phase_runners(ctx):
    mappings, state = ctx["mappings"], ctx["state"]
    return {
        "steady": lambda: steady_rows(mappings, ctx["madvise_mode"], ctx["madvise_errors"], state),
        "pressure": lambda: pressure_rows(ctx["mx"], state, ctx["physical_bytes"]),
        "sweep": lambda: sweep_rows(mappings, state),
    }


def run_phases(ctx):
    phases = {}
    runners = phase_runners(ctx)
    order = ctx.get("order", "default")
    for name in ORDERS[order]:
        if name not in ctx["wanted"]:
            phases[name] = {"stop_reason": "not_requested", "rows": []}
            continue
        if name == "pressure":
            skip = pressure_skip_reason(order, phases)
            if skip is not None:
                print("pressure skipped: " + skip, file=sys.stderr)
                phases[name] = {"stop_reason": skip, "rows": []}
                continue
        reset_phase_baseline(ctx["state"])
        rows, reason = runners[name]()
        phases[name] = {"stop_reason": reason, "rows": rows}
        if name == "pressure":
            phases[name]["start_rss_bytes"] = ctx["state"].pop("pressure_start_rss_bytes", None)
            ctx["state"].pop("pressure_arrays", None)
            mlx_call(ctx["mx"], "clear_cache")
            gc.collect()
    return phases


def parse_args():
    parser = argparse.ArgumentParser(description="Mapped-residency probe for the stepping-stone checkpoint.")
    parser.add_argument("checkpoint_dir", nargs="?")
    parser.add_argument("output", nargs="?")
    parser.add_argument(
        "--madvise",
        choices=("normal", "random"),
        default="normal",
        help="normal: no hint. random: apply MADV_RANDOM to every mapping before the steady phase; "
        "requires the steady phase in --phases (exit 2 otherwise).",
    )
    parser.add_argument("--phases", default=",".join(PHASE_ORDER))
    parser.add_argument(
        "--order",
        choices=tuple(ORDERS),
        default="default",
        help="phase order. default: steady, pressure, sweep. sweep-first: steady, sweep, pressure; "
        "the mappings stay open through pressure, and pressure is skipped (report invalid, exit 4) "
        "unless the sweep ends steps_done. --phases still selects which phases run.",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def wanted_phases(args):
    wanted = args.phases.split(",")
    unknown = [name for name in wanted if name not in PHASE_ORDER]
    if unknown or not wanted:
        raise SystemExit("--phases must be a comma list of " + ",".join(PHASE_ORDER))
    return set(wanted)


def prepare_baseline():
    physical_bytes = int(sysctl("hw.memsize"))
    swap_baseline, stable, polls = wait_for_stable_baseline()
    _page_size, vm_counts = capture_vm_stat()
    swap_baseline["pre_vm_stat"] = vm_counts
    swap_baseline["pre_max_rss_bytes"] = max_rss_bytes()
    swap_baseline["stability_polls"] = polls
    swap_baseline["stable"] = stable
    return physical_bytes, swap_baseline, stable, vm_counts


def baseline_state(swap_baseline, vm_counts):
    outs = vm_fields(vm_counts)["swapouts"]
    return {
        "baseline": {
            "used_bytes": swap_baseline["used_bytes"],
            "swapouts_pages": outs,
        },
        "previous_swapouts": outs,
    }


def requested_order(order, wanted):
    return [name for name in ORDERS[order] if name in wanted]


def derive(shards, phases, physical_bytes, order="default", wanted=PHASE_ORDER):
    pressure = phases["pressure"]
    largest = largest_no_swap_bytes(pressure["rows"])
    rss_start, rss_end, rss_min = pressure_rss_summary(pressure)
    if pressure["stop_reason"] == "not_requested":
        valid, reason = None, None
    elif pressure["stop_reason"].startswith("not_run_"):
        valid, reason = False, pressure["stop_reason"]
    else:
        valid, reason = pressure_validity(pressure["stop_reason"], largest, physical_bytes)
    return {
        "mapped_total_bytes": sum(size for _name, size in shards),
        "phase_order": requested_order(order, wanted),
        "pressure_start_rss_bytes": rss_start,
        "pressure_end_rss_bytes": rss_end,
        "pressure_min_rss_bytes": rss_min,
        "sweep_final_rss_bytes": last_rss(phases["sweep"]["rows"]),
        "steady_final_rss_bytes": last_rss(phases["steady"]["rows"]),
        "pressure_valid": valid,
        "pressure_invalid_reason": reason,
        "largest_no_swap_owned_bytes_with_mapping": largest if valid else None,
        "pressure_unvalidated_largest_bytes": largest,
    }


def emit_report(args, context, phases, derived):
    report = build_report(args, context, phases, derived)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
        handle.write("\n")
    print(
        "stop sweep=" + phases["sweep"]["stop_reason"]
        + " steady=" + phases["steady"]["stop_reason"]
        + " pressure=" + phases["pressure"]["stop_reason"]
        + f" pressure_valid={derived['pressure_valid']}"
        + f" reason={derived['pressure_invalid_reason']}"
        + f" largest_no_swap_owned={derived['largest_no_swap_owned_bytes_with_mapping']}"
    )


def madvise_needs_steady(args, wanted):
    return args.madvise == "random" and "steady" not in wanted


def main():
    args = parse_args()
    if args.self_test:
        return self_test()
    wanted = wanted_phases(args)
    if madvise_needs_steady(args, wanted):
        print("--madvise random requires the steady phase in --phases", file=sys.stderr)
        return 2
    import mlx.core as mx

    warm_up(mx)  # the pressure phase reports a missing footprint as invalid

    physical_bytes, swap_baseline, stable, vm_counts = prepare_baseline()
    if swap_baseline["used_bytes"] is None:
        print("vm.swapusage parse failed", file=sys.stderr)
        return 1
    shards = discover_shards(args.checkpoint_dir)
    mappings = map_shards(args.checkpoint_dir, shards)
    madvise_errors = []
    state = baseline_state(swap_baseline, vm_counts)
    if stable:
        ctx = {
            "mx": mx,
            "physical_bytes": physical_bytes,
            "mappings": mappings,
            "state": state,
            "madvise_mode": args.madvise,
            "madvise_errors": madvise_errors,
            "wanted": wanted,
            "order": args.order,
        }
        phases = run_phases(ctx)
    else:
        phases = {
            name: {"stop_reason": "unstable_baseline", "rows": []} for name in PHASE_ORDER
        }
    derived = derive(shards, phases, physical_bytes, args.order, wanted)
    context = {
        "machine": sysctl("hw.model"),
        "physical_bytes": physical_bytes,
        "metal_limits": metal_limits(mx),
        "swap_baseline": swap_baseline,
        "shards": [{"name": name, "bytes": size} for name, size in shards],
        "mapped_total_bytes": derived["mapped_total_bytes"],
        "madvise_errors": madvise_errors,
        "phases_requested": args.phases,
        "order": args.order,
        "mlx_version": getattr(mx, "__version__", "unknown"),
        "mlx_api": mlx_api_resolution(mx),
    }
    emit_report(args, context, phases, derived)
    for _name, mm in mappings:
        mm.close()
    return 0 if derived["pressure_valid"] is not False else 4


def largest_no_swap_bytes(rows):
    largest = 0
    for row in rows:
        if not swap_started(row) and row.get("resident_ok", False):
            largest = row["cumulative_bytes"]
    return largest


if __name__ == "__main__":
    sys.exit(main())
