# OS_RESERVE probe

`probe.py` measures the `OS_RESERVE` constant for the phase 0 memory budget.
The scheduler computes `budget = physical_bytes - OS_RESERVE` (scheduler.md:62, in the ledger).
The current value is 6,442,450,944 bytes from two schema `p0a05-os-reserve/4` runs (scheduler.md:156, in the ledger).
The measurement contract allocates MLX arrays in 512 MiB steps and watches for swap start (scheduler.md:158-160, in the ledger).
`OS_RESERVE` equals physical bytes minus the largest no-swap cumulative allocated bytes.

The probe allocates `float32` MLX arrays of 512 MiB with `mx.random.uniform`, keeps every array alive, and records host state after each step.
Zero-filled blocks also became resident in the first measurement. The compressor then absorbed the zero pages, so zero fill gave no real pressure.
Random data defeats compression, so the random fill stays resident and creates pressure.
MLX Metal buffers do not enter RSS. The probe gates residency on the process `phys_footprint`, read through `proc_pid_rusage(RUSAGE_INFO_V2)` with ctypes (`rusage_info_v2` word 9, byte 72).
RSS from `ps` and `ri_resident_size` (word 8) stay in each row as recorded columns only.
After each step the footprint must rise by at least 50% of the block size (`residency_fraction`).
The cumulative footprint growth since the baseline row must also be at least 90% of the cumulative allocated bytes (`cumulative_fraction`).
The probe grants no compressor credit and no start allowance.
A step that fails either check stops the run with `invalid_not_resident`.
If the footprint call fails or the host is not macOS, the run stops with `invalid_footprint_unavailable` and is invalid.
The probe warms up Metal before the baseline row, so device setup does not count as block growth.
After each `mx.eval(block)` it records the MLX peak memory, resets the peak, and calls the MLX `clear_cache` (`mx` or `mx.metal`, as the `mlx_api` field records).
Before the first step it waits for a stable swap baseline: four consecutive samples every 5 seconds with at most 8 MiB swap growth and at most 512 swapout pages between samples.
It stops when the swap baseline is never stable within 24 samples, when swap use grows by more than 8 MiB above the settled baseline, when swapouts grow by more than 512 pages within one step, when an allocation raises, when the allocated bytes would exceed physical memory (`physical_cap`, invalid), or after 128 steps.
If a swap counter delta is unavailable, the run stops with `invalid_swap_counters_unavailable` and is invalid.
Each step waits 2 seconds after the allocation before it records host state, so background reclamation settles first.
It writes the result JSON. It exits zero when `valid` is true and 4 when `valid` is false.

Check the validity logic on any host without MLX:

```sh
python3 qualification/os-reserve/probe.py --self-test
```

Run the one-block calibration first. It allocates one 512 MiB block and is valid only when the footprint grows by at least 256 MiB:

```sh
/Users/dan/edge0/.venv/bin/python qualification/os-reserve/probe.py --calibrate <calibration.json>
```

Record and compare: `footprint_growth_bytes` (gated, at least 268,435,456), `ri_resident_size_growth_bytes`, `free_pages_decrease`, and `rss_growth_bytes` (not gated). Exit 3 means invalid. Do not start the full run then.

Run it on the Apple Silicon Mac from the Mac checkout root. Wrap it in `/usr/bin/time -l` and compare the peak footprint with the final cumulative bytes:

```sh
/Users/dan/edge0/.venv/bin/python qualification/os-reserve/probe.py <output.json>
```

Write `output.json` to persistent task scratch. Close model-serving workloads before the run.
The probe needs no engine code and uses only `mlx` and the standard library.

Output JSON schema, `p0a05-os-reserve/4`:

| Key | Type | Contents |
|---|---|---|
| `schema` | string | `p0a05-os-reserve/4` |
| `valid` | boolean | True only when swap start was observed, every counted step was resident, and the largest no-swap bytes do not exceed physical bytes |
| `invalid_reason` | string or null | `allocated_bytes_not_resident`, `footprint_unavailable`, `largest_no_swap_exceeds_physical`, `swap_start_not_observed:<stop_reason>`, `no_resident_step_before_swap`, or null |
| `residency_fraction` | number | Minimum footprint growth per step as a fraction of the block size |
| `cumulative_fraction` | number | Minimum cumulative footprint growth as a fraction of cumulative bytes |
| `residency_counter` | string | `phys_footprint` |
| `recorded_at_utc` | string | UTC timestamp of the run |
| `machine` | string | `sysctl hw.model` |
| `physical_bytes` | integer | `sysctl hw.memsize` |
| `metal_limits` | object | Metal device limits and the attribute names that resolved |
| `swap_baseline` | object | Settled `vm.swapusage`, `vm_stat` summary, stability poll count, stability flag, and max RSS |
| `stop_reason` | string | `swap_start`, `invalid_not_resident`, `invalid_footprint_unavailable`, `invalid_swap_counters_unavailable`, `physical_cap`, `allocation_raised:<Exception>`, `unstable_baseline`, or `steps_done` |
| `steps` | array | One row per step: index, block bytes, cumulative bytes, swap used, swap growth over the settled baseline, swapout growth over the previous sample, `vm_stat` page counts, `phys_footprint_bytes`, `ri_resident_size_bytes` (recorded, not gated), current RSS, max RSS, `mlx_peak_memory_bytes`, `mlx_active_memory_bytes`, `mlx_cache_memory_bytes`, `resident_growth_bytes` (footprint step growth), `footprint_cumulative_growth_bytes`, `resident_ok` |
| `largest_no_swap_cumulative_bytes` | integer | Cumulative bytes at the last resident step before swap start. Zero if swap started before step 1 |
| `os_reserve_bytes` | integer or null | `physical_bytes` minus `largest_no_swap_cumulative_bytes`. Null when `valid` is false |
| `os_reserve_gib` | number or null | `os_reserve_bytes` divided by 1024^3 |
| `mlx_api` | object | Where `clear_cache`, `reset_peak_memory`, and `get_peak_memory` resolved (`mx`, `mx.metal`, or null). A null `clear_cache` means cached buffers can mix with active blocks |
| `python_version` | string | Interpreter version |
| `mlx_version` | string | MLX version |

Result limits:

- The probe observes whole-host swap use, so other processes can trigger the stop or shift the measured value.
- The stability pre-check rejects runs on a host with heavy background swap activity. An `unstable_baseline` result is not a measurement.
- It measures the host OS reserve. It does not measure Metal pressure semantics such as the recommended working set, wired limits, or allocator cache behavior.
- One run produces one sample. The recorded `metal_limits` and page counts support later comparison, not attribution.
- Only a run with `valid` true gives an `OS_RESERVE` figure. A run that reaches the step cap, raises an allocation error, or stops on residency gives no figure.
- Compressor growth earns no residency credit.
- `mx.random.uniform` creates an intermediate array, so the transient peak can approach two blocks. The figure can be conservative by one block.
- A calibration file records `p0a05-os-reserve-calibration/1` and does not give an `OS_RESERVE` figure.
