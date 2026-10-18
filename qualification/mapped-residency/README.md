# Mapped-residency probe

This probe measures how a read-only mapping of the stepping-stone checkpoint
shards behaves in physical memory on the Apple Silicon Mac. It provides the
measurement evidence for the mapped-page residency decision in
`docs/decisions/0027-memory-residency.md`.

The engine budget is 32,212,254,720 bytes (30.00 GiB), from two schema
`p0a05-os-reserve/4` runs. The stepping-stone base shards total 19,508,787,456
bytes (19.5 GB). The probe measures whether a mapped load fits and how much owned memory fits beside it.

## Phases

1. `baseline`: wait for stable swap usage before any phase starts. The probe
   stops with `unstable_baseline` when stability does not occur.
2. `steady`: touch 64 windows of 16 MiB per step for 8 steps with a fixed
   seed. This measures a bounded working set. With `--madvise random` the
   probe applies `MADV_RANDOM` to every mapping before this phase.
3. `pressure`: allocate 512 MiB MLX arrays until swap starts while the
   mapping stays held. Each array holds `mx.random.uniform` data. Zero
   blocks also became resident in the first measurement, but the compressor
   absorbed the zero pages, so zero fill gave no real pressure. Random data
   does not compress materially. The kernel can still move the blocks into the
   compressor. `phys_footprint` counts compressed pages, so the pressure figure
   counts owned bytes in the footprint. It does not prove uncompressed
   residency. This phase
   measures the largest no-swap owned total beside the mapping. MLX Metal
   buffers do not enter RSS, so residency uses the process `phys_footprint`
   from `proc_pid_rusage(RUSAGE_INFO_V2)` through ctypes (`rusage_info_v2`
   word 9). RSS stays in each row as a recorded column only. Every step
   must grow the footprint by at least 50% of the block size. The growth
   since the phase start must also be at least 90% of the owned bytes.
   Compressor growth and start RSS give no credit. Compressed owned pages still
   count in `phys_footprint`. A failed check stops the
   phase with `invalid_not_resident`. If the footprint call fails or the
   host is not macOS, the phase stops with `invalid_footprint_unavailable`
   and is invalid. If a swap counter delta is unavailable, the phase stops
   with `invalid_swap_counters_unavailable` and is invalid. The loop stops
   with `physical_cap` before owned bytes exceed physical memory, and that
   result is invalid. The probe warms up Metal before the baseline row, and
   calls the MLX `clear_cache` (`mx` or `mx.metal`, as the `mlx_api` field
   records) after each block with the MLX peak recorded. Before the sweep
   phase, the probe releases the pressure arrays, calls `clear_cache`, and
   runs `gc.collect()`.
4. `sweep`: touch every mapped page once in 512 MiB steps. This measures
   worst-case full-load mapped residency.

Each phase resets its swap deltas at its start and stops at swap start. One
phase cannot trip the next phase threshold.

## Phase order and selection

`--phases` selects which phases run. It never orders them. `--order` sets the
order:

- `default`: steady, pressure, sweep.
- `sweep-first`: steady, sweep, pressure. The mappings stay open through
  pressure.

`--phases sweep,pressure` with the default order runs pressure before the
sweep. Pressure then runs on a cold mapping and does not measure owned memory
beside a full sweep. Use `--order sweep-first` for that measurement.

In the `sweep-first` order, owned pressure can evict clean mapped pages before
swap starts. The pressure start, end, and minimum RSS fields record that
eviction. The probe does not stop on RSS loss.

In the `sweep-first` order, the probe skips pressure unless the sweep ends with
`steps_done`. The skip reasons are `not_run_sweep_not_requested` and
`not_run_sweep_incomplete:<reason>`. A skip makes the pressure figure invalid,
and the probe exits 4.

`--madvise random` applies `MADV_RANDOM` before the steady phase. It requires
`steady` in `--phases`. The probe exits 2 before it maps any shard otherwise.

## Run on the Mac

Use schema `p1-mapped-residency/5`. Run on an idle machine. Record the Git
`HEAD` and the sha256 of `probe.py` and `check.py` (if present) with each
report. From the Mac checkout root:

```sh
/Users/dan/edge0/.venv/bin/python qualification/mapped-residency/probe.py \
  /Users/dan/edge0/models/edge0-35b \
  /Users/dan/fifi/scrch/<base-commit>/<task>/mapped-residency/normal-1.json \
  --madvise normal --phases steady,sweep,pressure --order sweep-first
```

Run the `normal` measurement first. Then run the same command with
`--madvise random` and a new output name. Do not interleave the two runs.
Copy the reports and their `shasum -a 256` lines to the Linux task scratch
before handoff.

## Self-test

The self-test checks the page-touch, swap-threshold, and residency-validity
helpers with synthetic footprint series. It needs no checkpoint or MLX:

```sh
python3 qualification/mapped-residency/probe.py --self-test
```

## Report

The report uses schema `p1-mapped-residency/5`. It records the machine, the
physical byte count, Metal limits, the swap baseline, the shard list, the
mapped total, the madvise mode, `order`, `phases_requested`, one row per phase
step with swap and VM counters plus RSS, and derived totals:
`mapped_total_bytes`, `sweep_final_rss_bytes`, `steady_final_rss_bytes`,
`pressure_valid`, `pressure_invalid_reason`,
`largest_no_swap_owned_bytes_with_mapping`, and
`pressure_unvalidated_largest_bytes`. The report also records `mlx_api`, an
additive object that names where `clear_cache`, `reset_peak_memory`, and
`get_peak_memory` resolved (`mx`, `mx.metal`, or null). A null `clear_cache`
means cached buffers can mix with owned blocks. Steady, pressure, and sweep
rows record `phys_footprint_bytes` and `ri_resident_size_bytes` (both can be
null, recorded, not gated), `current_rss_bytes`, and `max_rss_bytes`.
Pressure rows also record `resident_growth_bytes`,
`footprint_cumulative_growth_bytes`, `resident_ok`, `mlx_peak_memory_bytes`,
`mlx_active_memory_bytes`, and `mlx_cache_memory_bytes`.
`pressure_invalid_reason` can be `footprint_unavailable`. The pressure phase
records `start_rss_bytes`. `derived.phase_order` lists the requested phases in
run order. It also lists a phase that was skipped or did not run. Read each
phase `stop_reason` for the executed state. `derived` also holds
`pressure_start_rss_bytes`,
`pressure_end_rss_bytes`, and `pressure_min_rss_bytes`. RSS includes clean
mapped pages. `phys_footprint` excludes them. A phase that is not requested reports `stop_reason`
`not_requested`.

The pressure figure is valid only when the phase stops with `swap_start`,
every counted step has `resident_ok` true, and the largest no-swap bytes do
not exceed the physical bytes. Otherwise `pressure_valid` is false,
`pressure_invalid_reason` names the cause, and
`largest_no_swap_owned_bytes_with_mapping` is null. Do not use an invalid
figure for decision 0027. The probe exits 4 when `pressure_valid` is false.

`mx.random.uniform` creates an intermediate array, so the transient peak can
approach two blocks. The figure can be conservative by one block.

`pages_compressed` records `Pages stored in compressor`, a source page count.
The probe does not record `Pages occupied by compressor` per row. A later probe
change can record both keys. That change needs its own task and review.
