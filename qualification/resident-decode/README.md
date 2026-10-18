# Resident decode qualification

This runner checks decision 0039 gates 3, 5, and 6 for the resident Flash-Next decode path on the Mac.
It runs the ignored tests in `engine/src/family/flash_next/resident/parity.rs` by their exact names.
It reads their reports, applies the gate rules in `src/gates.rs`, and writes one record.
It uses only the Rust standard library and does not link the engine.

## Builds

| Build | Engine features | Tests | Purpose |
| --- | --- | --- | --- |
| `q` | `native-mlx,native-fault-qualification` | teacher-forced, end-to-end, decode | Parity gates, synchronizations, forced retire flushes, peak retired bytes, memory |
| `p` | `native-mlx` | decode | Headline tokens per second without the counter overhead |

## Environment

| Variable | Use |
| --- | --- |
| `FLASH_NEXT_ARTIFACT` | Checkpoint directory. Every test reads it. |
| `FLASH_NEXT_ORACLE` | Oracle dump directory (`run1`). Build `q` only. |
| `FLASH_NEXT_CACHE_LIMIT` | MLX allocator cache limit in bytes. The decode test sets it and records it. |
| `MLX_RS_METAL_PATH` | Directory of `mlx.metallib`. The runner records its SHA-256. |

The runner sets `FLASH_NEXT_RESIDENT_BLOCKS_REPORT`, `FLASH_NEXT_RESIDENT_PROMPT_REPORT`, and `FLASH_NEXT_RESIDENT_DECODE_REPORT` for each test.

## Commands

```sh
cargo build --offline --release
"$CARGO_TARGET_DIR/release/resident-decode" run --engine ../../engine --out <dir> --build q
"$CARGO_TARGET_DIR/release/resident-decode" run --engine ../../engine --out <dir> --build p
"$CARGO_TARGET_DIR/release/resident-decode" check --out <dir> --build q
```

`run` writes these files in `<dir>`, with prefix `q` or `p`:

- `<b>-identity-before.txt` and `<b>-identity-after.txt`: source, hardware, artifact, swap, page-in, and port 8000 identity;
- `<b>-<test>.log`: the cargo output and the `/usr/bin/time -l` output;
- `<b>-<test>-report.txt`: the test report;
- `<b>-exits.txt`: the exit code of each test;
- `<b>-record.txt`: derived values, one line for each gate, and `verdict`.

`check` rewrites the record from the saved files.
The exit status is 0 only when every gate that the build requires passes. It is 1 for a failed gate and 2 for a usage or I/O error.

## Gates

| Record line | Rule | Required in |
| --- | --- | --- |
| `gate_tests_exit` | Every test exits with status 0. | `q`, `p` |
| `gate_build_matches` | The decode report `build` equals `--build`. | `q`, `p` |
| `gate_g3_teacher_forced` | `checks=1460` and `failures=0`. | `q` |
| `gate_g3_end_to_end` | 48 `layer_out` lines, one `logits t4` line, `device_token_t4`, and `failures=0`. | `q` |
| `gate_g5_first32` | The first 32 resident tokens equal the host tokens exactly. | `q`, `p` |
| `gate_g5_memory_flat` | `(max - min) / min <= 0.05` for the MLX peak of generated tokens 17 to 512. | `q`, `p` |
| `gate_g6_syncs` | Each generated token uses at most 110 synchronizations. | `q` |
| `gate_g6_counters` | Synchronizations, flushes, and peak retired bytes exist for every step. | `q` |

The thresholds are frozen. Do not change them after a result.
