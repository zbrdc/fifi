# Native copying-import measurement

This Mac-only command measures the pinned MLX C copying import from an owned anonymous mapping.
It records the source mapping, process RSS, MLX allocator bytes, copy time, GPU completion time, and retained bytes after array release.
The caller imports `uint8` values, changes two source bytes, and records whether the imported values stay unchanged.
It then runs a GPU add, records the first and last output values, and times a full host copy.
The command prints JSON lines and exits nonzero on a native failure.
It is a diagnostic. It adds no production engine interface or executable test.

The `BYTES` argument has a maximum of `1073741824`.
The command caps its own MLX allocator at 3 GiB.
Run a smaller diagnostic before the declared 1 GiB case.
Check Mac memory pressure, swap, and other workloads before each case.
Use the pinned source checkout and a task-specific `CARGO_TARGET_DIR`, `TMPDIR`, `TMP`, and `TEMP`.
The command measures an anonymous mapping, not direct file-backed import.
The `mlx_array_new_data` call copies into MLX-owned storage, as the pinned C header declares.
GPU addition measures a later materialization path.
It does not establish a zero-copy managed import, full-model latency, or safe concurrent use.

The `adapter` argument runs 128 completed cycles through the existing public affine-int4 adapter fixture.
It reports the fixture's typed staging bytes and native allocator bytes after selected adapter releases.
Its elapsed value covers staging, four MLX imports, GPU work, synchronization, and host readback together.
It does not isolate those internal phases or replace the 1,000-group parity result.

```sh
cargo run --offline --locked --release \
  --manifest-path qualification/native-import/Cargo.toml -- 1073741824
```
