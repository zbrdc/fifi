# Native lifecycle qualification

This Mac-only executable checks the public `MlxKernels` lifecycle under decision 0012 (docs/decisions/0012-native-stream-lifetime.md, in the ledger).
It computes the existing independent fixture and checks exact output after adapter drop.
After warmup, 128 same-thread cycles and 32 sequential new-thread cycles must preserve the GPU registration count.
Each new thread finishes before the next thread starts.

The C++ observer counts device-zero GPU registrations through public `mlx::core::get_streams()`.
It links the same pinned MLX library as the engine. It does not create or change streams.
Production execution uses the C binding. The observer adds no production symbol or Rust diagnostic interface.
The observer requires clean MLX headers at commit `64ea011cb65f14d9ce2737e60db9a4ae91ed7441`.
`INFERENCE_MLX_SOURCE` selects that source checkout. The build checks its revision and tracked changes.

Run these commands from the project root after activating the task's persistent Mac environment:

```sh
cargo fmt --manifest-path qualification/native-lifetime/Cargo.toml --all -- --check
cargo clippy --offline --locked --manifest-path qualification/native-lifetime/Cargo.toml --all-targets -- -D warnings
cargo run --offline --locked --manifest-path qualification/native-lifetime/Cargo.toml
```

Each JSON line records the stage, cycle when applicable, and registration count after completed work.
A count increase, invalid output, failed thread, or observer failure terminates the executable unsuccessfully.
This executable is separate from portable tests and requires Apple Silicon with the pinned native build environment.

The result establishes bounded registrations for successful replacement. It does not measure bytes, throughput, concurrent adapters, or failed synchronization.
The original and intermediate failures remain in persistent scratch with source hashes.
The implementation record (native-lifetime-implementation.md, in the ledger) records evidence and coverage limits.
