# Native thread qualification

This executable provides two separate modes for `P0A-04`.
The default mode keeps one owner thread and two channel callers under the supplemental contract (concurrent-caller-qualification.md, in the ledger).
The `gate` mode follows the two-thread contract (p0a04-two-thread-gate.md, in the ledger).
Each of its two worker threads creates, uses, and drops its own thread-confined adapter.
The engine dispatch guard serializes native work. No mode permits unsynchronized backend calls.
The run is its own process and the sole MLX owner in that process.
A supervisor enforces a 120-second deadline and stops only its own child.

The portable library owns the identifier ledger, both protocols, and the supervisor.
Its Linux tests use compute stand-ins. They establish protocol behavior, not native execution.
The Mac module runs a 10,000-job serial baseline, then the selected mode, followed by registration, replacement, and memory observations.
Both modes check actual results against the pinned fixture literals from decision 0011.

The default mode remains supplemental evidence. It cannot close the two-thread native gate.
The gate mode requires 5,000 jobs on each native owner and a third-constructor rejection while both owners remain live.
Its observer runs only after both worker threads join.
Registration counts must remain unchanged, and replacement construction must succeed after teardown.
Neither mode qualifies real driver failure, unsynchronized native dispatch, or full model execution.

Run the portable suite from Linux:

```sh
cargo fmt --manifest-path qualification/native-threads/Cargo.toml --all -- --check
cargo clippy --offline --locked --manifest-path qualification/native-threads/Cargo.toml --all-targets -- -D warnings
cargo test --offline --locked --manifest-path qualification/native-threads/Cargo.toml
```

Activate the task environment on the Mac.
Set `INFERENCE_MLX_SOURCE` to the pinned clean checkout at `64ea011cb65f14d9ce2737e60db9a4ae91ed7441`.
Run the supervised supplemental mode:

```sh
cargo run --offline --locked --manifest-path qualification/native-threads/Cargo.toml
```

Run the supervised two-thread gate separately:

```sh
cargo run --offline --locked --manifest-path qualification/native-threads/Cargo.toml -- gate
```

Do not use the internal `run` or `run-gate` child modes for qualification. They bypass the process supervisor.
JSON records identify memory observations, warmup, baseline, selected execution mode, final registrations, and replacement.
Gate records also identify the initialized owner threads and the expected third-owner rejection.
A count change, unexpected backend error, validation failure, or deadline expiry fails the run.
Only gate mode expects the exact bounded-lease initialization error from its third constructor.
The observer links the same pinned MLX library as the engine and adds no production symbol or Rust diagnostic interface.
