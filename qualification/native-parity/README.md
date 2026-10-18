# Native real-group parity

This qualification compares the existing Rust affine-int4 operation with the pinned Python MLX operation.
It uses 1,000 groups from the recorded Qwen3.6 stepping-stone checkpoint.
It does not implement model generation or qualify Flash-Next capacity.
Decision 0015 (docs/decisions/0015-real-group-parity.md, in the ledger) defines the acceptance contract.

## Preparation

Use the isolated Mac environment recorded in the result (native-parity-implementation.md, in the ledger).
Keep the checkpoint and prepared files quiescent throughout the run.
The commands validate file identities. They do not lock files against another writer.

```sh
TMPDIR=TASK_SCRATCH python -B qualification/native-parity/prepare.py \
  --checkpoint /Users/dan/edge0/models/edge0-35b \
  --output NEW_DIRECTORY
```

Preparation hashes every source shard before and after extraction.
It preserves original U32 and BF16 slice bytes in small safetensors files.
It freezes deterministic coordinates and both MLX reference probes in `manifest.json`.
Capture the printed `manifest_sha256` independently before running Rust.
Both commands refuse an existing destination.

```sh
cargo run --locked --offline \
  --manifest-path qualification/native-parity/Cargo.toml -- \
  --cases PREPARED_DIRECTORY \
  --manifest-sha256 CAPTURED_DIGEST \
  --output NEW_REPORT.json
```

The report retains source identities, reference versions, group coordinates, actual output hashes, and every numerical difference.
Exit zero requires a complete comparison with zero F32 value differences.
Exit one reports a numerical mismatch. Exit two reports invalid inputs, unavailable native execution, or a backend error.
Missing output or an interrupted process never establishes a passing gate.
Publication creates the destination from a completed temporary file without replacing an existing file.
A crash during publication can leave an extra temporary hard link.

## Tests

Both commands expose `--fixture` for synthetic command tests.
Fixture mode cannot qualify the real checkpoint gate, even when comparison passes.
The synthetic preparation source supplies `fixture-source.json` with file pins and a group count.
That count must cover all eligible modules.
No production engine method or test adapter bypasses MLX.

```sh
cargo test --locked --offline --manifest-path qualification/native-parity/Cargo.toml
TMPDIR=TASK_SCRATCH python -B qualification/native-parity/tests/test_prepare.py
```

Portable Rust tests cover input validation and unavailable native execution.
Mac tests additionally compare reference outputs, preserve signed-zero equality, and report a deliberate finite mismatch.
Python fixture preparation requires the pinned Mac packages and MLX binaries.
[The fixture record](tests/README.md) explains the independent expected values.
