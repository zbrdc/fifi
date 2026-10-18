# Tiered weights load check

This command loads the four pinned stepping-stone shards through `TieredWeights::from_readers`.
It wraps each `File` in a counting reader.
It checks `resident_bytes()` and the total bytes read.
It fails if bytes returned by the readers exceed the header bytes plus the resident figure.
Kernel readahead is not measured.

The expected resident figure is 1,389,394,176 bytes.
An independent audit of the real shard headers derived it.
The audit sums every tensor outside `.mlp.switch_mlp.`.
The expected byte count read is the header bytes plus that figure.

```sh
cargo run --locked --offline --manifest-path qualification/weights-tiered/Cargo.toml -- \
  /Users/dan/edge0/models/edge0-35b
```

Exit zero means both figures match. Exit one reports a mismatch. Exit two reports invalid input.
The command only reads the checkpoint. It uses no GPU.

This command does not run the S5 int8 numerical gate (pending check 2).
Decision 0016 repaired metadata only and did not qualify an int8 native operation.
The weights plan requires an int8 parity gate on the Mac.
No pinned int8 MLX reference fixture exists yet. That gate remains open.

Item 24 is partial. S4 passes. S5 remains open.

The run log `load.log` does not record checkpoint identity.
Shard sizes match `docs/environment/mac-phase0a.json`.
The run did not recompute shard hashes.
