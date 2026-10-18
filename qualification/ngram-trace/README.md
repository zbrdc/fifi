# ngram-trace

This standalone qualification harness checks synthetic PLE row hashes and page
simulations. The metadata command uses the engine's bounded `GgufHeader`
parser and reads PLE values through checked header metadata ranges. It does not
scan or hash the complete input prefixes.

## Checks

Run the portable harness on the Mac with task-local scratch paths:

```sh
cargo fmt --manifest-path qualification/ngram-trace/Cargo.toml -- --check
cargo clippy --offline --manifest-path qualification/ngram-trace/Cargo.toml --all-targets -- -D warnings
cargo test --offline --manifest-path qualification/ngram-trace/Cargo.toml
```

The synthetic tests pass explicit EOS, three positional multipliers, 16
vocabularies, and 16 offsets. They check chronological input, two-token heads
before three-token heads, unsigned 64-bit wrapping, EOS reset behavior, and
row bounds. These values are test fixtures, not Flash-Next metadata.

## G1 metadata command

The `metadata` command takes the directory holding the two G1 input prefixes.
It parses each header with `GgufHeader`, which bounds header reads to
`HEADER_LIMIT`, and verifies the recorded header SHA-256. It checks the split
set and n-gram tensor geometry. It reads PLE arrays only from checked
`ArrayMeta.byte_range` values. Header hashes cover header bytes only. They do
not identify shard payloads. The input prefixes may contain bytes after
`data_start`. The command does not read or hash those bytes.

This recovery did not run the command because it reads target PLE values. The
actual multiplier, offset, vocabulary-size, and EOS metadata values remain
unverified here. The synthetic executable test exercises metadata extraction
without reading past its synthetic header. The expected EOS value `248044`
appears in the cited plan, but this run did not read it from the checkpoint. Do
not treat the synthetic test fixture values as target values. The current-EOS
test provisionally retains preceding context. This recovery did not resolve
that edge against the pinned llama.cpp source, so it is not target EOS parity
evidence.

```sh
cargo run --offline --manifest-path qualification/ngram-trace/Cargo.toml -- \
  metadata /Users/dan/fifi/scrch/1c06bd1/p1-gguf-g1/prefix
```

The command loads no target tokenizer or prompt set. It cannot establish
vocabulary-only tokenizer parity, real-prompt row parity, physical I/O, or
performance. A real M2 trace remains blocked until a verified vocabulary-only
route, actual metadata, and pinned reference execution are available.

## Page simulation

`page_counts` accepts row IDs, absolute tensor offset, row bytes, and table row
count. It counts each distinct 4 KiB and 16 KiB file page intersecting the
entire half-open row range. The simulator includes the tensor offset and rows
that cross page boundaries. Its counts are address simulations, not physical
I/O measurements.
