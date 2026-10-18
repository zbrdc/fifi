# Rust storage qualification

`storage-spike` checks one regular safetensors file through the engine's owned mapped reader.
The initial read copies bytes into an anonymous mapping. This command does not execute a model.
Decision 0017 (docs/decisions/0017-owned-mapped-storage.md, in the ledger) defines its contract.

```sh
cargo run --locked --manifest-path qualification/storage-spike/Cargo.toml -- \
  --input FILE --sha256 HEX --max-input-bytes 268435456 \
  --tensor KEY --output REPORT [--mode MODE]
```

`MODE` selects the access policy. `owned_anonymous_map` is the default and copies the input into an anonymous mapping.
`direct_file_map` reads the same bytes through a read-only file mapping without the initial copy.
The mapped arm exists for the Phase 0B access-policy comparison only.
It parses the safetensors header through a harness-local checked walk with the same bounds rules.
It does not create a production weight source and does not qualify the P0A-02 mapping policy.
The mapping call is the one narrow unsafe operation in this crate. The harness verifies the pinned digest over the mapped bytes before it parses any entry.

Use a new report path in persistent task scratch.
Supply the independently pinned complete-file SHA-256 digest.
The command hashes consumed bytes and reads the retained tensor after releasing its source and file handle.
Its report includes build commit, build dirtiness, command, input identity, input limit, payload bytes, elapsed time, and tensor metadata.
The report retains each original argument as platform-encoded bytes in `command_os_bytes`, with its `runner_os`.
Display strings can replace non-UTF-8 characters. The recorded argument bytes preserve Linux and Mac filenames.
The input-byte limit does not include metadata, page rounding, caller allocations, or native copies.
Payload bytes do not measure resident pages.

Exit zero means this single-file storage run passed.
Argument errors exit two. Load, identity, tensor-key, and report-publication errors exit one.
After argument parsing, execution failures produce a report when publication succeeds.
Report publication uses a unique sibling temporary file and never replaces an existing destination.
The operator controls the report directory. The command does not protect against another user who can replace that directory.
A crash can leave a temporary sibling file. Existing reports remain unchanged.

Direct file mapping as a production weight source, backend import, thread stress, and full-model qualification remain explicitly unrun.
Build from a clean committed worktree for measured results.
Build-time Git identity does not replace the qualification evidence manifest and source hashes.
