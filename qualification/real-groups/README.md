# Real-group parity qualification kit (task 2, item 1)

This kit freezes 1,000 quantized groups from the real Edge0 35B checkpoint and
compares the engine's dequantized matmul against the pinned MLX build. It
implements the re-qualification of decision 0015 at base `52640f0` with the
parity environment, as recorded for task `P0A-05-01` in the ledger.

## Acceptance rule

The gate is the decision 0015 rule, from
`docs/decisions/0015-real-group-parity.md`:

> "The gate remains zero value differences in `f32`, including groups with
> negative biases."

Details of the rule, from the same decision:

- Exactly 1,000 groups. The report states the exact number of groups with
  negative biases and the gate fails if that number is zero.
- Comparison is exact numerical equality of `f32` values. There is no
  epsilon, relative tolerance, or ULP budget. Signed zeros (`+0.0` and
  `-0.0`) compare equal. The report records their bit-level differences as a
  separate count that does not affect the gate.
- Any NaN, any infinity, any shape mismatch, or any value difference fails
  the gate.
- Exit status of this prepare step: `0` is a complete frozen report with zero
  differences recorded in `counts`, `1` is a mismatch, `2` is invalid inputs
  or a backend failure. This script only prepares evidence. It exits `0`
  after freezing a complete fixture or `2` on any error. The zero-difference
  verdict comes from the Rust comparison step described in
  `rust-side-note.md`, which follows the exit convention above.

## Files in this kit

- `prepare-real-groups.py` — freezes the fixture and manifest from the real
  checkpoint. SHA-256 `a0d01615f17e8ba86289ddb4b1cfc27f14abcf63bfc98dc208ef8d73657e5e4a`,
  24,255 bytes at draft time.
- `README.md` — this file.
- `rust-side-note.md` — design note for the Rust comparison side.

## Pinned inputs

Checkpoint at `/Users/dan/edge0/models/edge0-35b` on the Mac. The script
checks the byte count and SHA-256 of each file before and after extraction
and refuses to run on any mismatch. Pins come from
`docs/environment/mac-phase0a.json`:

| File | Bytes | SHA-256 |
| --- | --- | --- |
| `config.json` | 23,591 | `a822a9e48b0aafbe3144ec37d4fb067e178ed96615ce6e4420b3149893cc5767` |
| `model.safetensors.index.json` | 180,182 | `e359cd95565275b923fed8df019dd8e869d9af5dc4a621b4114ab7a8e6c534d1` |
| `model-00001-of-00004.safetensors` | 4,395,019,264 | `f7fff28fecb8221e53787c6339c938b113bcb94c7bbc60d09a8826eba08ef1ab` |
| `model-00002-of-00004.safetensors` | 5,368,472,749 | `31dcdb1c49eebdb1505bd14e3cb33f9cf900bd2546b638f2464694ae763a033` |
| `model-00003-of-00004.safetensors` | 5,368,324,139 | `3e66de06a1f03dade16a612a368cfce4a4c9caa4efd7d28185454384082cec03` |
| `model-00004-of-00004.safetensors` | 4,377,211,365 | `a5d0cf03519c26f8b506df6b0ba60526e5c08c8cea22d0c21ce92950e58a5422` |

Reference backend: the pinned parity venv at
`/Users/dan/fifi/scrch/85dc6d0/p0a-oq04-mac/parity-env/venv`, containing
`mlx 0.32.2.dev20260927+1f8e74e3` with Metal. The script asserts the MLX
version string and asserts the SHA-256 of `core.cpython-*.so` as
`935cc4e023f4a3b548981bac866e7edb708256c42b04508f5f633f1c2f7839b3`, matching
the hash recorded in `oq04-parity-matrix.md` (in the ledger). The
manifest records the `libmlx.dylib` hash but does not assert it, for the same
separate-build reason as that review.

## Safetensors reader mode — check and record

The script needs no pip installs. It reads shard metadata in one of two modes
and records the mode in the fixture `provenance.reader` and in the manifest:

- `safetensors` — used when the venv provides `safetensors`, `numpy`, and
  `ml_dtypes` (bfloat16 numpy slices need `ml_dtypes`).
- `struct-fallback` — pure-Python read of the 8-byte little-endian header
  length, the JSON header, and mmap byte windows. Used otherwise.

Both modes were cross-checked against each other on synthetic shards on the
Linux drafting host (byte-identical reads, rank-2 and rank-3, plus rejection
of bad spans, out-of-range reads, and duplicate JSON names). The struct path
was the only mode available there because `ml_dtypes` is absent. Before the
real run, probe the venv on the Mac and record which mode ran:

```sh
/Users/dan/fifi/scrch/85dc6d0/p0a-oq04-mac/parity-env/venv/bin/python \
  -c "import ml_dtypes, numpy, safetensors; print('safetensors mode available')"
```

If the import fails, the script runs in `struct-fallback`. Both modes are
valid, but the evidence record must state which mode produced the fixture.

## Mac run steps

Run on the Mac Studio as the coordinating agent. The run takes about 10
minutes. Hashing roughly 19.5 GB of shards twice dominates, and the run
also executes 2,000 small MLX matmuls. Peak host memory stays below roughly
1 GB because the script reads only per-group byte windows.

```sh
VENV=/Users/dan/fifi/scrch/85dc6d0/p0a-oq04-mac/parity-env/venv
CKPT=/Users/dan/edge0/models/edge0-35b
EVID=/Users/dan/fifi/scrch/52640f0/p0a05-reserve-baseline/real-group

# 1. probe reader mode (see previous section) and record the output
"$VENV/bin/python" -c "import ml_dtypes, numpy, safetensors; print('safetensors mode available')"

# 2. freeze the fixture
"$VENV/bin/python" /path/to/this/kit/prepare-real-groups.py \
  --checkpoint "$CKPT" \
  --output "$EVID/run-1"

# 3. record the printed summary JSON (fixture sha256, manifest sha256, counts)

# 4. determinism check: a second run into a fresh directory must print the
#    same fixture sha256
"$VENV/bin/python" /path/to/this/kit/prepare-real-groups.py \
  --checkpoint "$CKPT" \
  --output "$EVID/run-2"
```

The script requires no environment variable. The script activates nothing
itself. Using the venv interpreter path is the activation. It sets the MLX default
device to the GPU, a 512 MB memory limit, and a 64 MB cache limit internally.
Copy `prepare-real-groups.py` from
`/media/dan/nvme/fifi/scrch/52640f0/task2-drafts/real-group/` to the Mac
first, and keep the hash above as the draft identity.

## Evidence file layout

The script creates the output directory fresh and refuses an existing
directory or symlink. The script writes each file once via exclusive create
and fsync.

- `<out>/real-groups.json` — the frozen fixture, schema
  `p0a05-real-group-1`, at most 4 MiB. Top-level blocks: `schema`,
  `selection` (algorithm, seed `p0a05-real-group-v1`, targets, phase A
  negative count), `provenance` (MLX version, Python version, platform,
  reader mode, binary hashes), `quantization` (default spec and the verbatim
  `config.json` quantization block), `tolerances` (`abs` 0.0, `rel` 0.0),
  `activations` (identity basis and dense pattern), `counts`, and `cells`.
- `<out>/manifest.json` — schema `p0a05-real-group-manifest-1`. Records the
  six input byte counts and hashes, `identity_recheck: passed`, reader mode,
  selection summary, counts, the sorted eligible module list, the excluded
  int8 module list, and the fixture byte count and SHA-256.
- `qualification/real-groups/real-groups.json` and
  `qualification/real-groups/manifest.json` — the committed copy of the
  frozen evidence. The portable tests read it when `REAL_GROUP_FIXTURES`
  is unset.
- Each of the 1,000 cells records: `id`, `module`, `payload_tensor`, shard,
  payload rank and shape, `groups_per_row`, `coordinates` (`expert` only for
  rank-3), `negative_bias`, dtypes (`U32` payload, `BF16` scale and bias),
  8 payload words as hex-free decimal `u32` values, scale and bias as
  `bf16_bits` plus exact `f32`, and `expected.basis` (64 values) and
  `expected.dense` (1 value) from the pinned MLX.

## Selection contract

Deterministic given the recorded seed. Module names sort by UTF-8 bytes. The
candidate chain digest is SHA-256 of `seed NUL module-name NUL decimal
ordinal`, first 8 bytes little-endian, modulo the module's group count. This
is the decision 0015 chain with the seed recorded in the fixture. Phase A
round-robins across modules accepting only groups with a negative bias, up
to 200. Phase B fills to exactly 1,000 from all groups. Ordinals never reset,
the script skips duplicates within a module, and each module tries at most 1,024
candidates per phase. The script fails if it finds no negative bias or if
the pool cannot supply 1,000 distinct groups.

Inventory guards: 1,757 index keys, 432 eligible affine-4 modules of which
120 are rank-3, 80 int8-overridden modules excluded by name in the manifest.
These counts come from `docs/decisions/0015-real-group-parity.md` and the
oq04 model-load review. Any deviation aborts with exit 2.

## What the coordinator must check on the Mac

1. Reader-mode probe output, and that `provenance.reader` in the fixture
   matches it.
2. `counts.negative_bias_groups` is at least 1 and equals
   `selection.phase_a_negative_groups`. Expect a value near 200. The gate
   does not assert an exact value.
3. `counts.matmul_calls` is 2,000 and `counts.compared_values` is 65,000.
4. The `run-2` determinism re-run prints the same `fixture_sha256`.
5. `manifest.inputs` hashes match the table above without any `mismatch`
   error, and `identity_recheck` is `passed`.
6. The Rust comparison from `rust-side-note.md` passes with zero value
   differences and a zero signed-zero bit-difference count.
