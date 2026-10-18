# Rust comparison side for the real-group parity fixture

Design note for turning `real-groups.json` into a passing engine
qualification at base `52640f0`. Commit `290b178` implements this slice.
This note records the design.

## Where it lives

Add `engine/tests/real_group_parity.rs` following the frozen-fixture pattern
of `engine/tests/parity_matrix.rs`. Prefer a test file over a separate
qualification crate: the comparison is one bounded pass over a fixture, the
pattern already exists, and a crate adds a second harness for no new
behavior. The `qualification/native-parity/` crate shape with `Suite::load`
and exit codes remains the fallback if the coordinator wants the runner
outside `cargo test`, for example to gate on the manifest digest. That crate
already implements the decision 0015 exit convention: 0 complete and zero
differences, 1 mismatch, 2 invalid inputs or backend failure.

## Pattern to copy from `engine/tests/parity_matrix.rs`

Line references are from the `52640f0` checkout:

- `fixture_path` (line 5): environment override for the fixture location.
  Copy as `REAL_GROUP_FIXTURES`, defaulting to the evidence directory
  layout, one file `real-groups.json`.
- `load_fixture` (line 14): read, assert top-level `schema` equals
  `p0a05-real-group-1`, and parse with serde_json. Duplicate-field
  rejection: the prepare script already enforces it at freeze time. The
  fixture is evidence. The test parses what the manifest hash
  pins.
- `tolerance` (line 26) and `compare` (line 35): copy the shape but hard-code
  zero tolerance. The fixture carries `tolerances.real-group = {abs: 0.0,
  rel: 0.0}`. Assert those fields rather than accepting a looser value from
  the file. The compare function must treat any value difference as a
  failure and must count `+0.0` versus `-0.0` bit differences separately
  without failing, matching decision 0015.
- `GPU_TEST` mutex (line 131): reuse the same static so native tests
  serialize with the existing parity suite.
- `exact_f32` (line 155): the comparison operator. Expected and actual must
  be bit-identical except signed zeros, using `to_bits()` for the recorded
  difference details.
- `exact_bf16` (line 176) and `float_tensor` (line 187): reuse for scale and
  bias reconstruction (below).
- `packed_weight` builder (line 233): the per-cell `PackedWeight`
  construction entry point.
- `output_values` (line 269) and `run_cell` (line 278): the per-cell driver
  shape. Each real group becomes one cell.
- `write_report_if_requested` (line 336): copy as `REAL_GROUP_REPORT` for
  the run record, temp file plus atomic rename, so a failed run leaves a
  machine-readable difference list.

Gate the test behind the same conditions as the native tests: `cfg(all(test,
feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))`. On
Linux or without the feature the test compiles away, so the change leaves
the portable suite unchanged. Run `cargo check --tests --features native-mlx` on the Mac
before the qualification round. That pre-gate caught issues in the oq04
round recorded in `oq04-parity-matrix.md`.

## Feeding one fixture cell through the engine

Each cell names one group: module, optional expert coordinate, row, and
group index. The engine side rebuilds a one-row packed weight:

1. Payload: `Tensor` of dtype `U32`, shape `[1, 8]`, from the cell's 8
   words. Logical shape `[1, 64]`.
2. Scales and biases: dtype `Bf16`, shape `[1, 1]`, built from
   `weight.scales.bf16_bits` and `biases.bf16_bits`. Reconstruct with
   `half::bf16::from_bits`, assert `to_f32()` equals the fixture's recorded
   `f32` exactly, mirroring `exact_bf16`. This also guards the bf16-to-f32
   widening that `Prepared` performs before the C call (engine/src/kernels/mlx/mod.rs,
   `float_values` conversion in `Prepared::new`).
3. `PackedWeight::try_new` with `bits 4`, `group_size 64`, mode affine.
   `engine/src/kernels/packed.rs` checks exactly this storage: affine
   bits 4 accepted at line 25, group size 64, payload dtype `U32` at line
   64, scales `F16|Bf16|F32` at line 89, biases required and dtype-matched
   at line 97, and sidecar shape equal to logical leading dims plus
   `width / group` at line 116.
4. Activations: the fixture's `activations.basis` (identity `[64, 64]`) and
   `activations.dense` (`[1, 64]`), rebuilt as `F32` tensors once. Assert
   they match the fixture bytes, then reuse across cells.
5. Call `MlxKernels::dequant_matmul` (engine/src/kernels/mlx/mod.rs line 25)
   twice per cell. Outputs: `basis` gives `[64, 1]`, `dense` gives `[1, 1]`.
   Compare every value with the zero-tolerance `compare`. Record
   `expected`, `actual`, and both `to_bits()` for any difference.

## Rank-3 expert groups

The fixture keeps the expert coordinate in the cell identity, but the
comparison feeds each group as a rank-2 `[1, 64]` logical row. This is
required: `dequant_matmul` goes through `Prepared::new`, and `matrix_shape`
(engine/src/kernels/mlx/mod.rs line 503) rejects any logical shape that is
not exactly rank two, at line 505, with `native matmul requires rank two`.
`PackedWeight::try_new` itself accepts rank 3 and above. Only the matmul
preparation restricts the rank. Slicing one expert row is lossless for this
gate because each group's dequantization and its matmul row are independent
of the expert axis. The oq04 routed cells already exercised whole rank-3
payloads through `gather_dequant_matmul` (line 40, `SelectionPrepared`),
and that path stays out of scope here. The test asserts the recorded
`payload_shape` has rank 2 or 3 and drops the expert axis only after
selecting the recorded row, so a fixture regression that changes group
coordinates cannot pass silently.

## Quantization metadata check

The fixture embeds the checkpoint's verbatim `config.json` quantization
block under `quantization.config`. The test parses it with
`QuantizationMap::from_json` (engine/src/family/quantization.rs line 24)
and asserts `resolve` returns the affine-4/64 spec for every sampled module
and the affine-8 override for every module the manifest lists as excluded.
`from_json` rejects `quant_method` entries and conflicting
`quantization`/`quantization_config` keys, and `resolve` (line 88) applies
per-module overrides, so this catches a fixture frozen from the wrong
checkpoint family or a module list that drifts from the pins.

## Totals and the gate

Per cell: 2 matmul calls, 65 compared values. For 1,000 cells: 2,000 calls,
65,000 values, all from the real checkpoint with 4-bit affine quantization
and negative biases included by construction. The gate from decision 0015:
zero `f32` value differences, signed-zero bit differences counted and
reported, any difference or nonfinite value fails. The earlier real-group
qualification in the ledger (task series before base `52640f0`) already
passed this shape of gate against an older reference environment. This kit
re-qualifies it against the pinned parity venv, so prior passes are not
evidence here.

## Cost estimate for the Mac run

Not run. Sizing basis only: the oq04 matrix ran 31 cells with the same
per-cell pattern. This is 1,000 cells of the same shape plus one fixture
read of roughly 4 MiB. Expected duration is minutes, not hours, dominated
by test startup and 2,000 GPU calls. Record the real duration in the
evidence review.
