# Layer-parity fixtures

## Frozen bulk fixture

`fixtures.json` is frozen reference data and stays outside Git.

- sha256 `eec14e9b114d0fb08cfc0166f08a65fb706013efe5a5904cb5fcd1c21cb985e4`
- size 16793051 bytes
- schema `layer-parity-1`
- contents: two `qmm` cells, three `mm` cells, one `qmm` probe, tolerances, provenance

The preserved copy lives in the task evidence scratch under
`frozen-fixtures/fixtures.json`. Regenerate an identical file only with
`prepare.py` on the pinned Mac environment recorded in its provenance
block. `prepare.py` refuses to overwrite an existing file.

## Committed portable fixture

`engine/tests/fixtures/layer-parity-portable.json` carries the
`mm-tiny` cell and the `qmm-uniform-hand` probe copied verbatim from the
frozen fixture. It keeps ordinary portable tests offline and small. It
does not replace the frozen fixture for native qualification.

## Native qualification setup

1. Copy the frozen `fixtures.json` to any path on the Mac runner.
2. Export `LAYER_PARITY_FIXTURES=<that path>`.
3. Run the native layer-parity tests.

The tests verify the file hash against the pinned sha256 above and fail
when the file is missing or the hash differs. They never substitute new
reference values.

When the variable is unset, each of the five native tests prints a `SKIPPED:` line
that names the test, and the test passes. A skipped run is not gate evidence.
A gate run sets the variable, passes `--nocapture`, and must show the lines
`compared 3 frozen layer-parity qmm cells within tolerance` (two cells and one probe)
and `compared 3 frozen layer-parity mm cells within tolerance`, with no `SKIPPED` line.

## RMS normalization fixture

`rms-fixtures-final.json` is frozen reference data and stays outside Git.

- sha256 `60a5a22b832d30d910ffade0947930e830ada71350d8ee6dc278182d0089d976`
- schema `resident-rms-norm-1`
- contents: two cells, each with an `x` tensor, a `weight` tensor, `eps`, and the expected output as dtype, shape, and hex bytes

`prepare_rms_norm.py` generates it on the pinned Mac environment.
To run the native comparison:

1. Copy `rms-fixtures-final.json` to any path on the Mac runner.
2. Export `RMS_NORM_FIXTURES=<that path>`.
3. Run `cargo test --features native-mlx --test native_rms_norm -- --nocapture`.

When the variable is unset, the comparison test prints a `SKIPPED:` line that names the test, and the test passes.
A gate run must set the variable and must show the line `compared 2 frozen RMS normalization cells bit-exactly`.
The test fails when the file is unreadable, malformed, has another digest, or does not hold two cells.

## Elementwise fixture (S8)

`elementwise-fixtures.json` is frozen reference data and stays outside Git.

- sha256 `09641c6406a45c197bb5587d594afe55fc6c53bea715458f32f46bb0173bf287`
- size 3563 bytes
- schema `resident-elementwise-1`
- contents: nine cells for `silu`, `mul`, `add`, and `softmax`, with F32 shapes and uint32 bit patterns for inputs and outputs, plus provenance

The preserved copy lives in the task evidence scratch under
`impl-02/fixture/elementwise-fixtures.json`.
Generate it on the pinned Mac environment with `prepare_elementwise.py`.
The script takes `--output` and `--mlx-source` and prints the sha256.
It refuses to overwrite an existing file.
A repeated run on the same environment produces the same digest.

The reference for `silu` is `x * mx.sigmoid(x)`. This matches the Rust decomposition.
The compiled `mlx.nn.silu` differs from this decomposition in the last bits for some inputs.
The provenance field `nn_silu_bitwise_matches_decomposition` records that result.

To run the native comparison:

1. Copy `elementwise-fixtures.json` to any path on the Mac runner.
2. Export `ELEMENTWISE_FIXTURES=<that path>`.
3. Run `cargo test --features native-mlx --test native_elementwise -- --nocapture`.

When the variable is unset, the comparison test prints a `SKIPPED:` line that names the test, and the test passes.
A gate run must set the variable and must show the `compared 9 frozen elementwise cells` line.
The test fails when the file is unreadable, malformed, has another digest, or does not hold nine cells.

## RoPE fixture (S7)

`rope-fixtures.json` is frozen reference data and stays outside Git.

- sha256 `6a3eb4bcb15fa200123a7905f402fa4afc5dc807d22b9e0a041cb6848f363332`
- size 3390046 bytes
- schema `resident-rope-1`
- contents: 15 cells and provenance
- each cell holds a step (`pos`, `theta`, `part`), the F32 `q` and `k` input shapes with uint32 bit patterns, and the pinned GPU output
- the 8 formula-gating cells also hold the float64 rotation of each input as `formula`
- GPU: `applegpu_g16s` on an Apple M4 Max, from the `device` field of the generation run

The preserved copies live in the integration batch 1 evidence under
`/media/dan/nvme/fifi/scrch/1135a85/integration-batch1/mac/mac-evidence/s7-rope/rope-fixtures-1.json`
and `rope-fixtures-2.json`. Both have the digest above, because two generation runs on the pinned
Mac environment gave identical output.
Generate it with `prepare_rope.py`. The script takes `--output` and `--mlx-source` and prints the sha256.
It refuses to overwrite a file, and it stops unless the MLX version, the core extension hash, the
`mlx.metallib` hash and size, the GPU device, and the clean source checkout at the pinned commit all match.
It runs each GPU call twice and stops if the two results differ in any bit.

The reference is `mx.fast.rope(x, part, traditional=False, base=theta, scale=1.0, offset=pos)` on the GPU stream.
The CPU fallback uses different arithmetic and is not a reference.
The cells are `identity-pos0`, `pos1`, `pos63`, `pos64`, `pos65`, `pos262143`, `theta10000-pos1`,
`part2-pos3`, `part256-pos1`, `layout-rank3-t3-pos63`, `layout-rank4-t3-pos0`,
`layout-rank4-t3-pos63`, `layout-rank4-b2-t1-pos1`, `layout-rank4-b2-t2-pos1`, and `part256-rank4-t3-pos63`.
The S7 contract lists the shapes. Inputs lie on the exact 1/16 grid, so the Linux script can regenerate them.

Two gates apply to each cell.

- Bit equality. Every output of the Rust call must equal the frozen GPU output bit for bit. No epsilon applies.
- Formula bound. The cells whose largest angle stays at or below pi, which are the cells with `pos + T - 1` at most 3, must also satisfy
  `|actual - reference| <= (|x1| + |x2|) * (2^-13 + 2^-16)` for each rotated pair. The reference is the float64 rotation.
  The test recomputes the gating set from each step and hard-codes the bound. It reads neither from the fixture flags.
  Lanes from `part` on must equal the input bit for bit. Eight cells gate on the formula.
  The other cells report their deviation and never fail on it, because the `fast::sin` and `fast::cos` accuracy has no published bound above pi.

To run the native comparison:

1. Copy `rope-fixtures.json` to any path on the Mac runner.
2. Export `ROPE_FIXTURES=<that path>`.
3. Run `cargo test --features native-mlx --test native_rope -- --nocapture`.

When the variable is unset, the comparison test prints a `SKIPPED:` line that names the test, and the test passes.
A gate run must set the variable and must show the lines `checked the formula bound on 8 gating cells` and `compared 15 frozen rope cells bit-exactly`.
The test fails when the file is unreadable, malformed, has another digest or size, has another schema, or does not hold 15 cells.
A gate run also records the SHA-256 of the metallib that the Rust `mlx-sys` build compiled, beside the test output.
The Python reference and the Rust build compile `rope.metal` separately from the same MLX commit, so the two metallibs can differ.

Observation from the generation run: the `identity-pos0` output equals its input bit for bit.
The hand test `rope_at_position_zero_leaves_nonzero_inputs_unchanged` relies on this, for nonzero finite inputs only.
The Metal specification does not promise `fast::cos(0) = 1`.
