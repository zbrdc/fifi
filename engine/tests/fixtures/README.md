# Portable loader fixtures

The official Python safetensors `0.8.0` package created these files on the Mac.
The fixture manifests record input shapes, dtypes, payload bytes, and SHA-256 hashes.
`scalar.safetensors` uses `safetensors.numpy.save` with NumPy `2.5.3`.
`mixed.safetensors` uses `safetensors.serialize` with the installed `TensorSpec` interface.
The producer kept every input buffer alive throughout serialization.
The committed fixtures are independent of the Rust loader implementation.
Ordinary tests read these local files and require no model download or Python installation.
The malformed-file tests construct deliberately invalid headers directly.
They do not call the loader to generate expected results.

Producer and raw evidence: `/media/dan/nvme/fifi/scrch/751440c/phase0a-portable-loader/`.
The arithmetic record (portable-loader-math.json, in the ledger) checks overflow and byte-count examples.

`packed.safetensors` uses the same official `TensorSpec` serializer.
It contains a `[2, 8]` U32 payload and `[2, 1]` BF16 scales and additive biases.
Its manifest records affine int4, group size `64`, and logical shape `[2, 64]`.
Producer: `/media/dan/nvme/fifi/scrch/751440c/phase0a-packed-weights/make-packed-fixture.py`.
The packed constructor tests read this independent fixture without calling MLX.

## oQ metadata fixture

`oq-metadata.json` retains defaults and four overrides from the pinned Flash-Next oQ candidate configuration.
The manifest records the source revision, original hash, selected paths, and fixture hash.
The expected settings come from the producer file. The Rust parser did not generate them.
This fixture checks metadata resolution only. It does not establish packed tensor validity or model execution.
The source inventory (oq-sources.json, in the ledger) preserves the original download URL.

## Expanded packed-weight fixture

`oq-packed.safetensors` contains ten small storage examples from `mlx.core.quantize` on the Mac Metal GPU.
`oq-packed-manifest.json` records MLX 0.30.6, input values, observed layouts, sample payload words, and artifact hashes.
The producer uses `mlx.core.save_safetensors`. It does not call the Rust implementation.
The examples cover all six affine widths, all three group sizes, F16, BF16, F32, and both MXFP modes.
MXFP scales retain their encoded U8 bytes. Affine examples include negative scales.
The constructor checks storage and retains owners. These fixtures do not establish Rust unpacking or native numerical parity.
Producer script: `/media/dan/nvme/fifi/scrch/567b92b/oq-weights/make-reference-fixtures.py`.
Arithmetic evidence (oq-weights-math.json, in the ledger) records the additional size and overflow checks.

## Family layer-weight fixtures

`family-full.safetensors` holds both layer kinds of a two-layer family with mixed quantization.
`family-missing-router.safetensors` omits one router weight, `family-wrong-rank.safetensors` stores it with an extra rank,
and `family-partial.safetensors` holds one dense norm only.
The geometry is hidden `128`, four query heads, two key-value heads, head dimension `16`, four experts,
expert and shared intermediate `64`, and auxiliary width `64`.
The default quantization is affine 4-bit, group size `32`.
Layer `0` overrides the router, the shared-expert gate, one expert `gate_proj`, and `q_proj` to 8/64, 8/64, 4/128, and 8/32.
The local Linux `safetensors` Python package lacked the `TensorSpec` serializer used by earlier fixtures,
so the producer writes the safetensors header and payload bytes manually from the recorded manifest values.
`family-fixtures.json` records schema `family-fixtures/2`, the geometry, the quantization configuration,
per-tensor shapes and fill values, and artifact hashes.
The tests derive expected shapes, widths, and metadata by hand from these manifest values, not from the Rust plan or resolver.
The replaced fixture files are preserved with hashes under `evidence/original-fixtures/` in the task scratch.
Producer script: `/media/dan/nvme/fifi/scrch/80b60ed/p1-weight-handoff/make-family-fixtures.py`.

Four dtype-rejection fixtures keep the full geometry, names, shapes, and quantization
of `family-full.safetensors` while changing exactly one declared storage dtype:
`family-bad-payload-dtype.safetensors` declares the layer-0 router payload `F32`,
`family-bad-expert-dtype.safetensors` declares the expert `gate_proj` payload `F32`,
`family-bad-sidecar-dtype.safetensors` declares the router scales `F16` against `BF16` biases,
and `family-bad-dense-dtype.safetensors` declares `input_layernorm.weight` `U32`.
The same producer emits them with the recorded manifest entries and hashes.
Official Python `safetensors` deserialization accepts all four containers.
They exist to prove resolver dtype rejection; no engine code generated them.

## DeltaNet ratio fixture

`family-delta-ratio.safetensors` holds one DeltaNet layer whose linear-attention counts break the old multiples of the hidden width.
The geometry is hidden `128`, two key heads of width `8`, four value heads of width `16`, and convolution kernel `3`.
By hand from the mlx-lm `GatedDeltaNet` constructor, the key width is `16`, the value width is `64`, and the fused width is `96`.
Every count differs, so a swapped head count and head width shows.
The old rules expected `512` fused rows and `256` gate rows. The fixture stores `96` and `64`.
The default quantization is affine 4-bit, group size `32`, with no overrides.
`family-delta-ratio.json` records schema `family-delta-ratio/1`, the geometry, per-tensor shapes and fill values, and artifact hashes.
`family-delta-ratio.index.json` is the matching weight map.
The local Python `safetensors` package, version `0.7.0`, deserializes the container and lists all 45 tensors.
The manual writer follows the `family-fixtures` producer. No engine code generated these files.
Producer script: `/media/dan/nvme/fifi/scrch/3a141a6/p1-family-fixes/producer/make-family-delta-ratio-fixture.py`.
