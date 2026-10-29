# Test fixtures

The tests in `engine/tests/`, and some unit tests in `engine/src/`, read these small files.
They need no model download, network access, or Python installation.
Expected values come from the tools named below (Python `safetensors`, MLX, llama.cpp) or from hand calculation, not from the Rust code under test.
The scripts that produced the files are not included in this repository.

## Portable loader fixtures

The official Python `safetensors` 0.8.0 package created `scalar.safetensors`, `mixed.safetensors`, and `packed.safetensors`.
Their manifests (`scalar.json`, `mixed.json`, `packed.json`) record input shapes, dtypes, payload bytes, and SHA-256 hashes.
`scalar.safetensors` uses `safetensors.numpy.save` with NumPy `2.5.3`.
`mixed.safetensors` uses `safetensors.serialize` with the `TensorSpec` interface.
The producer kept every input buffer alive throughout serialization.
The malformed-file tests construct deliberately invalid headers directly.
They do not call the loader to generate expected results.

`packed.safetensors` uses the same `TensorSpec` serializer.
It contains a `[2, 8]` U32 payload and `[2, 1]` BF16 scales and additive biases.
Its manifest records affine int4, group size `64`, and logical shape `[2, 64]`.
The packed constructor tests read this fixture without calling MLX.

## oQ metadata fixture

`oq-metadata.json` keeps the defaults and four overrides from the quantization settings in the `config.json` of the Qwen3.8 Flash-Next oQ checkpoint.
`oq-metadata-manifest.json` records the source repository and revision, the hash of the original file, the selected paths, and the fixture hash.
The expected settings come from that file, not from the Rust parser.
This fixture checks metadata resolution only. It does not establish packed tensor validity or model execution.

## Expanded packed-weight fixture

`oq-packed.safetensors` contains ten small storage examples from `mlx.core.quantize` on a Metal GPU, saved with `mlx.core.save_safetensors`.
`oq-packed-manifest.json` records MLX 0.30.6, input values, observed layouts, sample payload words, and artifact hashes.
The examples cover all six affine widths, all three group sizes, F16, BF16, F32, and both MXFP modes.
MXFP scales keep their encoded U8 bytes. Affine examples include negative scales.
The constructor tests check storage and owner retention. These fixtures do not establish Rust unpacking or native numerical parity.

## Family layer-weight fixtures

`family-full.safetensors` holds both layer kinds of a two-layer model family with mixed quantization.
`family-missing-router.safetensors` omits one router weight, `family-wrong-rank.safetensors` stores it with an extra rank,
and `family-partial.safetensors` holds one dense norm only.
The geometry is hidden `128`, four query heads, two key-value heads, head dimension `16`, four experts,
expert and shared intermediate `64`, and auxiliary width `64`.
The default quantization is affine 4-bit, group size `32`.
Layer `0` overrides the router, the shared-expert gate, one expert `gate_proj`, and `q_proj` to 8/64, 8/64, 4/128, and 8/32 (bits/group size).
The producer writes the safetensors header and payload bytes directly with Python's `struct` and `json` modules, from the recorded manifest values.
`family-fixtures.json` records schema `family-fixtures/2`, the geometry, the quantization configuration,
per-tensor shapes and fill values, and artifact hashes.
Each `*.index.json` file is the matching weight map.
The tests derive expected shapes, widths, and metadata by hand from these manifest values, not from the Rust plan or resolver.

Four dtype-rejection fixtures keep the full geometry, names, shapes, and quantization
of `family-full.safetensors` while changing exactly one declared storage dtype:
`family-bad-payload-dtype.safetensors` declares the layer-0 router payload `F32`,
`family-bad-expert-dtype.safetensors` declares the expert `gate_proj` payload `F32`,
`family-bad-sidecar-dtype.safetensors` declares the router scales `F16` against `BF16` biases,
and `family-bad-dense-dtype.safetensors` declares `input_layernorm.weight` `U32`.
The same producer writes them with the recorded manifest entries and hashes.
The official Python `safetensors` package reads all four files.
They exist to test that the resolver rejects the wrong dtypes. No engine code generated them.

## DeltaNet ratio fixture

`family-delta-ratio.safetensors` holds one DeltaNet layer whose linear-attention sizes are not the old multiples of the hidden width.
The geometry is hidden `128`, two key heads of width `8`, four value heads of width `16`, and convolution kernel `3`.
Worked by hand from the mlx-lm `GatedDeltaNet` constructor, the key width is `16`, the value width is `64`, and the fused width is `96`.
Every count differs, so a swapped head count and head width shows.
The old rules expected `512` fused rows and `256` gate rows. The fixture stores `96` and `64`.
The default quantization is affine 4-bit, group size `32`, with no overrides.
`family-delta-ratio.json` records schema `family-delta-ratio/1`, the geometry, per-tensor shapes and fill values, and artifact hashes.
`family-delta-ratio.index.json` is the matching weight map.
The Python `safetensors` package, version `0.7.0`, reads the file and lists all 45 tensors.
The writer is the same as for the family fixtures above. No engine code generated these files.

## Other fixtures

- `f8/`: hand-written E4M3FN weight codes and single F32 scale values (weight scales 2, 0.25, and 448; input scale 7) for `engine/tests/f8_weight.rs`.
- `nvfp4/`: hand-written NVFP4 packed values and E4M3 scale bytes, with small `config.json`, `hf_quant_config.json`, and `model.safetensors.index.json` files in the ModelOpt mixed FP8/NVFP4 metadata format, for `engine/tests/nvfp4.rs`.
- `gguf-dequant/`: GGUF block inputs (`*.blocks`) and expected F32 outputs (`*.f32`) for 15 GGUF types, two scale-coverage cases, and 15 blocks sampled from two real GGUF files. The expected outputs come from the llama.cpp reference dequantization (`ggml-quants.c` at commit `fb4b2737a808a3fb7c2117a498f43815dc9be53e`). `manifest.json` records the source hashes, compiler commands, seed, and per-file hashes. Used by `engine/tests/gguf_dequant.rs` and `engine/src/weights/gguf_weights.rs`.
- `gguf-nvfp4/`: NVFP4 GGUF blocks, scales, and expected outputs: one synthetic case that covers every scale byte with every E2M1 code, and three cases sampled from `RecursiveSI/flint-nvfp4-gguf` (Apache-2.0). The expected outputs follow the llama.cpp Python reference `gguf-py/gguf/quants.py`. `manifest.json` records the model revision, the reference commits, and per-file hashes. Used by `engine/tests/gguf_nvfp4.rs`.
- `nemotron35/`: configuration, index, and shard-header inputs for the Nemotron 3.5 tests. `PROVENANCE.md` and `tokenizer-notes.md` in that directory describe them.
- `qwen3_6_35b_a3b_config.json`: the `config.json` of a 4-bit MLX conversion of `Qwen/Qwen3.6-35B-A3B`. Several tests pin its SHA-256, for example `engine/tests/family_spec.rs`.
- `qwen3_6_35b_a3b_shard_headers/`: the 8-byte length prefix and the JSON header of each of the four safetensors shards of that checkpoint, without tensor data. Used by `engine/tests/q36_real_config.rs`.
- `qwen3_8_flash_next_config.json`: `config.json` values of the Qwen3.8 Flash-Next oQ checkpoint, for `engine/tests/flash_next_family.rs`, which pins its SHA-256.
- `layer-parity-portable.json`: one `mm` cell and one `qmm` probe with MLX reference outputs (`mx.matmul`, `mx.quantized_matmul`) and zero tolerances, copied unchanged from a larger layer-parity fixture that is not in this repository. Used by `engine/tests/layer_parity.rs`.
- `resident-rows.json`: four row-broadcast and weighted-row-sum cells with bit-exact MLX GPU outputs, the MLX version, and the generator hash. `engine/tests/native_rows.rs` reads it when `ROWS_FIXTURES` names its path, and checks its SHA-256 first.
