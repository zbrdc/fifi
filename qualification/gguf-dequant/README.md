# GGUF dequantizer oracle and fixtures

This directory builds the reference output for `engine/src/weights/gguf_dequant.rs`.
The oracle is the ggml CPU dequantizer from llama.cpp commit
`fb4b2737a808a3fb7c2117a498f43815dc9be53e`. The upstream files stay outside the repository.
`oracle.c` is original project code that calls `ggml-quants.c`.

## Upstream files (12, all at fb4b2737)

Paths are relative to the llama.cpp root. Their sha256 values are in `manifest.json`
(`oracle_files_sha256`). The first ten are included by the oracle build (`clang -MM`).
The last two are recorded for provenance only.

1. `ggml/src/ggml-common.h`
2. `ggml/src/ggml-quants.c`
3. `ggml/src/ggml-quants.h`
4. `ggml/src/ggml-impl.h`
5. `ggml/include/ggml.h`
6. `ggml/include/ggml-backend.h`
7. `ggml/include/gguf.h`
8. `ggml/include/ggml-cpu.h`
9. `ggml/include/ggml-alloc.h`
10. `ggml/src/ggml-cpu/ggml-cpu-impl.h`
11. `ggml/src/ggml-cpu/quants.h` (not included by the build)
12. `LICENSE`

## Run order

1. `gen_tables.py <ggml-common.h> engine/src/weights/gguf_tables.rs`, then `cargo fmt`.
2. `extract_real.py <shard.gguf> <real-dir>` reads a GGUF shard (read only) and writes a few
   blocks per type to `real-<type>.blocks` plus `real-index.json` with the shard sha256.
3. `build_oracle.sh <oracle-src> <work> <seed> <real-dir>` builds the oracle twice
   (`-ffp-contract=off` and `=on`), writes synthetic fixtures, and converts the real blocks.
   It records `clang --version`, the target triple and the build commands in `<work>/build-info.txt`.
4. `make_manifest.py <work> <fixture-dir> <seed> <oracle-src> <real-dir>` copies the
   `-ffp-contract=off` outputs into `engine/tests/fixtures/gguf-dequant/` and writes `manifest.json`
   with provenance, per-file sha256, an aggregate fixture sha256 and the count of values that
   differ in the contracted build.

The fixtures come from seed 20261002 on Apple clang 21.0.0, target arm64-apple-darwin25.5.0.
The `q8_0` and `real-q8_0` sets (packet 4) come from the same seed on Ubuntu clang 20.1.8,
target x86_64-pc-linux-gnu. Their manifest entries record the host, OS and compiler. That x86
build reproduced the other 14 `.blocks` files byte for byte, and their `.f32` files differed only
in NaN sign, so the committed arm64 files stay unchanged and only the aggregate sha256 is new.
Do not run `make_manifest.py` into the fixture directory on x86: it would overwrite them.

## Fixture sets

- `types`: 15 synthetic sets (edge blocks, random blocks, sweeps covering every grid and sign index).
  The `q8_0` set skips edge blocks with a NaN `d`, which the CUDA path rejects before upload.
  Its one NaN output (an infinite `d` times a zero quant) is stored as `0x7fc00000`, the arm64 value.
- `extra`: `q4_k_scales` and `q5_k_scales`, 64 blocks each. Every 6-bit scale and min value
  (0..63) occurs in all 8 slots.
- `real`: a few blocks per type taken from one tensor of the IQ3_XXS GGUF
  (`Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf`, shard 1). The manifest records the
  tensor name, shard sha256, tensor offset, start block and absolute file offset. All 14 types
  occur in shard 1. No type is absent. `real-q8_0` holds 32 blocks (1,088 B) of
  `blk.0.attn_gate.weight` from `Qwen3.6-35B-A3B-Q4_K_M.gguf` (Apache-2.0), because the IQ3_XXS
  shard has no Q8_0 tensor.

## Oracle build effects

- The NaN sign comes from `clang -O2`, which folds the multiply by `-1.0` into a negation.
  That flips the NaN sign bit. At `-O0` the sign is kept. The Rust rule follows the pinned
  oracle build, not IEEE or ggml semantics.
- Fixtures use `-ffp-contract=off`. A contracted build (`=on`) differs in 512 values each
  for Q4_K and Q5_K in the synthetic sets. Later kernels must not contract `d * q - m`.
- On x86 the default NaN has sign 1. The test accepts exactly got `0xffc00000` where the oracle
  has `0x7fc00000` off aarch64. Bit-exact gating happens on arm64 (the Mac).
