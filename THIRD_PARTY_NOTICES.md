# Third-party notices

## llama.cpp / ggml

- Source: https://github.com/ggml-org/llama.cpp, commit `fb4b2737a808a3fb7c2117a498f43815dc9be53e`.
- File: `ggml/src/ggml-common.h`, sha256 `0061131b615c5721fc88a78feeb22c1f8c450f1c2646a317d80796a653bf595c`.
- Taken: eight constant tables, as data only: `kmask_iq2xs`, `ksigns_iq2xs`, `iq2xxs_grid`, `iq2xs_grid`, `iq2s_grid`, `iq3xxs_grid`, `iq3s_grid`, `kvalues_iq4nl`.
- Location: `engine/src/weights/gguf_tables.rs`. The tables were extracted from `ggml/src/ggml-common.h` at the commit above. The extraction script is not included in this repository.
- Also used as a test oracle only (not shipped, not copied): the expected outputs in `engine/tests/fixtures/gguf-dequant/` come from the reference dequantization functions of `ggml/src/ggml-quants.c` at the same commit, compiled into a small driver program.
  Neither `ggml-quants.c` nor the driver program is included in this repository. `engine/tests/fixtures/gguf-dequant/manifest.json` records the hashes of the llama.cpp source files and the compiler commands.

License text (`LICENSE` at that commit, sha256 `94f29bbed6a22c35b992c5c6ebf0e7c92f13b836b90f36f461c9cf2f0f1d010d`):

```
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Functions

- Source: https://github.com/ggml-org/llama.cpp, commit `acecd56032ddc34bada14a2d978f110d9c987095`.
- License: MIT, `LICENSE` at that commit, sha256 `94f29bbed6a22c35b992c5c6ebf0e7c92f13b836b90f36f461c9cf2f0f1d010d`, "Copyright (c) 2023-2026 The ggml authors". The license text is the one above.
- Derived: the functions below, in `engine/vendor/xavier-qwen/src/`. Each source comment names the ggml function it follows.
  - `Q4_K::scale_min` in `engine/vendor/xavier-qwen/src/mma884.cuh:253`, from ggml `get_scale_min_k4`. It is called by `DQ4K::dq` in `engine/vendor/xavier-qwen/src/gemm.cuh:64-69`.
  - `scale_min_k4` in `engine/vendor/xavier-qwen/src/repack.cu:110`, from ggml `get_scale_min_k4`.
  - `q6` in `engine/vendor/xavier-qwen/src/repack.cu:46`, from ggml `dequantize_row_q6_K`.
  - `quantize_q8_0` in `engine/vendor/xavier-qwen/src/repack.cu:91`, from ggml `quantize_row_q8_0_ref`.
  - `dequant_q5k` in `engine/vendor/xavier-qwen/src/repack.cu:121`, from ggml `dequantize_row_q5_K`.
- Derived, without a ggml comment in the source: the Q8_0 path. Each item follows the ggml `block_q8_0` layout (fp16 `d`, then 32 int8 `qs`) or its value rule `d * q`.
  - `kQ8Block` in `engine/vendor/xavier-qwen/src/repack.cu:16`, from ggml `block_q8_0`.
  - `pack_q8_0` in `engine/vendor/xavier-qwen/src/repack.cu:32`, from ggml `block_q8_0`. It moves `d` and `qs` bytes and computes no values.
  - `Q8_0::dq` in `engine/vendor/xavier-qwen/src/mma884.cuh:312`, from ggml `dequantize_row_q8_0`.
  - `DQ8::dq` in `engine/vendor/xavier-qwen/src/gemm.cuh:102`, from ggml `dequantize_row_q8_0`. The gemm path runs it for more than 16 tokens.
- Derived, data only, outside the xavier-qwen tree: the J/config tables in
  `engine/src/kernels/cuda/prefill/mmq_size.rs`, from ggml
  `ggml-cuda/mmq-config-ampere.cuh` and `ggml-cuda/mmq-config-pascal-dp4a.cuh` at the same
  pin (MIT-derived constants, data only; a test re-parses both vendored headers and
  hash-pins the vendored J-selection loop).
- Derived, logic only, outside the xavier-qwen tree: `fixup_need` in
  `engine/src/kernels/cuda/native/prefill/mmq_dense.cu`, which re-expresses the tile-width
  selection loop of ggml `ggml-cuda/mmq.cuh` (`mul_mat_q_switch_J`, vendored mmq.cuh:1444-1461)
  and the stream-k block and fixup-size rule of `launch_mul_mat_q` (vendored
  mmq.cuh:1395-1412) at the same pin, on top of the vendored config helpers. A test hash-pins both
  passages, and a sweep cross-checks the function against the Rust sizing.
- Not derived: `dequant_256` in `engine/vendor/xavier-qwen/src/repack.cu:140` repeats no ggml Q4_K or Q6_K value logic. Its Q5_K branch calls `dequant_q5k`, listed above, and its other branches widen F32 and F16 values directly.

## ggml-cuda-mmq (llama.cpp CUDA MMQ and MMVQ kernels)

- Source: https://github.com/ggml-org/llama.cpp, commit `acecd56032ddc34bada14a2d978f110d9c987095`.
- Location: `engine/vendor/ggml-cuda-mmq/`. It holds the include closure of the MMQ instances `q4_k`, `q5_0`, `q6_k`, and `q8_0`, of `quantize.cu`, `mmid.cu`, and the MMVQ files `mmvq.cu` and `mmvq.cuh`. These files were imported byte for byte from the commit above and then patched.
- Changes: the eight patches in `engine/vendor/ggml-cuda-mmq/patches/` change the imported files:
  - `V-2-errors.patch`: return errors instead of ending the process.
  - `V-3a-backend-machinery.patch`: remove the ggml backend machinery that no kept source uses.
  - `V-3b-stream-fixup-device-guard.patch`: pass the stream, the fixup range, and the device facts as arguments, and check the device before every launch.
  - `V-3c-pdl-off.patch`: turn programmatic dependent launch off.
  - `V-4-trim.patch`: trim to the tile configurations for sm_72 and sm_86. Ten whole files (other tile configurations and the HIP and MUSA vendor headers) are removed.
  - `V-5-q4k-tiling-configs.patch`: make the Q4_K configurations for J 8 to 64 tiling configurations instead of stream-k.
  - `V-MV1-mmvq.patch`: trim MMVQ to a device-only form that returns errors.
  - `V-PACK1-reference-packing.patch`: add an opt-in Q8_0 packing policy that matches the CPU reference. The default behavior of the vendor entry points does not change.
- `engine/vendor/ggml-cuda-mmq/provenance.json` records, for every imported file, the upstream SHA-256 and the SHA-256 after the last patch. It also records the patch chain and the hashes of the removed files.
- Written for this tree: `src/ggml-cuda/template-instances/mmq-instance-nvfp4.cu`. It is a two-line file in the form of the upstream instance files that instantiates the vendored MMQ template for `GGML_TYPE_NVFP4`. `engine/build.rs` compiles it. It is not an upstream file and `provenance.json` does not list it.
- Shim headers: `engine/src/kernels/cuda/native/prefill/ggml_shim/` holds `ggml.h`, `ggml-impl.h`, `ggml-cuda.h`, and `unary.cuh`. Each one stands in for the llama.cpp header of the same name, so that the vendored sources compile without the rest of ggml. Most of their text is written for this tree. Lines marked `derived` are copied from the matching llama.cpp header at the commit above (MIT). `provenance.json` lists each shim with its SHA-256 and the SHA-256 of the upstream header.
- License: MIT, Copyright (c) 2023-2026 The ggml authors. The license text is the one quoted in the section "llama.cpp / ggml" above. The file `engine/vendor/ggml-cuda-mmq/LICENSE` carries it (sha256 `94f29bbed6a22c35b992c5c6ebf0e7c92f13b836b90f36f461c9cf2f0f1d010d`).

## xavier-qwen

- Source: the `xavier-qwen` repository, exported with `git archive`. `engine/vendor/xavier-qwen/provenance.json` records no public upstream URL.
- Commit: `2f9183c1d99b7f459316f89b7dd9450ac7e8df66`.
- License: MIT, Copyright (c) 2026 Dan Campos. License text, from `engine/vendor/xavier-qwen/LICENSE`:

```
MIT License

Copyright (c) 2026 Dan Campos

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- Files: every file under `engine/vendor/xavier-qwen/`, with its current SHA-256 (after the patches, where they apply). `provenance.json` records the upstream and pre-patch hashes and the trim and patch details for the `src/` files.

| File | SHA-256 |
|---|---|
| `LICENSE` | `56dc032b18ae6211f3a3589f625e362118f81b90e79c038af865829b9def1e74` |
| `check_patches.py` | `32ac1faa954f5a4e3966c1327ab3f2ea9de227b0a6103578941268f4734a6ea6` |
| `patches/KV8-B-attn-prep-rows.patch` | `c1104535893522a018b4815a78bc0e638b6d81164253529695e3b30d324d3c4d` |
| `patches/MEMSET-TRIM-xf-zero-tail.patch` | `c2bf9e1d840dfe5295a8382dc193c53e1a472652277a418b110cd3ac4b29ad82` |
| `patches/RD-0-explicit-stream.patch` | `2f0b5543b42c4fd1cb98865c8447078ca9dd42b1ac5e8762e584de81f4fc752d` |
| `patches/RD-0b-expert-index.patch` | `17a322f9196337d094b00d51bc692d9e7c16f939a5090c97e2d6b7e0d55511c4` |
| `patches/RD-4V-attn-prep-position.patch` | `de0dd4e6a8607b817cff585f628b3ca56762b69a881e63cbc1944b52cf40f4a1` |
| `patches/RD-4V-frag-layout.patch` | `e217615ce2c26f5e98f5512ad1fc1fd051724a35086e8845c8b2a4cf8f396a13` |
| `patches/RD-4V-norm-out.patch` | `bf673a95dab1b4b50bc6aac943e75d9078629256af3a15eb5e9aaa4ee0551cd3` |
| `patches/RD-4V-stream-and-errors.patch` | `872ea6326e91176d275bd399afac899c01d8394e89bf16fe92e06c215f1fadff` |
| `patches/RD-8a-attn-prep-poison.patch` | `c40f399614d261f8bd3c1692ca4d4f8bce538287e33400528096e2ff6ccd26d9` |
| `provenance.json` | `8b6c5750efdbcf99849263ac56d78aab44dfec4c6adb7a955e9fd951ab775f8a` |
| `src/frag_layout.cuh` | `b566a419e9896c30251116d6ff6613680f8cf9eff304cbddd81e9d14c86fd1bc` |
| `src/gemm.cuh` | `0ef3adf1a5a6be9893f900be1f0d54daa5b351a1e7c86f7e93537ca17dd58f6c` |
| `src/kernels.cu` | `b86627a78f879ae19de7020b9b4813ba33f570ea12d67c2d34c15d794b6083ee` |
| `src/kernels.h` | `6ca18bbc586cd02adf9c4103e87bade584159bb20be6300f657debed0b71f643` |
| `src/mma884.cuh` | `506043e1c8a2a32b1feea806107acf5c1b0545bea342f27ec02a037e53d8bfae` |
| `src/ops.cu` | `12ed2ca756a03886f68769432a5d11085a4d11781ee9387561624eceb03a2c50` |
| `src/ops.h` | `d5205048817aed7312589dcb9e0d77bcbd47293bafb8354e42ebbd4f92d05033` |
| `src/repack.cu` | `c80b8ff352664bbc12f8f3d8f08490949974cd523fea6ea399578f25522f20c5` |
| `src/repack.h` | `4cda8ee5aa5ae1a4ce0c99d8a2f46b8806b6184d12146fa8f3b0890959fa922a` |

## safetensors

- Source: crate `safetensors` 0.8.0 from crates.io (`https://static.crates.io/crates/safetensors/safetensors-0.8.0.crate`, archive sha256 `79b079b829cb27a1c3c374341345ed2e8b2c0c839034522cee576c140bd7f846`). Upstream repository: https://github.com/huggingface/safetensors, commit `a406ca3e7a90598be0cd05a50069cb9bf5ef6ba6`, directory `safetensors`, as recorded in the crate's VCS metadata.
- Location: `engine/vendor/safetensors/`. `engine/Cargo.toml` selects it as a `path` dependency.
- Changes:
  - `src/tensor.rs`: one line. The check that the tensor data ends exactly at the end of the buffer uses an overflow-checked addition. `if buffer_end + N_LEN + n != buffer_len {` becomes `if buffer_end.checked_add(stop) != Some(buffer_len) {`.
  - `src/slice.rs`: six existing upstream test expressions are reformatted by `cargo fmt` (formatting only).
  - `README.md`: one trailing space is removed.
  - The archive's `Cargo.lock` and `.cargo_vcs_info.json` are not included.
- `engine/vendor/safetensors/provenance.json` records the upstream and vendored SHA-256 of every file.
- License: Apache-2.0, as declared in the crate's `Cargo.toml`. The full text is in `engine/vendor/safetensors/LICENSE` (sha256 `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`).

## mlx-sys

- Source: `mlx-sys` 0.6.0, https://github.com/oxiglade/mlx-rs.
- Location: `engine/vendor/mlx-sys/`. `engine/Cargo.toml` replaces the crates.io crate with it through `[patch.crates-io]`. Only the optional `native-mlx` feature on Apple Silicon macOS uses it.
- Taken: `build.rs`, `src/lib.rs`, and the mlx-c sources in `src/mlx-c/`. The engine uses the compiled C library and the C type definitions. Every operation call goes through the engine's own `inference_mlx_*` shims in `engine/src/kernels/mlx/native/guard.cpp`.
- Changes: `engine/vendor/mlx-sys-provenance.json` records the crate archive, the files left out of the vendored copy, and the changed paths.
- Build: `build.rs` compiles the vendored mlx-c sources with CMake. `src/mlx-c/CMakeLists.txt` sets `MLX_C_VERSION` to 0.7.0. At build time it fetches MLX from https://github.com/ml-explore/mlx at commit `64ea011cb65f14d9ce2737e60db9a4ae91ed7441`. MLX is not stored in this repository. The same file also fetches fmt (commit `407c905e45ad75fc29bf0f9bb7c5c2fd3475976f`), nlohmann/json 3.11.3, and Apple metal-cpp 26, each pinned by commit or archive hash.
- Note: `engine/vendor/mlx-sys/README.md:5-6` still names the native sources of the published crate (mlx-c `c74db5307cc8ce122f48d97ef951b30578674e7f`, MLX `v0.32.2`). The CMake pin above is what this repository builds.
- License:
  - `mlx-sys`: `Cargo.toml` declares `MIT OR Apache-2.0`. The crate ships both license files: `engine/vendor/mlx-sys/LICENSE-MIT` and `engine/vendor/mlx-sys/LICENSE-APACHE`.
  - mlx-c (https://github.com/ml-explore/mlx-c): MIT, "Copyright (c) 2023 ml-explore". The text is in `engine/vendor/mlx-sys/src/mlx-c/LICENSE`.
  - MLX (fetched at build time, see above): MIT.

## mlx-lm (gated delta kernel)

- Source: Python package `mlx-lm` 0.31.0 (https://github.com/ml-explore/mlx-lm), file `mlx_lm/models/gated_delta.py`, sha256 `3e445528acc01f7bcb0e44847cbdbb1ced69ce81218080f7dd701754e4dee856` (282 lines). The copy was taken from an installed mlx-lm 0.31.0 wheel. The wheel's RECORD file lists the same hash.
- License: MIT, "Copyright © 2023 Apple Inc.", from `mlx_lm-0.31.0.dist-info/licenses/LICENSE`, sha256 `ccfab7ccb2ea306f71531c8ca77bb55507606cd90768b1e32b8b52ab5b48cf01`.
- Taken: the Metal Shading Language source of the gated delta rule kernel (the scalar-gate, unmasked variant), which `mlx_lm/models/gated_delta.py` holds as a Python f-string. `engine/vendor/mlx-lm-gated-delta/extract_kernel.py` extracts it without change into `engine/vendor/mlx-lm-gated-delta/gated_delta_step.metal`. `engine/src/kernels/mlx/native/resident_ops.rs` embeds that file with `include_str!` as `GATED_DELTA_SOURCE`.
- Derived without copying text: the kernel name, the input and output names, the template arguments, the grid, and the threadgroup shape in `gated_delta_kernel_call` (`engine/src/kernels/mlx/native/guard.cpp`, called by `inference_mlx_gated_delta`) follow `_make_gated_delta_kernel` and `gated_delta_kernel` of the same file. The operand layouts in `engine/src/kernels/gated_delta.rs` follow the same file.
- Not copied: no other part of mlx-lm.

License text (`LICENSE` of mlx-lm 0.31.0):

```
MIT License

Copyright © 2023 Apple Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- Files: every file under `engine/vendor/mlx-lm-gated-delta/`, with its current SHA-256. `provenance.json` records the upstream hashes, the derived file, and the empty patch list.

| File | SHA-256 |
|---|---|
| `LICENSE` | `ccfab7ccb2ea306f71531c8ca77bb55507606cd90768b1e32b8b52ab5b48cf01` |
| `README.md` | `cf205f2dc2abb8e0123851eaec93905058408d863dde25bd861a57622ee3fae4` |
| `check_provenance.py` | `938b09b7bacff679b27231a62beb4b029207d24b4c68daaa6ffa2831880803ac` |
| `extract_kernel.py` | `a1241fd3ed13a20af3f6de8f4031ee44a356fc9846c51b8afd757bb251370b5e` |
| `gated_delta_step.metal` | `b4ea3ae7ff2ca0ecb1d203109e8b333d310a0f18a5cf6f882f0e11332d6131fe` |
| `provenance.json` | `c086b791129c9d4fda9b69c135f6132f139a39833c569fc5f2e2cb5c085a2e23` |
| `src/gated_delta.py` | `3e445528acc01f7bcb0e44847cbdbb1ced69ce81218080f7dd701754e4dee856` |

## llguidance / toktrie / derivre

- Crates, from crates.io (not vendored; versions from `engine/Cargo.lock`):
  - `llguidance` 1.7.5, https://github.com/guidance-ai/llguidance. `engine/Cargo.toml` declares it with the `lark` feature and default features off.
  - `toktrie` 1.9.1, https://github.com/guidance-ai/llguidance. A dependency of `llguidance`.
  - `derivre` 0.3.11, https://github.com/microsoft/derivre. A dependency of `llguidance`.
- License: MIT, "Copyright (c) Microsoft Corporation.", for all three crates, as declared in each crate's `Cargo.toml` and `LICENSE`.

## Test data

- `engine/testdata/gemma4/humanizer-config.json` is the unmodified `config.json` of the Hugging Face model `jialinyyzz/humanizer` at revision `2e57b6528a4070c33f3e09ca24e3325a84272f68`. At that revision the repository's `LICENSE` is the Apache License, Version 2.0, and its model card declares `apache-2.0`. Its `NOTICE` file begins: "humanizer / Copyright 2026 Stephen Yu / This product is licensed under the Apache License, Version 2.0 (see LICENSE). This applies to the code in this repository and to the released model weights." `prompt-cases.json` in the same directory holds the prompt instruction and separator from that model's README. It and `parity-prompt.json` hold token ids from that model's tokenizer.
- Other fixtures taken from published model repositories name their source and revision in the README or provenance file of their directory.
