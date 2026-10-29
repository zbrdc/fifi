# Gemma CUDA kernel fixtures

These files hold inputs and expected outputs for the Gemma 4 CUDA kernel tests.
The expected outputs were computed outside the engine, independently of the CUDA kernels.
The scripts that generated them are not included in this repository.

`manifest.json` records the hashes, geometry, binary formats, and field offsets of the attention, RoPE, and GELU files.
It also records the SHA-256 of the two generator scripts (`source_attn_sha256`, `source_pointwise_sha256`).

Attention files contain F32 queries, head-major int8 K/V planes, binary16 scale
planes, and F64 expected outputs. RoPE files contain one F32 head followed by
its F64 output at position 8191. The GELU file contains eleven F32 gates,
eleven F32 multipliers, and eleven F64 expected outputs.

These references use Python standard-library arithmetic, not engine code.
The equations come from Transformers commit a08ace4bbd97e721c98751deec37d87b026acadc.

Two more files are not listed in `manifest.json`:

| File | Bytes | SHA-256 | Content | Test |
|---|---|---|---|---|
| `rmsnorm-frozen.bin` | 442,516 | `e10e8c1394a3661aed7133d5412bb58f2d4686c14c5671fb7eb1dc763789cd3f` | 18 RMSNorm cases (widths 256, 512, and 3840, six each; epsilon 1e-6). Layout: u32 case count, then per case u32 width `n`, f32 epsilon, `n` f32 inputs, `n` f32 weights, `n` f64 expected outputs. | `engine/src/kernels/cuda/resident/gemma4_reference_tests.rs` |
| `softcap-frozen.bin` | 928 | `8986ea3ec36f782108779cb93fc40820264d217db0a7b7b3d71b2504d0d73616` | 3 logit softcap cases (cap 30; 11, 32, and 32 values). Layout: u32 case count, then per case f32 cap, u32 count `n`, `n` f32 inputs, `n` f64 expected outputs. | `engine/src/kernels/cuda/resident/softcap/tests.rs` |

These fixtures do not qualify complete-model execution or device performance.
