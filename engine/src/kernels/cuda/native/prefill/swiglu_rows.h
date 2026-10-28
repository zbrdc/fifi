/* CPF-7a part 1: launch entry of swiglu_rows, the M-row SwiGLU with f32 output of the
 * chunked-prefill MoE path (decision 0049; CPF-V4-CPF7-PLAN C.2, N2), one header per kernel
 * file; abi_resident.h stays closed.
 *
 * act[r][c] = ggml_cuda_op_silu_single(gu[r][c]) * gu[r][up_off + c] for r < rows, c < n_ff:
 * the upstream silu of the in-tree shim (ggml_shim/unary.cuh, MIT-derived) under the MMQ glue
 * group's -use_fast_math, as the llama.cpp oracle builds it (__expf, approximate division,
 * subnormals flushed to zero; tolerance row SW0). The routed experts run it on 8M token-slot
 * rows, the shared expert on M rows; act feeds the MMQ q8_1 quantizer. No device index, no
 * status word, deterministic.
 *
 * gu  f32 [rows][ld_in], read only: gate at columns 0..n_ff, up at up_off..up_off + n_ff.
 * act f32 [rows][ld_out]: columns 0..n_ff written, no other byte.
 * Rules (else INFERENCE_CUDA_INVALID_ARGUMENT before any CUDA call): gu, act not NULL and
 * 16-byte aligned (the kernel moves float4); rows 1..16384 (8 x the 2,048-row chunk cap);
 * n_ff a multiple of 4 in 4..32768; up_off, ld_in, ld_out multiples of 4 with up_off >= n_ff,
 * ld_in >= up_off + n_ff, ld_out >= n_ff; the act extent ((rows - 1) * ld_out + n_ff floats)
 * shares no byte with the gu extent ((rows - 1) * ld_in + up_off + n_ff floats). The launch
 * is asynchronous on stream (NULL: the default stream; the Rust wrapper always passes one).
 */
#ifndef INFERENCE_CUDA_PREFILL_SWIGLU_ROWS_H
#define INFERENCE_CUDA_PREFILL_SWIGLU_ROWS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct inference_cuda_swiglu_rows_args {
    const void *gu; /* f32 [rows][ld_in], read only */
    void *act;      /* f32 [rows][ld_out] out */
    int rows;       /* 1..16384 */
    int n_ff;       /* a multiple of 4 in 4..32768 */
    int up_off;     /* column of the up half, a multiple of 4, >= n_ff */
    int ld_in;      /* a multiple of 4, >= up_off + n_ff */
    int ld_out;     /* a multiple of 4, >= n_ff */
} inference_cuda_swiglu_rows_args;

int inference_cuda_swiglu_rows_launch(const inference_cuda_swiglu_rows_args *args, void *stream,
                                      int *out_stage);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
static_assert(sizeof(inference_cuda_swiglu_rows_args) == 40 &&
              offsetof(inference_cuda_swiglu_rows_args, gu) == 0 &&
              offsetof(inference_cuda_swiglu_rows_args, act) == 8 &&
              offsetof(inference_cuda_swiglu_rows_args, rows) == 16 &&
              offsetof(inference_cuda_swiglu_rows_args, n_ff) == 20 &&
              offsetof(inference_cuda_swiglu_rows_args, up_off) == 24 &&
              offsetof(inference_cuda_swiglu_rows_args, ld_in) == 28 &&
              offsetof(inference_cuda_swiglu_rows_args, ld_out) == 32,
              "inference_cuda_swiglu_rows_args layout");
#else
_Static_assert(sizeof(inference_cuda_swiglu_rows_args) == 40 &&
               offsetof(inference_cuda_swiglu_rows_args, gu) == 0 &&
               offsetof(inference_cuda_swiglu_rows_args, act) == 8 &&
               offsetof(inference_cuda_swiglu_rows_args, rows) == 16 &&
               offsetof(inference_cuda_swiglu_rows_args, n_ff) == 20 &&
               offsetof(inference_cuda_swiglu_rows_args, up_off) == 24 &&
               offsetof(inference_cuda_swiglu_rows_args, ld_in) == 28 &&
               offsetof(inference_cuda_swiglu_rows_args, ld_out) == 32,
               "inference_cuda_swiglu_rows_args layout");
#endif

#endif /* INFERENCE_CUDA_PREFILL_SWIGLU_ROWS_H */
