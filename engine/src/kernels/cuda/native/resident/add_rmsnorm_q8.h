/*
 * The add + RMSNorm + q8_1 pack entry (native/resident/add_rmsnorm_q8.h, per-kernel header of
 * the layout in kernels/cuda/resident/mod.rs). Launch-only, like the other resident headers:
 * nothing here allocates, copies or synchronizes; a fault in queued work surfaces at the next
 * synchronize. abi_resident.h stays closed.
 */
#ifndef INFERENCE_CUDA_RESIDENT_ADD_RMSNORM_Q8_H
#define INFERENCE_CUDA_RESIDENT_ADD_RMSNORM_Q8_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* One row (the resident step's m = 1): h = h + delta (when delta is not NULL; h is updated in
 * place), then norm = h * rsqrt(mean(h^2) + eps) * w in the SAME reduction order and the same
 * expressions as add_rmsnorm_pack_k (xavier-qwen ops.cu), so norm_out is byte-identical to the
 * norm_out that entry writes on the same input; norm_out receives that f32 row ([n], the router
 * input). q8 receives the block_q8_1 quantization of the SAME normalized values with the vendored
 * quantize_q8_1 block rule (per 32 values amax, d = amax/127, q = amax == 0 ? 0 : roundf(x/d),
 * ds = half2(d, sum x) with the warp butterfly sum), computed in IEEE arithmetic: this entry's
 * translation unit is in the warnings-into-errors group WITHOUT -use_fast_math (unlike the
 * vendored quantize.cu object), so d is the IEEE division and the output is bit-checkable
 * against a host reference. The row is padded to MATRIX_ROW_PADDING (512) values: the tail
 * blocks past n are written as zeros (d = 0, q = 0, s = 0), q8 holds pad(n, 512)/32 blocks of
 * 36 bytes. One kernel, one block per row: 256 threads, or 1024 when `threads` says so (the
 * same block code over a quarter of the strided rounds; the q8 blocks are independent, but the
 * sum of squares adds in another float order, so its last bits differ).
 * h, delta, w, norm_out: f32 [n]; q8:
 * block_q8_1 [pad(n, 512)/32]. n is a multiple of 32 (one q8_1 block) in 32..32768, eps finite and > 0.
 * h (4-byte aligned), norm_out (16-byte aligned, like add_norm_pack's) and q8 (4-byte) are
 * written and must overlap no other range; delta (4) and w (4) are read and may share bytes
 * with each other only. stream must not be NULL. Returns INFERENCE_CUDA_INVALID_ARGUMENT
 * before any CUDA call for a bad argument. */
typedef struct inference_cuda_add_rmsnorm_q8_args {
    void *h;            /* f32 [n * rows], read; written with h + delta when delta is given */
    const void *delta;  /* f32 [n * rows] or NULL */
    const void *w;      /* f32 [n], the RMSNorm weight shared by every row */
    void *norm_out;     /* f32 [n * rows] out, 16-byte aligned: byte-identical to add_norm_pack's */
    void *q8;           /* block_q8_1 out: rows * pad(n, 512)/32 * 36 bytes */
    int n;              /* multiple of 32, 32..32768 */
    float eps;          /* finite, > 0 */
    int rows;           /* row count, 1..262144; one block per row */
    int h_stride;       /* f32 elements between rows of h: 0 = n, else >= n; delta, norm_out and q8 stay dense */
    int threads;        /* threads per block: 0 or 256 = add_rmsnorm_pack_k's summation order; 1024 = another order */
} inference_cuda_add_rmsnorm_q8_args;

#ifdef __cplusplus
static_assert(sizeof(inference_cuda_add_rmsnorm_q8_args) == 64 && offsetof(inference_cuda_add_rmsnorm_q8_args, h) == 0 && offsetof(inference_cuda_add_rmsnorm_q8_args, delta) == 8 && offsetof(inference_cuda_add_rmsnorm_q8_args, w) == 16 && offsetof(inference_cuda_add_rmsnorm_q8_args, norm_out) == 24 && offsetof(inference_cuda_add_rmsnorm_q8_args, q8) == 32 && offsetof(inference_cuda_add_rmsnorm_q8_args, n) == 40 && offsetof(inference_cuda_add_rmsnorm_q8_args, eps) == 44 && offsetof(inference_cuda_add_rmsnorm_q8_args, rows) == 48 && offsetof(inference_cuda_add_rmsnorm_q8_args, h_stride) == 52 && offsetof(inference_cuda_add_rmsnorm_q8_args, threads) == 56,
              "inference_cuda_add_rmsnorm_q8_args layout");
#else
_Static_assert(sizeof(inference_cuda_add_rmsnorm_q8_args) == 64 && offsetof(inference_cuda_add_rmsnorm_q8_args, h) == 0 && offsetof(inference_cuda_add_rmsnorm_q8_args, delta) == 8 && offsetof(inference_cuda_add_rmsnorm_q8_args, w) == 16 && offsetof(inference_cuda_add_rmsnorm_q8_args, norm_out) == 24 && offsetof(inference_cuda_add_rmsnorm_q8_args, q8) == 32 && offsetof(inference_cuda_add_rmsnorm_q8_args, n) == 40 && offsetof(inference_cuda_add_rmsnorm_q8_args, eps) == 44 && offsetof(inference_cuda_add_rmsnorm_q8_args, rows) == 48 && offsetof(inference_cuda_add_rmsnorm_q8_args, h_stride) == 52 && offsetof(inference_cuda_add_rmsnorm_q8_args, threads) == 56,
               "inference_cuda_add_rmsnorm_q8_args layout");
#endif

int inference_cuda_add_rmsnorm_q8_launch(const inference_cuda_add_rmsnorm_q8_args *args,
                                         void *stream, int *out_stage);

/* Scale-consistent, rounded-sum q8_1 variant (FIFI_CUDA_Q8_1_ROUNDED=1). */
int inference_cuda_add_rmsnorm_q8_rounded_launch(const inference_cuda_add_rmsnorm_q8_args *args,
                                                 void *stream, int *out_stage);

/* F32-only variant: updates h += delta and writes bit-identical norm_out, skipping q8_1 quantize
 * and stores when downstream consumers read norm_out directly (e.g. matvec_dq or QK-norm). */
int inference_cuda_add_rmsnorm_f32_launch(const inference_cuda_add_rmsnorm_q8_args *args,
                                          void *stream, int *out_stage);

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_CUDA_RESIDENT_ADD_RMSNORM_Q8_H */
