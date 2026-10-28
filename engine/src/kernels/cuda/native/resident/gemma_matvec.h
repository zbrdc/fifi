/*
 * Launch entry of gemma_matvec for the Gemma 4 Humanizer resident step: the dense
 * Q8_0 matvec against a repacked weight plane. The checkpoint's 34-byte blocks are
 * split at load into one int8 quant plane per row and one half scale per block, so a
 * lane can read 16 bytes at a time. The activation keeps the native block_q8_1
 * layout the existing quantizer writes. Launch-only: nothing here allocates, copies
 * or synchronizes.
 */
#ifndef INFERENCE_CUDA_RESIDENT_GEMMA_MATVEC_H
#define INFERENCE_CUDA_RESIDENT_GEMMA_MATVEC_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct inference_cuda_gemma_matvec_args {
    const void *w_qs;      /* int8 [rows][k], read */
    const void *w_scales;  /* half [rows][k/32], read */
    const void *x_q8;      /* block_q8_1 [ncols][ceil(k/512)*16], read */
    void *y;               /* f32 [ncols][rows], written */
    int rows;              /* 1..(1<<20) */
    int k;                 /* multiple of 32, <= 65536 */
    int ncols;             /* 1..8; each activation row padded to 512 values */
} inference_cuda_gemma_matvec_args;

#ifdef __cplusplus
static_assert(sizeof(inference_cuda_gemma_matvec_args) == 48 && offsetof(inference_cuda_gemma_matvec_args, w_qs) == 0 && offsetof(inference_cuda_gemma_matvec_args, w_scales) == 8 && offsetof(inference_cuda_gemma_matvec_args, x_q8) == 16 && offsetof(inference_cuda_gemma_matvec_args, y) == 24 && offsetof(inference_cuda_gemma_matvec_args, rows) == 32 && offsetof(inference_cuda_gemma_matvec_args, k) == 36 && offsetof(inference_cuda_gemma_matvec_args, ncols) == 40, "layout");
#else
_Static_assert(sizeof(inference_cuda_gemma_matvec_args) == 48 && offsetof(inference_cuda_gemma_matvec_args, w_qs) == 0 && offsetof(inference_cuda_gemma_matvec_args, w_scales) == 8 && offsetof(inference_cuda_gemma_matvec_args, x_q8) == 16 && offsetof(inference_cuda_gemma_matvec_args, y) == 24 && offsetof(inference_cuda_gemma_matvec_args, rows) == 32 && offsetof(inference_cuda_gemma_matvec_args, k) == 36 && offsetof(inference_cuda_gemma_matvec_args, ncols) == 40, "layout");
#endif

/* One launch on stream (must not be NULL: no default-stream fallback). Returns
 * INFERENCE_CUDA_INVALID_ARGUMENT before any CUDA call for a bad argument, else the
 * launch status. */
int inference_cuda_gemma_matvec_launch(const inference_cuda_gemma_matvec_args *args, void *stream,
                                       int *out_stage);

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_CUDA_RESIDENT_GEMMA_MATVEC_H */
