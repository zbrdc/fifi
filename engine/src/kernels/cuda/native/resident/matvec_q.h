/*
 * Launch entry of the vendored MMVQ matvec: y = x_q8 . W^T against native GGUF-layout weight
 * blocks, through fifi_mmvq_launch of the trimmed vendored mmvq.cu. Launch-only: nothing here
 * allocates, copies or synchronizes. Per-kernel header of the layout in
 * kernels/cuda/resident/mod.rs; abi_resident.h stays closed.
 */
#ifndef INFERENCE_CUDA_RESIDENT_MATVEC_Q_H
#define INFERENCE_CUDA_RESIDENT_MATVEC_Q_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct inference_cuda_matvec_q_args {
    int type;        /* INFERENCE_CUDA_TYPE_Q4_K, Q5_0, Q6_K or Q8_0 */
    const void *w;   /* native blocks [rows][k / qk] */
    const void *x_q8;/* block_q8_1 [ncols][pad(k, 512)/32] */
    float *y;        /* f32 out [ncols][rows], column j at y + j*rows */
    int k;           /* a multiple of the type's block size */
    int rows;        /* a multiple of 4 (the kernel reads whole row groups) */
    int ncols;       /* 1..8 */
} inference_cuda_matvec_q_args;

#ifdef __cplusplus
static_assert(sizeof(inference_cuda_matvec_q_args) == 48 && offsetof(inference_cuda_matvec_q_args, type) == 0 && offsetof(inference_cuda_matvec_q_args, w) == 8 && offsetof(inference_cuda_matvec_q_args, x_q8) == 16 && offsetof(inference_cuda_matvec_q_args, y) == 24 && offsetof(inference_cuda_matvec_q_args, k) == 32 && offsetof(inference_cuda_matvec_q_args, rows) == 36 && offsetof(inference_cuda_matvec_q_args, ncols) == 40,
              "inference_cuda_matvec_q_args layout");
#else
_Static_assert(sizeof(inference_cuda_matvec_q_args) == 48 && offsetof(inference_cuda_matvec_q_args, type) == 0 && offsetof(inference_cuda_matvec_q_args, w) == 8 && offsetof(inference_cuda_matvec_q_args, x_q8) == 16 && offsetof(inference_cuda_matvec_q_args, y) == 24 && offsetof(inference_cuda_matvec_q_args, k) == 32 && offsetof(inference_cuda_matvec_q_args, rows) == 36 && offsetof(inference_cuda_matvec_q_args, ncols) == 40,
               "inference_cuda_matvec_q_args layout");
#endif

/* One launch on stream (must not be NULL). The weights are read in the file's native block layout;
 * x_q8 is the padded quantize_q8_1 output of ncols activation rows. Returns
 * INFERENCE_CUDA_INVALID_ARGUMENT before any CUDA call for a bad argument, else the launch status.
 * The kernel's column j is computed from x_q8 row j alone (the row-invariance contract). */
int inference_cuda_matvec_q_launch(const inference_cuda_matvec_q_args *args, void *stream,
                                   int *out_stage);

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_CUDA_RESIDENT_MATVEC_Q_H */
