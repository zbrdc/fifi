/*
 * Launch entry of row_add for the Gemma 4 Humanizer residual sums: one kernel that
 * computes out = x + y element-wise in f32. Launch-only: nothing here allocates,
 * copies or synchronizes.
 */
#ifndef INFERENCE_CUDA_RESIDENT_ROW_ADD_H
#define INFERENCE_CUDA_RESIDENT_ROW_ADD_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct inference_cuda_row_add_args {
    void *out;       /* f32 [n], written, disjoint from x and y */
    const void *x;   /* f32 [n], read */
    const void *y;   /* f32 [n], read */
    int n;           /* values, a multiple of 32, 32..4194304 */
} inference_cuda_row_add_args;

#ifdef __cplusplus
static_assert(sizeof(inference_cuda_row_add_args) == 32 && offsetof(inference_cuda_row_add_args, out) == 0 && offsetof(inference_cuda_row_add_args, x) == 8 && offsetof(inference_cuda_row_add_args, y) == 16 && offsetof(inference_cuda_row_add_args, n) == 24, "layout");
#else
_Static_assert(sizeof(inference_cuda_row_add_args) == 32 && offsetof(inference_cuda_row_add_args, out) == 0 && offsetof(inference_cuda_row_add_args, x) == 8 && offsetof(inference_cuda_row_add_args, y) == 16 && offsetof(inference_cuda_row_add_args, n) == 24, "layout");
#endif

/* One launch on stream (must not be NULL). out is written; x and y are read.
 * Returns INFERENCE_CUDA_INVALID_ARGUMENT before any CUDA call for a bad argument,
 * else the launch status. */
int inference_cuda_row_add_launch(const inference_cuda_row_add_args *args, void *stream,
                                  int *out_stage);

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_CUDA_RESIDENT_ROW_ADD_H */
