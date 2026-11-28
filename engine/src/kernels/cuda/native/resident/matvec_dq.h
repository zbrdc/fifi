/*
 * Launch entry of the decode dq matvec: y = W . x for one f32 activation row over the file's
 * native Q4_K (144-byte), Q5_K (176-byte), Q6_K (210-byte), or Q8_0 (eight 34-byte blocks = 272
 * bytes per 256 values) blocks (no repack, no quantize, no vendored code).
 * f32 arithmetic throughout. Launch-only: nothing here allocates, copies or synchronizes.
 * Per-kernel layout notes in native/resident/matvec_dq.cu.
 *
 * Every includer is a C++ TU of the native adapter, so the layout is pinned with a C++
 * static_assert here and with the same numbers by the Rust mirror
 * (native/resident/matvec_dq.rs).
 */
#ifndef INFERENCE_CUDA_RESIDENT_MATVEC_DQ_H
#define INFERENCE_CUDA_RESIDENT_MATVEC_DQ_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct inference_cuda_matvec_dq_args {
    const void *w; /* native blocks [rows][k / 256], 16-byte aligned */
    const void *x; /* f32 in [k], 16-byte aligned */
    float *y;      /* f32 out [rows], 4-byte aligned */
    int k;         /* a positive multiple of 256 */
    int rows;      /* any positive count (the kernel owns whole rows) */
    int type;      /* INFERENCE_CUDA_TYPE_Q4_K, Q5_K, Q6_K or Q8_0 */
} inference_cuda_matvec_dq_args;

/* One launch on stream (must not be NULL). Returns INFERENCE_CUDA_INVALID_ARGUMENT before
 * any CUDA call for a NULL pointer or stream, a type the kernel does not serve, k that is not
 * a positive multiple of 256, rows below one, a pointer off its alignment, or a y that shares
 * a byte with w or x; else the launch status. */
int inference_cuda_matvec_dq_launch(const inference_cuda_matvec_dq_args *args,
                                    void *stream, int *out_stage);

#ifdef __cplusplus
}
static_assert(sizeof(inference_cuda_matvec_dq_args) == 40 &&
                  offsetof(inference_cuda_matvec_dq_args, w) == 0 &&
                  offsetof(inference_cuda_matvec_dq_args, x) == 8 &&
                  offsetof(inference_cuda_matvec_dq_args, y) == 16 &&
                  offsetof(inference_cuda_matvec_dq_args, k) == 24 &&
                  offsetof(inference_cuda_matvec_dq_args, rows) == 28 &&
                  offsetof(inference_cuda_matvec_dq_args, type) == 32,
              "inference_cuda_matvec_dq_args layout");
#endif

#endif /* INFERENCE_CUDA_RESIDENT_MATVEC_DQ_H */
