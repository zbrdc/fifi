/*
 * Launch entry of the vendored quantize_row_q8_1: one kernel that turns f32 activation rows into
 * block_q8_1 (d = amax/127, codes, s = the raw f32 sum of the inputs), in the MATRIX_ROW_PADDING
 * (512) layout MMVQ reads. Launch-only, like the other resident headers: nothing here allocates,
 * copies or synchronizes; a fault in queued work surfaces at the next synchronize. Per-kernel
 * header of the layout in kernels/cuda/resident/mod.rs; abi_resident.h stays closed.
 */
#ifndef INFERENCE_CUDA_RESIDENT_QUANTIZE_Q8_1_H
#define INFERENCE_CUDA_RESIDENT_QUANTIZE_Q8_1_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct inference_cuda_quantize_q8_1_args {
    const void *x; /* f32 [ncols][k], rows of k values back to back */
    void *y;       /* block_q8_1 out: q8_1_bytes(k, ncols) bytes */
    int k;         /* values per row, a multiple of 32 */
    int ncols;     /* rows, 1..8 */
} inference_cuda_quantize_q8_1_args;

#ifdef __cplusplus
static_assert(sizeof(inference_cuda_quantize_q8_1_args) == 24 && offsetof(inference_cuda_quantize_q8_1_args, x) == 0 && offsetof(inference_cuda_quantize_q8_1_args, y) == 8 && offsetof(inference_cuda_quantize_q8_1_args, k) == 16 && offsetof(inference_cuda_quantize_q8_1_args, ncols) == 20,
              "inference_cuda_quantize_q8_1_args layout");
#else
_Static_assert(sizeof(inference_cuda_quantize_q8_1_args) == 24 && offsetof(inference_cuda_quantize_q8_1_args, x) == 0 && offsetof(inference_cuda_quantize_q8_1_args, y) == 8 && offsetof(inference_cuda_quantize_q8_1_args, k) == 16 && offsetof(inference_cuda_quantize_q8_1_args, ncols) == 20,
               "inference_cuda_quantize_q8_1_args layout");
#endif

/* One launch on stream (must not be NULL: no default-stream fallback). y holds
 * ncols * pad(k, 512)/32 blocks of 36 bytes; the kernel writes the padded tail of each row as
 * zeros (xi = 0 gives d = 0, q = 0, s = 0). x is read as k values per row, unpadded. Returns
 * INFERENCE_CUDA_INVALID_ARGUMENT before any CUDA call for a bad argument, else the launch status.
 * The group is built with -use_fast_math: the f32 scale can be up to 3 ulp from the IEEE value and
 * codes can differ at rounding boundaries. */
int inference_cuda_quantize_q8_1_launch(const inference_cuda_quantize_q8_1_args *args, void *stream,
                                        int *out_stage);

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_CUDA_RESIDENT_QUANTIZE_Q8_1_H */
