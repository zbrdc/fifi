/* NVFP4 fused batched matvec: one launch and one device synchronization for a
 * whole layer's selected routed experts, replacing thousands of per-expert
 * synchronous calls (the transport-accounting diagnosis). */
#ifndef INFERENCE_CUDA_NVFP4_GEMV_BATCH_H
#define INFERENCE_CUDA_NVFP4_GEMV_BATCH_H

#include "abi.h"

// Included only from CUDA C++ translation units (today: nvfp4_gemv_batch.cu),
// so the extern "C" scope needs no dual-language guard; a future C includer
// fails at compile time instead of silently linking with C++ mangling.
extern "C" {

/* Computes y[e * rows + r] = sum_c W_e[r][c] * x[e * columns + c] for n
 * experts of one uniform shape, each with its own input row. Every expert's
 * W_e, per-expert scale table and per-expert weight_scale_2 decode exactly as
 * inference_cuda_nvfp4_gemv_ws: the same E2M1 values, the same E4M3FN group
 * scales, one rounded multiply per decode and per input multiply, one rounded
 * add per element, column order, no fused multiply-add, so the batch is bit
 * for bit n independent host reference matvecs.
 *
 * packed_ptrs and scale_ptrs are DEVICE arrays of n device pointers; entry e
 * must address a row-major packed payload of rows * columns / 2 bytes and a
 * group-scale payload of rows * columns / 16 bytes. weight_scales is a
 * DEVICE array of n per-tensor multipliers; the safe wrapper validates it
 * host-side before uploading (finite, positive). x is a borrowed host
 * array of n * columns finite floats; y a host array of n * rows floats; host
 * x and y may not overlap any other host argument. workspace->x holds
 * n * columns floats, workspace->y holds n * rows floats, and workspace->xf
 * holds the 4-byte non-finite flag; the three regions must be pairwise
 * disjoint, in bounds, aligned and live, as must every addressed device
 * payload. One launch covers all experts; the call synchronizes once, then
 * copies y back. A non-finite decoded value in any expert raises the flag:
 * INFERENCE_CUDA_INVALID_ARGUMENT is returned, STAGE_NONE reported, and y
 * must not be used. No memory is allocated, freed or retained. Statuses and
 * stages follow abi.h; out_stage may be NULL. */
int inference_cuda_nvfp4_gemv_batch_ws(
    const void *const *packed_ptrs, size_t packed_ptrs_bytes,
    const void *const *scale_ptrs, size_t scale_ptrs_bytes,
    const float *weight_scales, size_t weight_scales_bytes,
    size_t n, size_t rows, size_t columns,
    const float *x, size_t x_bytes, float *y, size_t y_bytes,
    const inference_cuda_workspace *workspace, int *out_stage);

}

#endif /* INFERENCE_CUDA_NVFP4_GEMV_BATCH_H */
