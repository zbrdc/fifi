/* Staged Nemotron projection. Rust/context and Mamba wiring are separate work. */
#ifndef INFERENCE_CUDA_NEMOTRON_GEMV_H
#define INFERENCE_CUDA_NEMOTRON_GEMV_H

#include "abi.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Computes y[r] = sum_c weight[r * columns + c] * x[c] for one input row.
 * Weights are contiguous row-major F32, already decoded from FP8/NVFP4.
 * Products and accumulation use F32, without activation conversion.
 * rows and columns must be positive. All extents are byte capacities.
 * weight is a borrowed device address; x and y are borrowed host addresses.
 * The caller guarantees that each declared extent belongs to live memory.
 * All data addresses must be float-aligned, and x must contain finite values.
 * workspace->x and workspace->y hold columns and rows floats on the device.
 * Their declared extents and weight's declared extent must not overlap.
 * workspace->xf and xf_bytes are unused and ignored, including NULL/zero.
 * Host x and y may overlap: x is copied before y is written.
 * No memory is allocated, freed, or retained by this call.
 * The caller serializes access and selects the buffers' current CUDA device.
 * Work uses the default stream. The call synchronizes before returning after
 * device work, including failed copies or launches, then copies y to the host.
 * Statuses and first-failure stages follow abi.h. Invalid arguments and native
 * exceptions report STAGE_NONE unless a CUDA failure preceded the exception.
 * Sticky CUDA errors remain terminal; this function never resets the device.
 * out_stage may be NULL. On failure, y must not be used as a result. */
int inference_cuda_nemotron_gemv_f32_ws(
    const void *weight, size_t weight_bytes, size_t rows, size_t columns,
    const float *x, size_t x_bytes, float *y, size_t y_bytes,
    const inference_cuda_workspace *workspace, int *out_stage);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Greedy head step: computes y[r] as above, then returns only the winning token.
 * The full row stays on the device and the caller never receives it.
 * scratch must be a device region of at least 16 bytes, 4-byte aligned, and
 * disjoint from weight and from the workspace regions. The adapter writes the
 * winning index (little-endian u32) to scratch + 0, its logit (f32) to
 * scratch + 4, and a non-finite flag (u32) to scratch + 8.
 * out_host must have at least 8 bytes and receives the index and logit only.
 * The lowest index wins a tie. If any y value is not finite, the call returns
 * INFERENCE_CUDA_INVALID_ARGUMENT and leaves out_host unchanged.
 * rows must fit u32. Other statuses, stages, synchronization and stream rules
 * match the GEMV call above. */
int inference_cuda_nemotron_gemv_argmax_f32_ws(
    const void *weight, size_t weight_bytes, size_t rows, size_t columns,
    const float *x, size_t x_bytes, void *scratch, size_t scratch_bytes,
    void *out_host, size_t out_host_bytes,
    const inference_cuda_workspace *workspace, int *out_stage);

#ifdef __cplusplus
}
#endif


#endif /* INFERENCE_CUDA_NEMOTRON_GEMV_H */
