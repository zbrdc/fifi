#ifndef INFERENCE_CUDA_NVFP4_GROUPED_GEMM_H
#define INFERENCE_CUDA_NVFP4_GROUPED_GEMM_H
#include "abi.h"
extern "C" {

/* One grouped GEMM over n expert/token pairs. Consecutive pairs sharing packed,
 * scale and positive finite weight_scale_2 form a group. Each pointer table is
 * on device; each payload has [rows,columns] with low-even NVFP4 nibbles and one
 * E4M3FN scale per16 columns. FP16 operands, FP32 MMA accumulation, FP32 output:
 * unlike scalar GEMV this is approximate and requires the prefill quality gate.
 *
 * Caller owns all buffers and selects their CUDA device. workspace x/y hold
 * n*columns/n*rows floats; xf is a4-byte rejection flag. All regions and tables
 * must be live, aligned, in bounds and pairwise disjoint. Addressed weight
 * payloads must be live, read-only and correctly sized for the whole operation.
 * SM70+ required. Nonfinite/negative scales or FP16 overflow reject the result.
 * No function allocates, frees or retains pointers.
 *
 * launch clears xf and enqueues ONE GEMM on stream (NULL=default), with no wait.
 * Caller must retain every owner, wait for completion and check xf before using
 * output; xf!=0 invalidates all output. The synchronous ws variant takes finite
 * HOST x/y, copies in, launches, waits, checks xf and copies out. A rejection
 * returns INVALID_ARGUMENT/STAGE_NONE. Other status/stages follow abi.h. */
int inference_cuda_nvfp4_grouped_gemm_launch(
    const void *const *packed_ptrs, size_t packed_ptrs_bytes,
    const void *const *scale_ptrs, size_t scale_ptrs_bytes,
    const float *weight_scales, size_t weight_scales_bytes,
    size_t n, size_t rows, size_t columns,
    const inference_cuda_workspace *workspace, void *stream, int *out_stage);

int inference_cuda_nvfp4_grouped_gemm_ws(
    const void *const *packed_ptrs, size_t packed_ptrs_bytes,
    const void *const *scale_ptrs, size_t scale_ptrs_bytes,
    const float *weight_scales, size_t weight_scales_bytes,
    size_t n, size_t rows, size_t columns,
    const float *x, size_t x_bytes, float *y, size_t y_bytes,
    const inference_cuda_workspace *workspace, int *out_stage);
}
#endif
