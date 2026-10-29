/*
 * Launch entries of the Q5_0 decode matmul: the dense form (m <= 16 tokens against a Q5_0 weight)
 * and the resident expert form (m <= 8 tokens against expert ids_dev[slot] of a stack). The weight
 * is the 8-row x 128-k tile of tile_q5_0.rs (704 bytes: d[32] fp16, qh[32] u32, qs[32][16]),
 * group-major. The kernel is glue over the unmodified vendored mma884.cuh; the vendored loop cannot
 * take a 128-wide tile, so no vendored matmul is called. Launch-only, like abi_resident.h: nothing
 * here allocates, copies or synchronizes; a fault in queued work surfaces at the next synchronize.
 * This is the per-kernel header of kernels/cuda/resident/mod.rs.
 *
 * Activation contract (K = 896 is not a multiple of 256): xf is the vendored fragment-order buffer
 * on the 256 grid, ceil(k / 256) tiles wide, as inference_cuda_pack_launch writes it from a row of
 * k_x = ceil(k / 256) * 256 values whose tail (k..k_x-1) is zero. The kernel never reads the xf
 * content beyond k, so the tail may hold anything.
 */
#ifndef INFERENCE_CUDA_RESIDENT_MATMUL_Q5_0_H
#define INFERENCE_CUDA_RESIDENT_MATMUL_Q5_0_H

#include <stddef.h>

#ifdef __cplusplus
#include <cuda_runtime.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* y[t * ldy + r] = (x W^T)[t][r] for m <= 16 tokens against the n_rows x k Q5_0 tiles at weight, on
 * stream. weight and xf_dev are 16-byte aligned, y_dev 4-byte aligned, ldy >= n_rows, and y_dev
 * must not overlap xf (the fixed fragment-order buffer, xf-sized for m <= 16) or the weight. stream
 * must not be NULL. The shape must be one that inference_cuda_q5_0_shape_ok accepts (expert 0). A
 * rejected argument returns INFERENCE_CUDA_INVALID_ARGUMENT before any CUDA call; otherwise the
 * launch status. */
int inference_cuda_matmul_q5_0_launch(int m, const void *weight, const void *xf_dev, float *y_dev,
                                      int ldy, int n_rows, int k, void *stream, int *out_stage);

/* The expert form, m <= 8: expert ids_dev[slot] of `count` experts of n_rows x k back to back at
 * stack. Argument order, alignment and overlap rules of inference_cuda_matmul_expert_launch (the
 * type argument is gone); the 12 arguments mirror that entry. An id outside [0, count) writes NaN
 * to y and reads no weight. */
int inference_cuda_matmul_expert_q5_0_launch(int m, const void *stack, const int *ids_dev, int slot,
                                             int count, const void *xf_dev, float *y_dev, int ldy,
                                             int n_rows, int k, void *stream, int *out_stage);

/* Host only, no CUDA call: INFERENCE_CUDA_OK when m tokens against n_rows x k Q5_0 values are a
 * shape the entries accept: m in [1, 8] (expert != 0) or [1, 16], n_rows a positive multiple of 8,
 * k a positive multiple of 128 and at most 32768; else INFERENCE_CUDA_INVALID_ARGUMENT. */
int inference_cuda_q5_0_shape_ok(int m, int n_rows, int k, int expert);

#ifdef __cplusplus
}

/* C++ only, for projector.cu's blocking linear path: the dense kernel on `stream` (nullptr: the
 * default stream). The caller has validated the shape. Returns the launch status. */
namespace q5_0 {
cudaError_t matmul(int m, const uint4 *weight, const uint4 *xf, float *y, int ldy, int n_rows,
                   int k, cudaStream_t stream);
}  // namespace q5_0
#endif

#endif /* INFERENCE_CUDA_RESIDENT_MATMUL_Q5_0_H */
