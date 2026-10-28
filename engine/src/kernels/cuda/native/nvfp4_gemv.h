/* NVFP4 fused matvec: the device decodes ModelOpt NVFP4 weights, so the host
 * never expands the matrix to F32. Rust/context wiring is separate work. */
#ifndef INFERENCE_CUDA_NVFP4_GEMV_H
#define INFERENCE_CUDA_NVFP4_GEMV_H

#include "abi.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Computes y[r] = sum_c W[r][c] * x[c] for one input row, with W the packed
 * NVFP4 weight of the host reference in engine/src/kernels/nvfp4.rs: rows of
 * columns values, columns a positive multiple of 16, one nibble per value in
 * the ModelOpt 0.37.0 packing (odd << 4) | even, so the low nibble (shift 0)
 * is the even column of each pair, eight packed bytes per group of sixteen
 * consecutive values, and one E4M3FN group-scale byte per group, both arrays
 * row-major. weight_scale_2 is the F32 per-tensor multiplier and must be
 * finite and positive. The device decodes each value as
 * E2M1(nibble) * (E4M3FN(scale byte) * weight_scale_2) and accumulates each
 * row in column order with one F32 rounding per multiply and per add and no
 * fused multiply-add, so y is bit for bit the host WeightBlock::matvec.
 * packed and scale are borrowed device addresses whose extents must equal
 * rows * columns / 2 and rows * columns / 16 bytes exactly. All other extents
 * are byte capacities. x and y are borrowed host addresses, float-aligned,
 * holding at least columns and rows floats. x must contain finite values.
 * Host x and y may overlap: x is copied in before y is written.
 * workspace->x and workspace->y hold columns and rows floats on the device.
 * workspace->xf holds the 4-byte non-finite flag of the kernel and is the
 * only use this call makes of that region: it must hold at least 4 bytes,
 * be unsigned-aligned, and be disjoint from the other regions. The declared
 * extents of packed, scale and the three used workspace regions must be
 * pairwise disjoint. The caller guarantees that each declared extent belongs
 * to live memory.
 * A group scale that is NaN or negative (signed zero included), a
 * group-scale product that is not finite, or a decoded weight that is not
 * finite raises the flag: the call returns INFERENCE_CUDA_INVALID_ARGUMENT,
 * reports STAGE_NONE, and y must not be used. These are exactly the
 * rejections of the host reference. No memory is allocated, freed, or
 * retained by this call. The caller serializes access and selects the
 * buffers' current CUDA device. Work uses the default stream. The call
 * synchronizes before returning after device work, including failed copies
 * or launches, then copies y to the host. Statuses and first-failure stages
 * follow abi.h. Invalid arguments and native exceptions report STAGE_NONE
 * unless a CUDA failure preceded the exception. Sticky CUDA errors remain
 * terminal; this function never resets the device. out_stage may be NULL. */
int inference_cuda_nvfp4_gemv_ws(
    const void *packed, size_t packed_bytes,
    const void *scale, size_t scale_bytes, float weight_scale_2,
    size_t rows, size_t columns,
    const float *x, size_t x_bytes, float *y, size_t y_bytes,
    const inference_cuda_workspace *workspace, int *out_stage);

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_CUDA_NVFP4_GEMV_H */
