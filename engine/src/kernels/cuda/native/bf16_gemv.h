/* BF16 dense matvec: the device reads the stored 16-bit codes, so the host
 * never expands a dense matrix to F32 (M2 rule: no F32 weight materialization). */
#ifndef INFERENCE_CUDA_BF16_GEMV_H
#define INFERENCE_CUDA_BF16_GEMV_H

#include "abi.h"

// Included only from CUDA C++ translation units (today: bf16_gemv.cu), so the
// extern "C" scope needs no dual-language guard; a future C includer fails
// at compile time instead of silently linking with C++ mangling.
extern "C" {

/* Computes y[r] = sum_c W[r][c] * x[c] for one input row, with W the stored
 * BF16 dense weight exactly as the host loader reads it: rows of columns
 * little-endian 16-bit codes, row-major, widened to F32 by appending sixteen
 * zero bits, which is exact. The kernel accumulates each row in column order
 * with one F32 rounding per multiply and per add and no fused multiply-add,
 * so y is bit for bit the host matvec of the widened matrix.
 * codes is a borrowed device address whose extent must equal
 * rows * columns * 2 bytes exactly. All other extents are byte capacities.
 * x and y are borrowed host addresses, float-aligned, holding at least
 * columns and rows floats. x must contain finite values. Host x and y may
 * overlap: x is copied in before y is written. workspace->x and workspace->y
 * hold columns and rows floats on the device. workspace->xf holds the 4-byte
 * non-finite flag of the kernel and is the only use this call makes of that
 * region: it must hold at least 4 bytes, be unsigned-aligned, and be
 * disjoint from the other regions. The declared extents of codes and the
 * three used workspace regions must be pairwise disjoint. The caller
 * guarantees that each declared extent belongs to live memory.
 * A widened weight that is not finite raises the flag: the call returns
 * INFERENCE_CUDA_INVALID_ARGUMENT, reports STAGE_NONE, and y must not be
 * used. These are exactly the rejections of the host loader. No memory is
 * allocated, freed, or retained by this call. The caller serializes access
 * and selects the buffers' current CUDA device. Work uses the default
 * stream. The call synchronizes before returning after device work,
 * including failed copies or launches, then copies y to the host. Statuses
 * and first-failure stages follow abi.h. Invalid arguments and native
 * exceptions report STAGE_NONE unless a CUDA failure preceded the exception.
 * Sticky CUDA errors remain terminal; this function never resets the device.
 * out_stage may be NULL. */
int inference_cuda_bf16_gemv_ws(
    const void *codes, size_t codes_bytes,
    size_t rows, size_t columns,
    const float *x, size_t x_bytes, float *y, size_t y_bytes,
    const inference_cuda_workspace *workspace, int *out_stage);

/* Launch-only variant for the device chain (NEM35_DEVICE_CHAIN): x is a
 * borrowed DEVICE address of at least columns floats; the result lands in
 * workspace->y on the device. Enqueues a flag clear and the GEMV kernel on
 * the given cudaStream_t (NULL = default) and returns without waiting: no
 * host-device copy, no allocation, no synchronization. The caller retains
 * every owner, waits on the stream, then checks workspace->xf before using
 * the output. All other contracts of the _ws variant carry over. */
int inference_cuda_bf16_gemv_launch(
    const void *codes, size_t codes_bytes,
    size_t rows, size_t columns,
    const float *x, size_t x_bytes,
    const inference_cuda_workspace *workspace, void *stream, int *out_stage);

}

#endif /* INFERENCE_CUDA_BF16_GEMV_H */
