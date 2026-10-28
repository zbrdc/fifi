/*
 * gemma_matvec: the dense Q8_0 matvec of the resident step against a repacked plane.
 *
 * Layout, per weight row: k int8 quants, then k/32 half scales. The checkpoint stores
 * 34-byte blocks (half scale, then 32 quants); the loader splits them so a lane reads
 * 16-byte quant vectors. The activation keeps the native block_q8_1 layout: 36 bytes
 * per block, the half2 scale and sum first, then the 32 quants. One warp owns one row
 * and strides the blocks, which keeps four 16-byte loads in flight per lane.
 */
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stddef.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"
#include "resident/gemma_matvec.h"

namespace {

constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kActBlockBytes = 36;  // block_q8_1: half2 ds, then 32 quants

bool aligned(const void *p, size_t a) { return (uintptr_t)p % a == 0; }

bool apart(const void *a, size_t ab, const void *b, size_t bb) {
    const uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    return x <= y ? ab <= y - x : bb <= x - y;
}

__device__ __forceinline__ float dot_block(const int4 w0, const int4 w1, const int *a) {
    int d = __dp4a(w0.x, a[0], 0);
    d = __dp4a(w0.y, a[1], d);
    d = __dp4a(w0.z, a[2], d);
    d = __dp4a(w0.w, a[3], d);
    d = __dp4a(w1.x, a[4], d);
    d = __dp4a(w1.y, a[5], d);
    d = __dp4a(w1.z, a[6], d);
    d = __dp4a(w1.w, a[7], d);
    return (float)d;
}

template <int NCols>
__global__ void gemma_matvec_kernel(const signed char *__restrict__ w_qs,
                                    const __half *__restrict__ w_scales,
                                    const signed char *__restrict__ x_q8,
                                    float *__restrict__ y, int rows, int k) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * kWarps + warp;
    if (row >= rows) return;
    const int blocks = k >> 5;
    const signed char *row_qs = w_qs + (size_t)row * (size_t)k;
    const __half *row_scales = w_scales + (size_t)row * (size_t)blocks;
    float acc0[NCols] = {}, acc1[NCols] = {};
    const size_t stride = (size_t)((k + 511) / 512) * 16 * kActBlockBytes;
    for (int b = lane; b < blocks; b += 32) {
        const int4 w0 = *reinterpret_cast<const int4 *>(row_qs + (size_t)b * 32);
        const int4 w1 = *reinterpret_cast<const int4 *>(row_qs + (size_t)b * 32 + 16);
#pragma unroll
        for (int col = 0; col < NCols; ++col) {
            const signed char *act = x_q8 + col * stride + (size_t)b * kActBlockBytes;
            const int *a = reinterpret_cast<const int *>(act + 4);
            const float s = __half2float(row_scales[b]) *
                            __half2float(*reinterpret_cast<const __half *>(act));
            if (b & 32) {
                acc1[col] += s * dot_block(w0, w1, a);
            } else {
                acc0[col] += s * dot_block(w0, w1, a);
            }
        }
    }
#pragma unroll
    for (int col = 0; col < NCols; ++col) {
        float acc = acc0[col] + acc1[col];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0) y[(size_t)col * rows + row] = acc;
    }
}

}  // namespace

extern "C" int inference_cuda_gemma_matvec_launch(const inference_cuda_gemma_matvec_args *args,
                                                  void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || stream == NULL) return INFERENCE_CUDA_INVALID_ARGUMENT;
        if (args->w_qs == NULL || args->w_scales == NULL || args->x_q8 == NULL || args->y == NULL)
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        const int rows = args->rows, k = args->k, ncols = args->ncols;
        if (ncols < 1 || ncols > 8) return INFERENCE_CUDA_INVALID_ARGUMENT;
        if (rows < 1 || rows > (1 << 20)) return INFERENCE_CUDA_INVALID_ARGUMENT;
        if (k < 32 || k > 65536 || k % 32 != 0) return INFERENCE_CUDA_INVALID_ARGUMENT;
        if (!aligned(args->w_qs, 16) || !aligned(args->x_q8, 4) || !aligned(args->w_scales, 2) ||
            !aligned(args->y, 4))
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        const size_t blocks = (size_t)(k / 32);
        const size_t w_qs_bytes = (size_t)rows * (size_t)k;
        const size_t w_scales_bytes = (size_t)rows * blocks * 2;
        const size_t y_bytes = (size_t)rows * ncols * 4;
        const size_t stride = (size_t)((k + 511) / 512) * 16;
        const size_t x_bytes = ((size_t)(ncols - 1) * stride + blocks) * kActBlockBytes;
        if (!apart(args->y, y_bytes, args->w_qs, w_qs_bytes) ||
            !apart(args->y, y_bytes, args->w_scales, w_scales_bytes) ||
            !apart(args->y, y_bytes, args->x_q8, x_bytes))
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        const int grid = (rows + kWarps - 1) / kWarps;
        Failure failure;
#define LAUNCH_COLS(N) \
    case N: gemma_matvec_kernel<N><<<grid, kThreads, 0, (cudaStream_t)stream>>>( \
        (const signed char *)args->w_qs, (const __half *)args->w_scales, \
        (const signed char *)args->x_q8, (float *)args->y, rows, k); break
        switch (ncols) {
            LAUNCH_COLS(1);
            LAUNCH_COLS(2);
            LAUNCH_COLS(3);
            LAUNCH_COLS(4);
            LAUNCH_COLS(5);
            LAUNCH_COLS(6);
            LAUNCH_COLS(7);
            LAUNCH_COLS(8);
        }
#undef LAUNCH_COLS
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
