/*
 * The in-house Q5_K decode matvec. No vendored includes: the weights are the file's own
 * 176-byte Q5_K blocks (see the layout note at q5k_group_factors), the activation is the
 * shared quantize_q8_1 output (36 bytes per 32 values: fp16 d, fp16 s, 32 int8 quants;
 * only d and the quants feed this kernel), and the arithmetic is plain f32.
 *
 * One warp owns one weight row and walks every activation column in turn, so a row's
 * blocks stay hot in cache across columns instead of being re-fetched per column. Each
 * lane carries 4 chunks x 2 halves = 8 of the 256 values of every block, so a block's
 * factor bytes and qh byte are read by the same lane positions across blocks. The
 * numerics match the CPU dequant oracle (weights/gguf_dequant.rs q5_k) exactly.
 *
 * Every exported function has C linkage, takes and returns C types only, and wraps its body
 * in try/catch, like matvec_q.cu.
 */
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stddef.h>
#include <stdint.h>

#include <string.h>

#include "abi.h"
#include "device.h"
#include "resident/matvec_q5k.h"
#include "resident/k4_scales.cuh"

namespace {

const int kQK = 256;    /* values per Q5_K block */
const int kBlockBytes = 176;
const int kXBlock = 36; /* bytes per q8_1 block of 32 values */
const int kWarps = 8;   /* warps per CTA */

__device__ __forceinline__ float half_at(const unsigned char *p) {
    __half raw;
    memcpy(&raw, p, sizeof(raw));
    return __half2float(raw);
}

/* Q5_K block layout (176 bytes): an f16 value scale at +0, an f16 offset scale at +2,
 * 12 packed 6-bit group factors at +4, 32 high-bit bytes at +16, 128 4-bit code bytes at
 * +48. The 256 values split into 8 groups of 32; group g carries a 6-bit multiplier and
 * 6-bit offset, and value i of the block weighs
 *   (value_scale * mult[g]) * (code_i + 16 * high_i) - offset_scale * offs[g].
 * The factor bytes unpack by scale_min_k4 (resident/k4_scales.cuh). */
__device__ __forceinline__ void q5k_group_factors(const unsigned char *raw, float *mult,
                                                  float *offs) {
    #pragma unroll
    for (int g = 0; g < 8; ++g) {
        unsigned sc, mn;
        scale_min_k4(raw, g, sc, mn);
        mult[g] = (float)sc;
        offs[g] = (float)mn;
    }
}

/* One lane's f32 terms of the block: chunk c (0..3) carries values [64c, 64c+64) of the
 * block, the low half in code bits 0..3 with high bit 2c, the high half in bits 4..7 with
 * high bit 2c+1, each half against its own q8_1 block of 32. */
__device__ __forceinline__ void chunk_terms(const unsigned char *block, const unsigned char *xrow,
                                            int b, int c, int l, float d, float dmin,
                                            const float *mult, const float *offs, float *acc) {
    const float mul1 = d * mult[2 * c];
    const float off1 = dmin * offs[2 * c];
    const float mul2 = d * mult[2 * c + 1];
    const float off2 = dmin * offs[2 * c + 1];
    const unsigned bit1 = 1u << (2 * c);
    const unsigned bit2 = 2u << (2 * c);
    const unsigned char code = block[48 + 32 * c + l];
    const unsigned char high = block[16 + l];
    const float v1 = (float)((code & 0x0f) + ((high & bit1) ? 16 : 0));
    const float v2 = (float)((code >> 4) + ((high & bit2) ? 16 : 0));
    /* Value 64c + l sits in q8_1 block 8b + 2c, value 64c + 32 + l in 8b + 2c + 1. */
    const unsigned char *xb1 = xrow + (size_t)(8 * b + 2 * c) * kXBlock;
    const unsigned char *xb2 = xrow + (size_t)(8 * b + 2 * c + 1) * kXBlock;
    const float x1 = half_at(xb1) * (float)(int8_t)xb1[4 + l];
    const float x2 = half_at(xb2) * (float)(int8_t)xb2[4 + l];
    *acc = fmaf(mul1 * v1 - off1, x1, *acc);
    *acc = fmaf(mul2 * v2 - off2, x2, *acc);
}

__global__ void __launch_bounds__(256, 2) matvec_q5k(const unsigned char *w,
                                                     const unsigned char *x_q8, float *y, int k,
                                                     int rows, int ncols, int w_blocks,
                                                     size_t x_row_bytes) {
    const int warp = (int)(threadIdx.x >> 5);
    const int lane = (int)(threadIdx.x & 31);
    const int row = blockIdx.x * kWarps + warp;
    if (row >= rows) {
        return;
    }
    const unsigned char *wrow = w + (size_t)row * w_blocks * kBlockBytes;
    for (int col = 0; col < ncols; ++col) {
        const unsigned char *xrow = x_q8 + (size_t)col * x_row_bytes;
        float acc = 0.0f;
        for (int b = 0; b < w_blocks; ++b) {
            const unsigned char *block = wrow + (size_t)b * kBlockBytes;
            const float d = half_at(block);
            const float dmin = half_at(block + 2);
            float mult[8];
            float offs[8];
            q5k_group_factors(block + 4, mult, offs);
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                chunk_terms(block, xrow, b, c, lane, d, dmin, mult, offs, &acc);
            }
        }
        /* Warp reduce the 32 lane partials; every lane holds the sum after the loop. */
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc += __shfl_down_sync(0xffffffffu, acc, off);
        }
        if (lane == 0) {
            y[(size_t)col * rows + row] = acc;
        }
    }
}

bool aligned(const void *p, size_t alignment) { return (uintptr_t)p % alignment == 0; }

bool apart(const void *a, size_t a_bytes, const void *b, size_t b_bytes) {
    return (uintptr_t)a + a_bytes <= (uintptr_t)b || (uintptr_t)b + b_bytes <= (uintptr_t)a;
}

}  // namespace

extern "C" int inference_cuda_matvec_q5k_launch(const inference_cuda_matvec_q5k_args *args,
                                                size_t x_row_bytes, void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || stream == NULL || x_row_bytes == 0) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const int k = args->k;
        const int rows = args->rows;
        const int ncols = args->ncols;
        if (k <= 0 || k % kQK != 0 || rows <= 0 || ncols <= 0) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const size_t w_bytes = (size_t)rows * (size_t)(k / kQK) * kBlockBytes;
        const size_t x_bytes = x_row_bytes * (size_t)ncols;
        const size_t y_bytes = (size_t)rows * (size_t)ncols * sizeof(float);
        if (!aligned(args->w, 4) || !aligned(args->x_q8, 4) || !aligned(args->y, 4)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        if (!apart(args->w, w_bytes, args->y, y_bytes) || !apart(args->x_q8, x_bytes, args->y, y_bytes)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const int blocks = (rows + kWarps - 1) / kWarps;
        matvec_q5k<<<blocks, kWarps * 32, 0, (cudaStream_t)stream>>>(
            (const unsigned char *)args->w, (const unsigned char *)args->x_q8, args->y, k, rows,
            ncols, k / kQK, x_row_bytes);
        Failure failure;
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
