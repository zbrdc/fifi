/*
 * The in-house Q4_K decode matvec, sibling of matvec_q80.cu. No vendored includes: the
 * weights are the file's own 144-byte Q4_K super-blocks as the CPU oracle of
 * resident/matvec_q/tests.rs reads them (an f16 d, an f16 dmin, 12 packed scale bytes, 128
 * code bytes): group g of a block (32 values) takes the low nibbles of the 32 code bytes at
 * 32 * (g / 2) for an even g and the high nibbles for an odd g, and a weight is
 * d * sc * q - dmin * m with the 6-bit scale sc and minimum m of the group (resident/k4_scales.cuh,
 * the one home of the rule). The activation is
 * the shared quantize_q8_1 output (36 bytes per 32 values), and the arithmetic is plain f32.
 *
 * One warp owns one weight row, and one lane owns the groups g, g + 32, ... of the row, so
 * the lane's integer work is exact: A = sum q * xq over a group's 32 values (dp4a on the raw
 * nibbles and the int8 activation quants) and B = sum xq (dp4a against ones), on four
 * independent chains. A group's term is xd * (d * sc * A - dmin * m * B), folded with
 * __fmaf_rn in group order with the g & 32 split, and the reduce is a full xor tree. Like
 * matvec_q80, every column repeats the same lane arithmetics: column j of an 8-column launch
 * is bit-identical to a 1-column launch over the same bytes.
 *
 * Every exported function has C linkage, takes and returns C types only, and wraps its body
 * in try/catch, like matvec_q40.cu.
 */
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stddef.h>
#include <stdint.h>

#include <string.h>

#include "abi.h"
#include "device.h"
#include "resident/k4_scales.cuh"
#include "resident/matvec_q4k.h"

namespace {

const int kQK = 256;         /* values per Q4_K block */
const int kBlockBytes = 144;
const int kGroups = 8;       /* 32-value groups per block */
const int kXBlock = 36;      /* bytes per q8_1 block of 32 values */
const int kWarps = 8;        /* warps per CTA */

__device__ __forceinline__ float half_at(const unsigned char *p) {
    __half raw;
    memcpy(&raw, p, sizeof(raw));
    return __half2float(raw);
}

/* A 32-bit word of the code bytes. The f16 scales of a block put the code at an even offset
 * from w only, so two u16 loads build the int, as in matvec_q80.cu. */
__device__ __forceinline__ int word2(const unsigned char *p) {
    const uint16_t *q = (const uint16_t *)p;
    return (int)q[0] | ((int)q[1] << 16);
}

/* Four 256-thread blocks a streaming processor is the ceiling (1024 threads). The one-column
 * kernel takes 48 registers at any bound; the bound of 4 holds the eight-column kernel to 64
 * registers with no spill (128 at 2), so the multi-column path keeps its four blocks. */
template <int NCols>
__global__ void __launch_bounds__(256, 4) matvec_q4k(const unsigned char *w,
                                                     const unsigned char *x_q8, float *y, int k,
                                                     int rows, size_t x_row_bytes) {
    const int warp = (int)(threadIdx.x >> 5);
    const int lane = (int)(threadIdx.x & 31);
    const int row = blockIdx.x * kWarps + warp;
    if (row >= rows) {
        return;
    }
    const unsigned char *wrow = w + (size_t)row * (size_t)(k / kQK) * kBlockBytes;
    const int groups = (k / kQK) * kGroups;
    /* One lane per 32-value group of the row, striding 32: the nibbles of the group's eight
     * code words meet the group's eight activation words in dp4a on four independent chains,
     * the two integer sums are exact, and the scales fold in once per group. The column count
     * is a template parameter so the accumulators stay in registers; __fmaf_rn pins the fused
     * form. */
    float acc0[NCols] = {}, acc1[NCols] = {};
    for (int g = lane; g < groups; g += 32) {
        const unsigned char *blk = wrow + (size_t)(g / kGroups) * kBlockBytes;
        const int grp = g % kGroups;
        unsigned sc, mn;
        scale_min_k4(blk + 4, grp, sc, mn);
        const unsigned char *code = blk + 16 + 32 * (grp >> 1);
        int q[8];
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            const int word = word2(code + 4 * t);
            q[t] = (grp & 1) ? ((word >> 4) & 0x0F0F0F0F) : (word & 0x0F0F0F0F);
        }
        const float d = half_at(blk), dmin = half_at(blk + 2);
        #pragma unroll
        for (int col = 0; col < NCols; ++col) {
            const unsigned char *act = x_q8 + (size_t)col * x_row_bytes + (size_t)g * kXBlock;
            const int *a = (const int *)(act + 4);
            int d0 = 0, d1 = 0, s0 = 0, s1 = 0;
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                d0 = __dp4a(q[t], a[t], d0);
                s0 = __dp4a(a[t], 0x01010101, s0);
                d1 = __dp4a(q[t + 4], a[t + 4], d1);
                s1 = __dp4a(a[t + 4], 0x01010101, s1);
            }
            const float term =
                __fmaf_rn(__fmul_rn(d, (float)sc), (float)(d0 + d1),
                          -__fmul_rn(__fmul_rn(dmin, (float)mn), (float)(s0 + s1)));
            if (g & 32) {
                acc1[col] = __fmaf_rn(half_at(act), term, acc1[col]);
            } else {
                acc0[col] = __fmaf_rn(half_at(act), term, acc0[col]);
            }
        }
    }
    #pragma unroll
    for (int col = 0; col < NCols; ++col) {
        float acc = acc0[col] + acc1[col];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            acc += __shfl_xor_sync(0xffffffffu, acc, o);
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

extern "C" int inference_cuda_matvec_q4k_launch(const inference_cuda_matvec_q4k_args *args,
                                                size_t x_row_bytes, void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || stream == NULL || x_row_bytes == 0) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const int k = args->k;
        const int rows = args->rows;
        const int ncols = args->ncols;
        if (k <= 0 || k % kQK != 0 || rows <= 0 || ncols <= 0 || ncols > 8) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const size_t w_bytes = (size_t)rows * (size_t)(k / kQK) * kBlockBytes;
        const size_t x_bytes = x_row_bytes * (size_t)ncols;
        const size_t y_bytes = (size_t)rows * (size_t)ncols * sizeof(float);
        /* The f16 scales of a 144-byte block start at an even offset from w, so 2 bytes of
         * weight alignment are enough; the q8_1 rows and the f32 out want 4. */
        if (!aligned(args->w, 2) || !aligned(args->x_q8, 4) || !aligned(args->y, 4)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        if (!apart(args->w, w_bytes, args->y, y_bytes) ||
            !apart(args->x_q8, x_bytes, args->y, y_bytes)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const int blocks = (rows + kWarps - 1) / kWarps;
#define LAUNCH_COLS(N) \
    case N: matvec_q4k<N><<<blocks, kWarps * 32, 0, (cudaStream_t)stream>>>( \
        (const unsigned char *)args->w, (const unsigned char *)args->x_q8, args->y, k, rows, \
        x_row_bytes); break
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
        Failure failure;
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
