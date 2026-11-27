/*
 * The in-house Q6_K decode matvec, sibling of matvec_q80.cu, matvec_q4k.cu and matvec_q50.cu.
 * No vendored includes: the weights are the file's own 210-byte Q6_K super-blocks as the CPU
 * oracle of resident/matvec_q/tests.rs reads them: 128 ql bytes, 64 qh bytes, 16 signed
 * scales and an f16 d at 208. Value i of a block sits in half i / 128 and quarter
 * (i % 128) / 32: its low four bits are the low nibble (quarters 0 and 1) or the high nibble
 * (quarters 2 and 3) of ql byte 64 * half + i % 32 + 32 * (quarter & 1), its high two bits are
 * bits 2 * quarter and 2 * quarter + 1 of qh byte 128 + 32 * half + i % 32, and its weight is
 * d * scales[i / 16] * (q - 32). The activation is the shared quantize_q8_1 output (36 bytes
 * per 32 values: an f16 d, an f16 s, 32 int8 quants), and the arithmetic is plain f32.
 *
 * The form is the exact integer one of the vendored kernel and of the test file's reference:
 * per sixteen values, dp4a of the signed bytes q - 32 and the activation quants, times the
 * scale in exact integers, times the activation block's d; the two scales of a 32-value piece
 * sum, and the piece's term takes the block's d.
 *
 * One warp owns one weight row, and one lane owns the 32-value pieces b, b + 32, ... of the
 * row. The term folds with __fmaf_rn in piece order with the b & 32 split, and the reduce is
 * a full xor tree. Every column repeats the same lane arithmetics: column j of an 8-column
 * launch is bit-identical to a 1-column launch over the same bytes.
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
#include "resident/matvec_q6k.h"

namespace {

const int kQK = 256;         /* values per Q6_K super-block */
const int kBlockBytes = 210;
const int kPiece = 32;       /* values per lane piece: one q8_1 block */
const int kXBlock = 36;      /* bytes per q8_1 block of 32 values */
const int kWarps = 8;        /* warps per CTA */

__device__ __forceinline__ float half_at(const unsigned char *p) {
    __half raw;
    memcpy(&raw, p, sizeof(raw));
    return __half2float(raw);
}

/* A 32-bit word at an even address, as in matvec_q80.cu: two u16 loads build the int. */
__device__ __forceinline__ int word2(const unsigned char *p) {
    const uint16_t *q = (const uint16_t *)p;
    return (int)q[0] | ((int)q[1] << 16);
}

/* Four 256-thread blocks a streaming processor is the ceiling (1024 threads). The one-column
 * kernel takes 64 registers at the bounds of 2 and 4; the bound of 4 holds the eight-column
 * kernel to 64 registers with no spill (127 at 2), so the multi-column path keeps its four
 * blocks. */
template <int NCols>
__global__ void __launch_bounds__(256, 4) matvec_q6k(const unsigned char *w,
                                                     const unsigned char *x_q8, float *y, int k,
                                                     int rows, size_t x_row_bytes) {
    const int warp = (int)(threadIdx.x >> 5);
    const int lane = (int)(threadIdx.x & 31);
    const int row = blockIdx.x * kWarps + warp;
    if (row >= rows) {
        return;
    }
    const unsigned char *wrow = w + (size_t)row * (size_t)(k / kQK) * kBlockBytes;
    const int pieces = k / kPiece;
    float acc0[NCols] = {}, acc1[NCols] = {};
    for (int b = lane; b < pieces; b += 32) {
        const unsigned char *blk = wrow + (size_t)(b >> 3) * kBlockBytes;
        const int piece = b & 7;                /* the 32-value piece of the block */
        const int half = piece >> 2;
        const int quarter = piece & 3;
        const int nib = 4 * (quarter >> 1);     /* the ql nibble with the low four bits */
        const int shift = 2 * quarter;          /* the qh bit pair with the high two */
        const unsigned char *ql = blk + 64 * half + 32 * (quarter & 1);
        const unsigned char *qh = blk + 128 + 32 * half;
        const int sc0 = (signed char)blk[192 + 2 * piece];
        const int sc1 = (signed char)blk[192 + 2 * piece + 1];
        const float d = half_at(blk + 208);
        /* v[t] holds sixteen codes of the piece as the signed bytes q - 32. */
        int v[8];
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            const int lo = (word2(ql + 4 * t) >> nib) & 0x0F0F0F0F;
            const int hi = ((word2(qh + 4 * t) >> shift) & 0x03030303) << 4;
            v[t] = __vsubss4(lo | hi, 0x20202020);
        }
        #pragma unroll
        for (int col = 0; col < NCols; ++col) {
            const unsigned char *act = x_q8 + (size_t)col * x_row_bytes + (size_t)b * kXBlock;
            const int *a = (const int *)(act + 4);
            int is0 = __dp4a(v[0], a[0], 0);
            int is1 = __dp4a(v[4], a[4], 0);
            is0 = __dp4a(v[1], a[1], is0);
            is1 = __dp4a(v[5], a[5], is1);
            is0 = __dp4a(v[2], a[2], is0);
            is1 = __dp4a(v[6], a[6], is1);
            is0 = __dp4a(v[3], a[3], is0);
            is1 = __dp4a(v[7], a[7], is1);
            const float dx = half_at(act);
            const float sumf = __fadd_rn(__fmul_rn(dx, (float)(is0 * sc0)),
                                         __fmul_rn(dx, (float)(is1 * sc1)));
            if (b & 32) {
                acc1[col] = __fmaf_rn(d, sumf, acc1[col]);
            } else {
                acc0[col] = __fmaf_rn(d, sumf, acc0[col]);
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

extern "C" int inference_cuda_matvec_q6k_launch(const inference_cuda_matvec_q6k_args *args,
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
        /* A 210-byte block starts at an even offset from w and keeps its f16 d at 208, so 2
         * bytes of weight alignment are enough; the q8_1 rows and the f32 out want 4. */
        if (!aligned(args->w, 2) || !aligned(args->x_q8, 4) || !aligned(args->y, 4)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        if (!apart(args->w, w_bytes, args->y, y_bytes) ||
            !apart(args->x_q8, x_bytes, args->y, y_bytes)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const int blocks = (rows + kWarps - 1) / kWarps;
#define LAUNCH_COLS(N) \
    case N: matvec_q6k<N><<<blocks, kWarps * 32, 0, (cudaStream_t)stream>>>( \
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
