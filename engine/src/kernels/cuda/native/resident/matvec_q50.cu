/*
 * The in-house Q5_0 decode matvec, sibling of matvec_q80.cu and matvec_q4k.cu. No vendored
 * includes: the weights are the file's own 22-byte Q5_0 blocks as the CPU oracle of
 * resident/matvec_q/tests.rs reads them (an f16 d, a u32 qh and 16 code bytes): value i takes
 * the low nibble of code byte i for i < 16 and the high nibble of code byte i - 16 otherwise,
 * with bit i of qh as its fifth bit, so q runs 0..31 and a weight is d * (q - 16). The
 * activation is the shared quantize_q8_1 output (36 bytes per 32 values: an f16 d, an f16 s,
 * 32 int8 quants), and the arithmetic is plain f32.
 *
 * A block's term is the vendored kernel's exact form, which the test file's reference mirrors:
 * d * (xd * sum(q * xq) - 16 * s), with s the activation block's stored raw input sum, not xd
 * times the quantized sum. The exact quantized form would differ from every served Q5_0
 * output by quantization-level noise, so it is not used.
 *
 * One warp owns one weight row, and one lane owns the blocks b, b + 32, ... of the row, so
 * the lane's integer sum is exact: sum(q * xq) is dp4a on the five-bit codes as unsigned
 * bytes and the int8 activation quants, on four independent chains. The term folds with
 * __fmaf_rn in block order with the b & 32 split, and the reduce is a full xor tree. Like
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
#include "resident/matvec_q50.h"

namespace {

const int kQK = 32;          /* values per Q5_0 block */
const int kBlockBytes = 22;
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

/* 0x10 in byte r when bit r of the four is set: the fifth bit of four codes. */
__device__ __forceinline__ int fifth_bits(int bits) {
    return ((bits & 1) << 4) | ((bits & 2) << 11) | ((bits & 4) << 18) | ((bits & 8) << 25);
}

/* Four 256-thread blocks a streaming processor is the ceiling (1024 threads). The one-column
 * kernel takes 52 registers at the bounds of 2 and 4; the bound of 4 holds the eight-column
 * kernel to 64 registers with no spill (128 at 2), so the multi-column path keeps its four
 * blocks. */
template <int NCols>
__global__ void __launch_bounds__(256, 4) matvec_q50(const unsigned char *w,
                                                     const unsigned char *x_q8, float *y, int k,
                                                     int rows, size_t x_row_bytes) {
    const int warp = (int)(threadIdx.x >> 5);
    const int lane = (int)(threadIdx.x & 31);
    const int row = blockIdx.x * kWarps + warp;
    if (row >= rows) {
        return;
    }
    const unsigned char *wrow = w + (size_t)row * (size_t)(k / kQK) * kBlockBytes;
    const int blocks = k / kQK;
    float acc0[NCols] = {}, acc1[NCols] = {};
    for (int b = lane; b < blocks; b += 32) {
        const unsigned char *blk = wrow + (size_t)b * kBlockBytes;
        const int qh = word2(blk + 2);
        /* q[t] holds codes 4t..4t+3 as unsigned bytes: the low nibbles of the first four code
         * words for t < 4, the high nibbles for t >= 4, each with its fifth bit from qh. */
        int q[8];
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            const int code = word2(blk + 6 + 4 * (t & 3));
            const int nibble = (t < 4) ? (code & 0x0F0F0F0F) : ((code >> 4) & 0x0F0F0F0F);
            q[t] = nibble | fifth_bits((qh >> (4 * t)) & 0xF);
        }
        const float d = half_at(blk);
        #pragma unroll
        for (int col = 0; col < NCols; ++col) {
            const unsigned char *act = x_q8 + (size_t)col * x_row_bytes + (size_t)b * kXBlock;
            const int *a = (const int *)(act + 4);
            int s0 = __dp4a(q[0], a[0], 0);
            int s1 = __dp4a(q[1], a[1], 0);
            int s2 = __dp4a(q[2], a[2], 0);
            int s3 = __dp4a(q[3], a[3], 0);
            s0 = __dp4a(q[4], a[4], s0);
            s1 = __dp4a(q[5], a[5], s1);
            s2 = __dp4a(q[6], a[6], s2);
            s3 = __dp4a(q[7], a[7], s3);
            const float inner =
                __fmaf_rn(-16.0f, half_at(act + 2),
                          __fmul_rn(half_at(act), (float)((s0 + s1) + (s2 + s3))));
            if (b & 32) {
                acc1[col] = __fmaf_rn(d, inner, acc1[col]);
            } else {
                acc0[col] = __fmaf_rn(d, inner, acc0[col]);
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

extern "C" int inference_cuda_matvec_q50_launch(const inference_cuda_matvec_q50_args *args,
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
        /* The f16 d of a 22-byte block starts at an even offset from w, so 2 bytes of weight
         * alignment are enough; the q8_1 rows and the f32 out want 4. */
        if (!aligned(args->w, 2) || !aligned(args->x_q8, 4) || !aligned(args->y, 4)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        if (!apart(args->w, w_bytes, args->y, y_bytes) ||
            !apart(args->x_q8, x_bytes, args->y, y_bytes)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const int blocks = (rows + kWarps - 1) / kWarps;
#define LAUNCH_COLS(N) \
    case N: matvec_q50<N><<<blocks, kWarps * 32, 0, (cudaStream_t)stream>>>( \
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
