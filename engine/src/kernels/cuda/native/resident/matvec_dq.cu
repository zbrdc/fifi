/*
 * The decode dq matvec: y = W . x over the file's native Q4_K and Q6_K blocks, with x one f32
 * activation row. No vendored includes, no repack and no q8_1 quantize: the weight side is the
 * native block decoded per value, the activation is read as it lies, and the arithmetic is
 * plain f32.
 *
 * Geometry: one warp serves four weight rows, eight lanes per row (a block of 128 threads
 * serves 16 rows). Lane l = lane mod 8 owns values [32 l, 32 l + 32) of every K block, so its
 * 32 x values are contiguous (eight float4 loads, 16-byte aligned), and the four lanes at one
 * position of the four rows ask for the same x values in one warp load: every x value is read
 * once per four weight rows, with no shared memory.
 *
 * The block layouts are the CPU oracle's (weights/gguf_dequant.rs). Every code goes to f32
 * through its mantissa (code_f, code_at), which runs on the FMA pipe where __int2float_rn
 * costs a convert.
 *
 * The weight bytes are read as 16-byte lines (load_line) or as one 16-bit unit (load_half),
 * through typed pointers: a load through a byte pointer comes out as byte loads, one warp
 * instruction for a few useful bytes. A Q4_K block is 16-byte aligned, so its 32 code bytes are
 * two lines. A Q6_K block is 210 bytes at an even address and a lane's 32-byte window is
 * 16-byte aligned only where the block is, so load_window reads the three lines that cover the
 * window and moves it onto word boundaries with one byte perm a word.
 *
 * Every exported function has C linkage, takes and returns C types only, and wraps its body
 * in try/catch, like matvec_q.cu.
 */
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stddef.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"
#include "resident/matvec_dq.h"
#include "resident/k4_scales.cuh"

namespace {

const int kQK = 256;         /* values per K block */
const int kRowsPerWarp = 4;  /* weight rows one warp serves */
const int kLanesPerRow = 8;  /* lanes per weight row: 32 values of a K block each */
const int kWarps = 4;        /* warps per CTA: 128 threads, 16 rows */

/* The 16 bytes at a 16-byte aligned address, and the two at an even one. */
__device__ __forceinline__ uint4 load_line(const unsigned char *p) {
    return *reinterpret_cast<const uint4 *>(p);
}

__device__ __forceinline__ unsigned short load_half(const unsigned char *p) {
    return *reinterpret_cast<const unsigned short *>(p);
}

/* An integer code 0..2^23-1 as an f32 through its mantissa: 0x4B000000 | code is 2^23 + code
 * and the subtraction is exact. */
__device__ __forceinline__ float code_f(unsigned code) {
    return __int_as_float(0x4B000000u | code) - 8388608.0f;
}

/* 2^23 + a code of at most 6 bits held in byte j of `packed`: one byte perm moves the byte
 * under the exponent 0x4B. */
__device__ __forceinline__ float code_at(unsigned packed, int j) {
    return __uint_as_float(__byte_perm(packed, 0x4B000000u, 0x7540u + (unsigned)j));
}

/* The 32 f32 values a lane owns in one K block, at a 16-byte aligned address. */
struct Run {
    float4 v[8];
};

__device__ __forceinline__ Run load_run(const float *x) {
    Run r;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        r.v[i] = *reinterpret_cast<const float4 *>(x + 4 * i);
    }
    return r;
}

__device__ __forceinline__ float run_sum(const Run &r) {
    float4 s = r.v[0];
    #pragma unroll
    for (int i = 1; i < 8; ++i) {
        s.x += r.v[i].x;
        s.y += r.v[i].y;
        s.z += r.v[i].z;
        s.w += r.v[i].w;
    }
    return (s.x + s.y) + (s.z + s.w);
}

/* The 6-bit multiplier and offset of Q4_K sub-block `s` of `head` (the block's first 16 bytes). */
__device__ __forceinline__ void k4_factors(const uint4 &head, int s, float *mult, float *offs) {
    unsigned sc, mn;
    scale_min_k4(head, s, sc, mn);
    *mult = code_f(sc);
    *offs = code_f(mn);
}

/* Q4_K block, 144 bytes: f16 d at +0, f16 dmin at +2, 12 packed factor bytes at +4, 128 code
 * bytes at +16. The 256 values are eight sub-blocks of 32. Sub-block l is one nibble of code
 * unit l / 2 (the low nibble for an even l, the high nibble for an odd one) and has a 6-bit
 * multiplier and offset. Value i of sub-block l weighs d * mult * code - dmin * offs, so a row
 * is the sum over blocks of d * mult * sum(code * x) - dmin * offs * sum(x): the two sums stay
 * apart because d and dmin are per block. */
struct Q4kSide {
    static const bool kOffsetSum = true;
    static const int kBlockBytes = 144;
    float sum_d;
    float sum_m;

    __device__ __forceinline__ void start() {
        sum_d = 0.0f;
        sum_m = 0.0f;
    }

    /* One K block of the lane's row: `raw` is 16-byte aligned (144 = 16 * 9). */
    __device__ __forceinline__ void add(const unsigned char *raw, int l, const Run &x, float sx) {
        const uint4 *unit = reinterpret_cast<const uint4 *>(raw + 16 + 32 * (l >> 1));
        const uint4 u0 = unit[0];
        const uint4 u1 = unit[1];
        const unsigned codes[8] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
        const unsigned nib = 4u * (l & 1);
        float sd = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            /* Four codes, one in the low 4 bits of each byte. */
            const unsigned packed = (codes[i] >> nib) & 0x0F0F0F0Fu;
            const float4 xv = x.v[i];
            sd = fmaf(code_at(packed, 0) - 8388608.0f, xv.x, sd);
            sd = fmaf(code_at(packed, 1) - 8388608.0f, xv.y, sd);
            sd = fmaf(code_at(packed, 2) - 8388608.0f, xv.z, sd);
            sd = fmaf(code_at(packed, 3) - 8388608.0f, xv.w, sd);
        }
        const uint4 head = load_line(raw);
        float mult;
        float offs;
        k4_factors(head, l, &mult, &offs);
        const float d = __half2float(__ushort_as_half((unsigned short)head.x));
        const float dmin = __half2float(__ushort_as_half((unsigned short)(head.x >> 16)));
        sum_d = fmaf(d * mult, sd, sum_d);
        sum_m = fmaf(dmin * offs, sx, sum_m);
    }

    __device__ __forceinline__ void reduce(int bit) {
        sum_d += __shfl_xor_sync(0xffffffffu, sum_d, bit);
        sum_m += __shfl_xor_sync(0xffffffffu, sum_m, bit);
    }

    __device__ __forceinline__ float result() const { return sum_d - sum_m; }
};

/* A signed byte as an f32 through its mantissa: s + 128 is the byte with its top bit flipped. */
__device__ __forceinline__ float signed_f(unsigned byte) {
    return code_f(byte ^ 0x80u) - 128.0f;
}

/* The 32 bytes at `p` as eight words on word boundaries. p is even, but in a 210-byte block it
 * is 16-byte aligned only where the block is: so the three 16-byte lines that cover [p, p + 32)
 * are read, and every word is cut from two neighbours by one byte perm, placed by the window's
 * start offset in its first line: s words (0 to 3) and r bytes (0 or 2). The lines end inside
 * the block's 210 bytes: the last window ends at +192, the third line at +208 at most. They
 * can start up to 14 bytes before the window, in the previous block of the row or in the last
 * block of the previous row, so still inside the allocation. That holds only because the first
 * block of row 0 starts at the 16-byte aligned w, so no line starts before it: this is why
 * the launcher refuses a w that is not 16-byte aligned. */
struct Window {
    unsigned w[8];
};

template <int S>
__device__ __forceinline__ Window cut_window(const unsigned (&v)[12], unsigned sel) {
    Window out;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        out.w[i] = __byte_perm(v[S + i], v[S + i + 1], sel);
    }
    return out;
}

__device__ __forceinline__ Window load_window(const unsigned char *p) {
    const uintptr_t a = (uintptr_t)p & 15u;
    const uint4 *lines = reinterpret_cast<const uint4 *>(p - a);
    const uint4 l0 = lines[0];
    const uint4 l1 = lines[1];
    /* When p is already 16-byte aligned (a == 0, e.g. Q8_0 lane 7 at +240 of a 272-byte
     * super-group), the 32-byte window fits inside l0 and l1 alone; skipping lines[2] avoids an
     * out-of-bounds read past the end of the weight buffer. */
    const uint4 l2 = a != 0u ? lines[2] : make_uint4(0u, 0u, 0u, 0u);
    const unsigned v[12] = {l0.x, l0.y, l0.z, l0.w, l1.x, l1.y, l1.z, l1.w,
                            l2.x, l2.y, l2.z, l2.w};
    const unsigned sel = 0x3210u + 0x2222u * (unsigned)((a >> 1) & 1u);
    switch (a >> 2) {
        case 0: return cut_window<0>(v, sel);
        case 1: return cut_window<1>(v, sel);
        case 2: return cut_window<2>(v, sel);
        default: return cut_window<3>(v, sel);
    }
}

/* Q6_K block, 210 bytes at an even address: ql[128] at +0, qh[64] at +128, 16 signed scales at
 * +192 (one per 16 values), f16 d at +208. The 256 values are two halves of 128, each four
 * quarters of 32. Lane l owns quarter q = l % 4 of half h = l / 4 (values [32 l, 32 l + 32)).
 * Value m of that quarter weighs d * scale * (code - 32) with
 *   code = ((ql[64 h + m + 32 (q & 1)] >> 4 (q >> 1)) & 0xF) | ((qh[32 h + m] >> 2 q & 3) << 4)
 *   scale = scales[8 h + 2 q + m / 16]
 * (the CPU oracle, weights/gguf_dequant.rs q6_k). Word i of each window holds the bytes of
 * values 4 i to 4 i + 3, so their four codes are built in one word and taken out by code_at;
 * the scale joins once per 16 values, when its two words are done. */
struct Q6kSide {
    static const bool kOffsetSum = false;
    static const int kBlockBytes = 210;
    float sum;

    __device__ __forceinline__ void start() { sum = 0.0f; }

    __device__ __forceinline__ void add(const unsigned char *raw, int l, const Run &x, float) {
        const int q = l & 3;
        const int h = l >> 2;
        const unsigned nib = 4u * (unsigned)(q >> 1);
        const unsigned pair = 2u * (unsigned)q;
        const Window lo = load_window(raw + 64 * h + 32 * (q & 1));
        const Window hi = load_window(raw + 128 + 32 * h);
        /* qh's bit pair 2 q of every byte moves to bits 4 and 5 of the byte: one of the two
         * shifts is the one that does it, the other sends its bits past the mask. */
        const unsigned left = (4u - pair) & 31u;
        const unsigned right = (pair - 4u) & 31u;
        const unsigned short sc = load_half(raw + 192 + 8 * h + 2 * q);
        float sd[2] = {0.0f, 0.0f};
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const unsigned low = (lo.w[i] >> nib) & 0x0F0F0F0Fu;
            const unsigned high = ((hi.w[i] << left) | (hi.w[i] >> right)) & 0x30303030u;
            const unsigned packed = low | high;
            const float4 xv = x.v[i];
            sd[i >> 2] = fmaf(code_at(packed, 0) - 8388640.0f, xv.x, sd[i >> 2]);
            sd[i >> 2] = fmaf(code_at(packed, 1) - 8388640.0f, xv.y, sd[i >> 2]);
            sd[i >> 2] = fmaf(code_at(packed, 2) - 8388640.0f, xv.z, sd[i >> 2]);
            sd[i >> 2] = fmaf(code_at(packed, 3) - 8388640.0f, xv.w, sd[i >> 2]);
        }
        const float d = __half2float(__ushort_as_half(load_half(raw + 208)));
        sum = fmaf(d * signed_f(sc & 0xFFu), sd[0], sum);
        sum = fmaf(d * signed_f(sc >> 8), sd[1], sum);
    }

    __device__ __forceinline__ void reduce(int bit) {
        sum += __shfl_xor_sync(0xffffffffu, sum, bit);
    }

    __device__ __forceinline__ float result() const { return sum; }
};

/* Q8_0 super-group of 8 consecutive 34-byte blocks (256 values, 272 = 17 * 16 bytes): each
 * super-group starts 16-byte aligned when w is 16-byte aligned. Lane l (0..7) owns block l at
 * raw + 34 * l: f16 d at +0, 32 signed int8 codes at +2..+34. Flipping the sign bit of each
 * byte maps [-128..127] to [0..255] so code_at + (-8388736.0f) converts on the FMA pipe. */
struct Q80Side {
    static const bool kOffsetSum = false;
    static const int kBlockBytes = 272;
    float sum;

    __device__ __forceinline__ void start() { sum = 0.0f; }

    __device__ __forceinline__ void add(const unsigned char *raw, int l, const Run &x, float) {
        const unsigned char *blk = raw + 34 * l;
        const float d = __half2float(__ushort_as_half(load_half(blk)));
        const Window q = load_window(blk + 2);
        float sd = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const unsigned biased = q.w[i] ^ 0x80808080u;
            const float4 xv = x.v[i];
            sd = fmaf(code_at(biased, 0) - 8388736.0f, xv.x, sd);
            sd = fmaf(code_at(biased, 1) - 8388736.0f, xv.y, sd);
            sd = fmaf(code_at(biased, 2) - 8388736.0f, xv.z, sd);
            sd = fmaf(code_at(biased, 3) - 8388736.0f, xv.w, sd);
        }
        sum = fmaf(d, sd, sum);
    }

    __device__ __forceinline__ void reduce(int bit) {
        sum += __shfl_xor_sync(0xffffffffu, sum, bit);
    }

    __device__ __forceinline__ float result() const { return sum; }
};

template <class Side>
__global__ void __launch_bounds__(kWarps * 32) matvec_dq_kernel(const unsigned char *w,
                                                                const float *x, float *y, int rows,
                                                                int blocks) {
    const int lane = (int)(threadIdx.x & 31u);
    const int row = (int)blockIdx.x * (kWarps * kRowsPerWarp) +
                    (int)(threadIdx.x >> 5) * kRowsPerWarp + lane / kLanesPerRow;
    const int l = lane % kLanesPerRow;
    /* The last block of a row count below a multiple of 16 has rows that are not there: their
     * lanes read x and join the reduction but read no weight and write nothing. */
    const bool live = row < rows;
    const unsigned char *wrow = live ? w + (size_t)row * blocks * Side::kBlockBytes : w;
    Side side;
    side.start();
    for (int b = 0; b < blocks; ++b) {
        const Run xv = load_run(x + (size_t)b * kQK + 32 * l);
        const float sx = Side::kOffsetSum ? run_sum(xv) : 0.0f;
        if (live) {
            side.add(wrow + (size_t)b * Side::kBlockBytes, l, xv, sx);
        }
    }
    /* Eight lanes hold one row: fold the lane index's low three bits. Every lane joins. */
    #pragma unroll
    for (int bit = 1; bit < kLanesPerRow; bit <<= 1) {
        side.reduce(bit);
    }
    if (live && l == 0) {
        y[row] = side.result();
    }
}

bool aligned(const void *p, size_t alignment) { return (uintptr_t)p % alignment == 0; }

bool apart(const void *a, size_t a_bytes, const void *b, size_t b_bytes) {
    return (uintptr_t)a + a_bytes <= (uintptr_t)b || (uintptr_t)b + b_bytes <= (uintptr_t)a;
}

}  // namespace

extern "C" int inference_cuda_matvec_dq_launch(const inference_cuda_matvec_dq_args *args,
                                                void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || stream == NULL) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const int k = args->k;
        const int rows = args->rows;
        const bool q4k = args->type == INFERENCE_CUDA_TYPE_Q4_K;
        const bool q6k = args->type == INFERENCE_CUDA_TYPE_Q6_K;
        const bool q80 = args->type == INFERENCE_CUDA_TYPE_Q8_0;
        if (args->w == NULL || args->x == NULL || args->y == NULL || k <= 0 || k % kQK != 0 ||
            rows <= 0 || !(q4k || q6k || q80)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const size_t blocks = (size_t)k / kQK;
        const int block_bytes =
            q4k ? Q4kSide::kBlockBytes : (q6k ? Q6kSide::kBlockBytes : Q80Side::kBlockBytes);
        const size_t w_bytes = (size_t)rows * blocks * (size_t)block_bytes;
        const size_t x_bytes = (size_t)k * sizeof(float);
        const size_t y_bytes = (size_t)rows * sizeof(float);
        if (!aligned(args->w, 16) || !aligned(args->x, 16) || !aligned(args->y, 4) ||
            !apart(args->y, y_bytes, args->w, w_bytes) ||
            !apart(args->y, y_bytes, args->x, x_bytes)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const unsigned int grid = (unsigned int)(((size_t)rows + 15) / 16);
        if (q4k) {
            matvec_dq_kernel<Q4kSide><<<grid, kWarps * 32, 0, (cudaStream_t)stream>>>(
                (const unsigned char *)args->w, (const float *)args->x, args->y, rows, (int)blocks);
        } else if (q6k) {
            matvec_dq_kernel<Q6kSide><<<grid, kWarps * 32, 0, (cudaStream_t)stream>>>(
                (const unsigned char *)args->w, (const float *)args->x, args->y, rows, (int)blocks);
        } else {
            matvec_dq_kernel<Q80Side><<<grid, kWarps * 32, 0, (cudaStream_t)stream>>>(
                (const unsigned char *)args->w, (const float *)args->x, args->y, rows, (int)blocks);
        }
        Failure failure;
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
