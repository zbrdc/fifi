/*
 * The per-type block decoders of the in-house dequantizing GEMM (gemm_dq.cu). A decoder is a
 * struct with the block geometry of its GGUF type and `decode_chunk`, which writes the 64
 * dequantized halves of one 64-value slice (chunk) of one on-disk block, straight into the
 * shared-memory weight tile. The layouts are the GGUF block layouts, read out in this
 * engine's own structure (the CPU oracle that checks them is weights/gguf_dequant.rs).
 *
 * A new quant type adds a decoder here, not a kernel: gemm_dq.cu is generic over the decoder.
 */
#ifndef INFERENCE_CUDA_PREFILL_GEMM_DQ_DECODERS_CUH
#define INFERENCE_CUDA_PREFILL_GEMM_DQ_DECODERS_CUH

#include <cuda_fp16.h>

#include <stdint.h>

/* The lookup tables of the codebook types ride by-value kernel arguments (never a
 * __constant__ definition: native/constant_bank_guard.rs keeps those out of the tree, and its
 * comment says why). */
#include "resident/gguf_decode.cuh"

namespace gemm_dq {

__device__ __forceinline__ float half_at(const unsigned char *p) {
    __half raw;
    memcpy(&raw, p, sizeof(raw));
    return __half2float(raw);
}

/* Q5_K block layout (176 bytes for 256 values, split into 8 groups of 32): an f16 value
 * scale at +0, an f16 offset scale at +2, 12 packed 6-bit group factors at +4, 32 high-bit
 * bytes at +16, 128 4-bit code bytes at +48. Group g carries a 6-bit multiplier and 6-bit
 * offset, and value i of the block weighs
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

/* Chunk c (0..3) of a Q5_K block carries values [64c, 64c+64): the low code nibbles of
 * bytes [48+32c, 48+32c+32) with high bit 2c, then the high nibbles with high bit 2c+1,
 * each half against its own group pair from q5k_group_factors. A chunk is a quarter block,
 * so the row's chunk-c blocks start at block c/4. */
struct Q5kDecoder {
    static const int kQK = 256;
    static const int kBlockBytes = 176;
    static const int kChunkBlocks = 1;
    static const int kTabWords = 4;
    static const int kHostTabWords = 0;
    static const int kParts = 1;

    __device__ static int chunk_block_off(int c) { return c >> 2; }

    /* c is the raw 64-value chunk index along k (0..k/64); the slice inside this block is
     * c mod 4, because four chunks share one 256-value block. */
    __device__ static void decode_chunk(const unsigned char *block, int c, __half *dst,
                                             const uint32_t *, int) {
        c &= 3;
        const float d = half_at(block);
        const float dmin = half_at(block + 2);
        float mult[8];
        float offs[8];
        q5k_group_factors(block + 4, mult, offs);
        const float mul1 = d * mult[2 * c];
        const float off1 = dmin * offs[2 * c];
        const float mul2 = d * mult[2 * c + 1];
        const float off2 = dmin * offs[2 * c + 1];
        const unsigned bit1 = 1u << (2 * c);
        const unsigned bit2 = 2u << (2 * c);
        const unsigned char *high = block + 16;
        const unsigned char *codes = block + 48 + 32 * c;
        for (int l = 0; l < 32; ++l) {
            const unsigned char code = codes[l];
            const unsigned char h = high[l];
            const float v1 = (float)((code & 0x0f) + ((h & bit1) ? 16 : 0));
            const float v2 = (float)((code >> 4) + ((h & bit2) ? 16 : 0));
            dst[l] = __float2half(fmaf(mul1, v1, -off1));
            dst[32 + l] = __float2half(fmaf(mul2, v2, -off2));
        }
    }
};

/* Q4_K block layout (144 bytes for 256 values, 8 groups of 32): an f16 value scale at +0, an
 * f16 offset scale at +2, the same 12 packed 6-bit group factors as Q5_K at +4, and 128 code
 * bytes at +16 whose byte j holds the codes of values j (low nibble) and j + 128 (high
 * nibble). Value i of group g weighs (value_scale * mult[g]) * code - offset_scale * offs[g]. */
struct Q4kDecoder {
    static const int kQK = 256;
    static const int kBlockBytes = 144;
    static const int kChunkBlocks = 1;
    static const int kTabWords = 4;
    static const int kHostTabWords = 0;
    static const int kParts = 1;

    __device__ static int chunk_block_off(int c) { return c >> 2; }

    /* Chunk c (0..3) carries values [64c, 64c+64): the low nibbles of bytes [16+32c,
     * 16+32c+32) against group 2c, then the high nibbles against group 2c+1. */
    __device__ static void decode_chunk(const unsigned char *block, int c, __half *dst,
                                             const uint32_t *, int) {
        c &= 3;
        const float d = half_at(block);
        const float dmin = half_at(block + 2);
        float mult[8];
        float offs[8];
        q5k_group_factors(block + 4, mult, offs);
        const float mul1 = d * mult[2 * c];
        const float off1 = dmin * offs[2 * c];
        const float mul2 = d * mult[2 * c + 1];
        const float off2 = dmin * offs[2 * c + 1];
        const unsigned char *codes = block + 16 + 32 * c;
        for (int l = 0; l < 32; ++l) {
            const unsigned char code = codes[l];
            dst[l] = __float2half(fmaf(mul1, (float)(code & 0x0f), -off1));
            dst[32 + l] = __float2half(fmaf(mul2, (float)(code >> 4), -off2));
        }
    }
};

/* Q6_K block layout (210 bytes for 256 values, 16 groups of 16): 128 code bytes at +0 (the
 * low 4 bits of values [0,128), then of [128,256)), 32 bytes at +128 holding bits 4-5 of value
 * l and bits 6-7 of value l+16 of each 32-value quarter q at byte 32q + l, 16 signed 8-bit
 * group scales at +192 and an f16 value scale at +208. Value i weighs value_scale *
 * scales[i / 16] * (code - 32). */
struct Q6kDecoder {
    static const int kQK = 256;
    static const int kBlockBytes = 210;
    static const int kChunkBlocks = 1;
    static const int kTabWords = 4;
    static const int kHostTabWords = 0;
    static const int kParts = 1;

    __device__ static int chunk_block_off(int c) { return c >> 2; }

    /* Chunk c (0..3) carries values [64c, 64c+64): the low or the high nibbles of the c/2-th
     * 128-value half, the second chunk of a half reading the high nibbles. */
    __device__ static void decode_chunk(const unsigned char *block, int c, __half *dst,
                                             const uint32_t *, int) {
        c &= 3;
        const int half = c >> 1;
        const int nibble = c & 1;
        const float d = half_at(block + 208);
        const unsigned char *ql = block + 64 * half;
        const unsigned char *qh = block + 128 + 32 * half;
        const signed char *scales =
            reinterpret_cast<const signed char *>(block + 192 + 8 * half + 4 * nibble);
        const unsigned shift = 4 * nibble;
        for (int l = 0; l < 32; ++l) {
            const int is = l >> 4;
            const unsigned h = qh[l];
            const int lo1 = nibble ? (ql[l] >> 4) : (ql[l] & 0x0f);
            const int lo2 = nibble ? (ql[l + 32] >> 4) : (ql[l + 32] & 0x0f);
            const float q1 = (float)(lo1 | (int)((h >> shift) & 3) << 4) - 32.0f;
            const float q2 = (float)(lo2 | (int)((h >> (shift + 2)) & 3) << 4) - 32.0f;
            dst[l] = __float2half(d * (float)scales[is] * q1);
            dst[32 + l] = __float2half(d * (float)scales[is + 2] * q2);
        }
    }
};

/* Q4_0 block layout (18 bytes for 32 values): an f16 scale d at +0, then 16 code bytes at
 * +2 whose byte j holds the codes of values j (low nibble) and j + 16 (high nibble); value i
 * of the block is d * (code_i - 8). A 64-value chunk is exactly two whole blocks, so the
 * row's chunk-c blocks start at block 2c. */
struct Q40Decoder {
    static const int kQK = 32;
    static const int kBlockBytes = 18;
    static const int kChunkBlocks = 2;
    static const int kTabWords = 4;
    static const int kHostTabWords = 0;
    static const int kParts = 1;

    __device__ static int chunk_block_off(int c) { return 2 * c; }

    /* A block's 32 values, in GGUF order: value j (0..16) is the LOW nibble of
     * qs[j], value j + 16 the HIGH nibble (the layout weights/gguf_dequant.rs
     * q4_0 decodes and the ggml reference pins). */
    __device__ static void decode_chunk(const unsigned char *chunk, int c, __half *dst,
                                             const uint32_t *, int) {
        (void)c;
        for (int b = 0; b < 2; ++b) {
            const unsigned char *blk = chunk + 18 * b;
            const float d = half_at(blk);
            for (int j = 0; j < 16; ++j) {
                const unsigned char code = blk[2 + j];
                dst[32 * b + j] = __float2half(d * (float)((code & 0x0f) - 8));
                dst[32 * b + 16 + j] = __float2half(d * (float)((code >> 4) - 8));
            }
        }
    }
};

/* Q3_K block layout (110 bytes for 256 values), per the CPU oracle of
 * weights/gguf_dequant.rs (the ggml reference `dequantize_row_q3_K`): 32
 * hmask bytes at +0, 64 qs bytes at +32, 12 packed 6-bit sub-scale bytes at
 * +96, the f16 super scale at +108. A 256-value block is two 128-value
 * halves; each half reads qs[32*half..) with lane j (0..4) at bit pair 2*j,
 * groups of 16 at +16, and its hmask bit 4*half + j of hmask[r] (set
 * subtracts 0, clear 4); the
 * 16 sub-scales come from the aux shuffle (see `scale_of`) biased by 32. */
struct Q3kDecoder {
    static const int kQK = 256;
    static const int kBlockBytes = 110;
    static const int kChunkBlocks = 1;
    static const int kTabWords = 4;
    static const int kHostTabWords = 0;
    static const int kParts = 1;

    __device__ static int chunk_block_off(int c) { return c >> 2; }

    /* One of the 16 packed 6-bit sub-scales of the ggml aux shuffle (the CPU
     * oracle's aux[0..3] expansion, addressed directly): scale `is` is byte
     * is&3 of row is>>2, low nibble of ks[(row&1)*4 + byte] at shift
     * 4*(row>>1), plus the 2-bit high field of ks[8+byte] at shift 2*row. */
    __device__ static int scale_of(const unsigned char *ks, int is) {
        const int row = is >> 2;
        const int byte = is & 3;
        const int lo = (ks[(row & 1) * 4 + byte] >> (4 * (row >> 1))) & 0x0f;
        const int hi = ((ks[8 + byte] >> (2 * row)) & 0x03) << 4;
        return lo | hi;
    }

    /* Chunk c carries block values [64c, 64c + 64) in the oracle's output
     * order: half = c >> 1 owns qs[32*half..); within the half, lane j covers
     * 32 values (qs shift 2*j, hmask bit 4*half + j: one mask byte carries
     * all eight lanes of both halves), groups of 16 at qs+16 and hm+16; a
     * CLEAR hmask bit subtracts 4. */
    __device__ static void decode_chunk(const unsigned char *block, int c, __half *dst,
                                             const uint32_t *, int) {
        c &= 3;
        const float d = half_at(block + 108);
        const unsigned char *hm = block;
        const unsigned char *qs = block + 32 + 32 * (c >> 1);
        const unsigned char *ks = block + 96;
        #pragma unroll
        for (int s = 0; s < 64; ++s) {
            const int j = 2 * (c & 1) + (s >> 5);
            const int g2 = (s >> 4) & 1;
            const int l = s & 15;
            const int qv =
                (int)((qs[16 * g2 + l] >> (2 * j)) & 3) -
                ((hm[16 * g2 + l] & (1 << (4 * (c >> 1) + j))) == 0 ? 4 : 0);
            const float dl = d * (float)(scale_of(ks, (c >> 1) * 8 + 2 * j + g2) - 32);
            dst[s] = __float2half(dl * (float)qv);
        }
    }
};

/* The window decoders of the in-house codebook matvec (resident/gguf_decode.cuh) as GEMM
 * decoders: a 64-value chunk is two 32-value windows. For the 256-value blocks (and Q2_0's
 * 64-value block) the chunk is a slice of one block and windows 2 * (c mod (windows / 2)) and
 * the next; IQ4_NL's 32-value blocks put two whole blocks in a chunk. The lookup table of the
 * type is the shared-memory copy the kernel stages from its by-value argument. */
template <class T>
struct WindowDecoder {
    static const int kQK = 32 * T::kWinPerBlock;
    static const int kBlockBytes = T::kBlockBytes;
    static const int kChunkBlocks = T::kWinPerBlock >= 2 ? 1 : 2;
    static const int kTabWords = T::kTabWords;
    static const int kHostTabWords = T::kHostTabWords;
    static const int kParts = 2;
    /* The raw chunk is staged in 16-byte words only when its size is a multiple of 16, which
     * needs a 16-byte aligned weight; the window types stage bytes and need two. */
    static_assert(((T::kWinPerBlock >= 2 ? 1 : 2) * T::kBlockBytes) % 16 != 0,
                  "a window type whose chunk is a multiple of 16 bytes would need 16-byte alignment");

    __device__ static int chunk_block_off(int c) {
        return T::kWinPerBlock >= 2 ? c / (T::kWinPerBlock / 2) : 2 * c;
    }

    __device__ static void write_window(const gguf::Win &wn, __half *dst) {
        #pragma unroll
        for (int j = 0; j < 32; j += 2) {
            *reinterpret_cast<__half2 *>(dst + j) = __floats2half2_rn(
                gguf::window_value<T>(wn, j), gguf::window_value<T>(wn, j + 1));
        }
    }

    __device__ static void decode_chunk(const unsigned char *chunk, int c, __half *dst,
                                        const uint32_t *tab, int h) {
        gguf::Win wn;
        if constexpr (T::kWinPerBlock >= 2) {
            const int first = 2 * (c % (T::kWinPerBlock / 2));
            T::decode(T::load(chunk, first + h), tab, wn);
        } else {
            T::decode(T::load(chunk + h * T::kBlockBytes, 0), tab, wn);
        }
        write_window(wn, dst + 32 * h);
    }
};

}  // namespace gemm_dq

#endif /* INFERENCE_CUDA_PREFILL_GEMM_DQ_DECODERS_CUH */
