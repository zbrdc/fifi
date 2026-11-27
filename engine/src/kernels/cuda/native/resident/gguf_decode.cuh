/*
 * Window decoders of the in-house GGUF matvec (matvec_gguf.cu). A window is 32 consecutive
 * weights of one row. Every type here turns one window into the same shape: 32 signed int8
 * weights packed four per int (Win::w) and two f32 scales, s0 for weights 0..15 and s1 for
 * weights 16..31, so that
 *
 *     dequantized weight j  =  (j < 16 ? s0 : s1) * (float)int8_j
 *
 * is the exact f32 value of weights/gguf_dequant.rs (the CPU oracle) for the types whose
 * weights are a scale times an integer. Two types add a term, flagged per type at compile
 * time: kHasAdd (IQ1_S, IQ1_M: the grid value is shifted before scaling, one shift a[g] per
 * group of eight) and kHasMin (Q2_K: a per-16 minimum m0, m1 is subtracted after scaling),
 * so that in general
 *
 *     weight j  =  (s_h * ((float)int8_j + a_g)) - m_h        (h = j / 16, g = j / 8)
 *
 * with a = 0 and m = 0 where the flag is off. The dot product then adds
 * s_h * a_g * (sum of the activation quants of group g) and subtracts m_h * (sum of the
 * activation quants of half h). The matvec multiplies
 * the int8 weights against the int8 activation quants with __dp4a and applies the scales in
 * f32; the device dequant test applies them to the int8 weights directly.
 *
 * The scale expressions use __fmul_rn/__fadd_rn in the oracle's evaluation order, so the
 * compiler cannot contract them and the device value equals the CPU value bit for bit.
 *
 * A type is a struct with
 *   kBlockBytes   bytes of one block on disk
 *   kWinPerBlock  windows per block (1 for the 32-value blocks, 8 for the 256-value blocks)
 *   kTabWords     32-bit words of the lookup table the kernel stages into shared memory
 *   kHostTabWords words the host passes (kTabWords; Q2_0 has no table: the kernel gets a
 *                 4-word zero dummy)
 *   Raw           the bytes of one window as loaded from global memory
 *   load(block, wi)        loads window wi of the block; issues only global loads
 *   decode(raw, tab, win)  decodes without touching global memory; tab is the shared copy of
 *                          the table the host passed by value
 *
 * Lookup tables are never __constant__ storage (see iq4_nl_values.cuh and
 * constant_bank_guard.rs): the host passes the table words from weights/gguf_tables.rs as a
 * by-value kernel argument and the kernel copies them to shared memory once per CTA. The
 * sign byte of the IQ2/IQ3 families is not a table: the oracle's KSIGNS_IQ2XS[i] is i with
 * the parity of its seven bits in bit 7 (pinned by a host test), computed with __popc.
 *
 * Every grid byte of the IQ2 and IQ3 grids is in 1..=62, which the sign step relies on: it
 * negates four packed bytes as (g ^ m) + (m & 0x01010101) with m = 0xFF in the negated bytes,
 * and the +1 never carries out of a byte because (~g) is at most 254. A host test scans the
 * grids for that.
 */
#ifndef INFERENCE_CUDA_RESIDENT_GGUF_DECODE_CUH
#define INFERENCE_CUDA_RESIDENT_GGUF_DECODE_CUH

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "resident/k4_scales.cuh"

#include <stdio.h>

#include <stdint.h>
#include <string.h>

namespace gguf {

/* A failed launch of a kernel that carries a lookup table by value: when the driver stack
 * rejects the argument size (older than CUDA 12.1, or before Volta, the limit is 4 KiB) the
 * runtime reports an invalid value or configuration; say what the argument was. */
static inline void note_argument_failure(cudaError_t err, size_t argument_bytes) {
    if (err == cudaErrorInvalidValue || err == cudaErrorInvalidConfiguration) {
        fprintf(stderr,
                "codebook kernel launch refused (%s): the lookup table rides a %zu-byte kernel "
                "argument, which needs a CUDA 12.1 or newer driver stack on Volta or newer\n",
                cudaGetErrorString(err), argument_bytes);
    }
}

struct Win {
    int w[8];
    float s0;
    float s1;
    float a[4]; /* kHasAdd types: the shift of each group of eight */
    float m0;   /* kHasMin types: the minima of weights 0..15 and 16..31 */
    float m1;
};

/* The exact f32 weight j of a decoded window, in the oracle's operation order. */
template <class T>
__device__ __forceinline__ float window_value(const Win &wn, int j) {
    const int v = (int)(signed char)((wn.w[j >> 2] >> (8 * (j & 3))) & 0xFF);
    float x = (float)v;
    if constexpr (T::kHasAdd) {
        x = __fadd_rn(x, wn.a[j >> 3]);
    }
    x = __fmul_rn(j < 16 ? wn.s0 : wn.s1, x);
    if constexpr (T::kHasMin) {
        x = __fsub_rn(x, j < 16 ? wn.m0 : wn.m1);
    }
    return x;
}

/* The lookup table of a type as a by-value kernel argument (see the file comment). */
template <int N>
struct __align__(16) Tab {
    uint32_t w[N];
};

/* Copies the by-value table to shared memory: thread t loads and stores the 16-byte chunks
 * t, t + Threads, ... The loop is fully unrolled (the chunk count is a compile-time
 * constant), so a thread has all its constant-bank loads in flight before the first store. */
template <int N, int Threads>
__device__ __forceinline__ void stage_table(const Tab<N> &t, uint32_t *sm) {
    const int chunks = N / 4;
#pragma unroll
    for (int c = (int)threadIdx.x; c < chunks; c += Threads) {
        *reinterpret_cast<uint4 *>(&sm[4 * c]) = *reinterpret_cast<const uint4 *>(&t.w[4 * c]);
    }
}

/* The GGUF tensor type ids the in-house matvec serves. */
enum {
    kIq2Xxs = 16,
    kIq2Xs = 17,
    kIq3Xxs = 18,
    kIq4Nl = 20,
    kIq3S = 21,
    kIq2S = 22,
    kIq4Xs = 23,
    kQ2_0 = 42,
    kQ4_0 = 2,
    kQ5_0 = 6,
    kQ8_0 = 8,
    kQ4K = 12,
    kQ6K = 14,
    kQ3K = 11,
    kQ5K = 13,
    kQ2K = 10,
    kIq1S = 19,
    kIq1M = 29,
};

__device__ __forceinline__ unsigned rd16(const unsigned char *p) {
    unsigned short v;
    memcpy(&v, p, sizeof(v));
    return v;
}

/* Little-endian u32 from a 2-byte-aligned address. */
__device__ __forceinline__ unsigned rd32(const unsigned char *p) {
    return rd16(p) | (rd16(p + 2) << 16);
}

__device__ __forceinline__ float rd_half(const unsigned char *p) {
    __half h;
    memcpy(&h, p, sizeof(h));
    return __half2float(h);
}

/* The oracle's d * (0.5 + n) * q, evaluated left to right without contraction. */
__device__ __forceinline__ float half_step(float d, unsigned n, float q) {
    return __fmul_rn(__fmul_rn(d, __fadd_rn(0.5f, (float)n)), q);
}

/* Bytes 0xFF where bit j (0..3) of nib is set. */
__device__ __forceinline__ unsigned sign_mask4(unsigned nib) {
    return ((nib * 0x00204081u) & 0x01010101u) * 0xFFu;
}

/* The four packed bytes of g, negated where the sign mask m is set. */
__device__ __forceinline__ int apply_signs(unsigned g, unsigned nib) {
    const unsigned m = sign_mask4(nib);
    return (int)((g ^ m) + (m & 0x01010101u));
}

/* The eight sign bits of a 7-bit sign index: bit 7 is the parity of the seven bits. */
__device__ __forceinline__ unsigned sign_byte(unsigned s7) {
    return s7 | ((__popc(s7) & 1u) << 7);
}

/* Four table entries from four nibbles. sel holds nibble j in bits 4j..4j+3 (j = 0..3), t0..t3
 * the sixteen table bytes as four words. Output byte j is table byte nibble_j. PRMT reads
 * its selector as four nibbles whose low three bits pick one of eight source bytes, so the
 * first two PRMTs resolve the low three bits against table bytes 0..7 and 8..15, and the last
 * picks between the two results by bit 3. */
__device__ __forceinline__ int lookup16(unsigned sel, unsigned t0, unsigned t1, unsigned t2,
                                        unsigned t3) {
    const unsigned low3 = sel & 0x7777u;
    const unsigned a = __byte_perm(t0, t1, low3);
    const unsigned b = __byte_perm(t2, t3, low3);
    const unsigned pick = ((sel >> 1) & 0x4444u) | 0x3210u;
    return (int)__byte_perm(a, b, pick);
}

/* The four low nibbles of the four bytes of q, as one 16-bit selector (byte i -> nibble i). */
__device__ __forceinline__ unsigned low_nibbles(unsigned q) {
    const unsigned x = q & 0x0F0F0F0Fu;
    const unsigned t = (x | (x >> 4)) & 0x00FF00FFu;
    return (t | (t >> 8)) & 0xFFFFu;
}

/* Windows of 16 code bytes holding two nibbles each (IQ4_NL, IQ4_XS): weight j (0..15) is
 * the table entry of the low nibble of byte j, weight 16 + j the entry of its high nibble. */
__device__ __forceinline__ void decode_nibbles(const unsigned (&q)[4], const uint32_t *tab,
                                               Win &out) {
    const unsigned t0 = tab[0], t1 = tab[1], t2 = tab[2], t3 = tab[3];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        out.w[r] = lookup16(low_nibbles(q[r]), t0, t1, t2, t3);
        out.w[4 + r] = lookup16(low_nibbles(q[r] >> 4), t0, t1, t2, t3);
    }
}

/* IQ4_NL: 18-byte block, f16 d then 16 code bytes. */
struct Iq4Nl {
    static constexpr int kBlockBytes = 18;
    static constexpr int kWinPerBlock = 1;
    static constexpr int kTabWords = 4;
    static constexpr int kHostTabWords = kTabWords;
    static constexpr bool kHasAdd = false;
    static constexpr bool kHasMin = false;
    struct Raw {
        unsigned q[4];
        float d;
    };
    static __device__ __forceinline__ Raw load(const unsigned char *blk, int) {
        Raw r;
        r.d = rd_half(blk);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            r.q[i] = rd32(blk + 2 + 4 * i);
        }
        return r;
    }
    static __device__ __forceinline__ void decode(const Raw &r, const uint32_t *tab, Win &o) {
        decode_nibbles(r.q, tab, o);
        o.s0 = r.d;
        o.s1 = r.d;
    }
};

/* IQ4_XS: 136-byte super-block of eight windows: f16 d, u16 scales_h, scales_l[4], 128 code
 * bytes. Window ib has the 6-bit scale (scales_l nibble ib | 2 bits of scales_h) - 32. */
struct Iq4Xs {
    static constexpr int kBlockBytes = 136;
    static constexpr int kWinPerBlock = 8;
    static constexpr int kTabWords = 4;
    static constexpr int kHostTabWords = kTabWords;
    static constexpr bool kHasAdd = false;
    static constexpr bool kHasMin = false;
    struct Raw {
        unsigned q[4];
        unsigned scale;
        float d;
    };
    static __device__ __forceinline__ Raw load(const unsigned char *blk, int wi) {
        Raw r;
        r.d = rd_half(blk);
        const unsigned lo = (blk[4 + (wi >> 1)] >> (4 * (wi & 1))) & 0x0Fu;
        const unsigned hi = (rd16(blk + 2) >> (2 * wi)) & 3u;
        r.scale = lo | (hi << 4);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            r.q[i] = *reinterpret_cast<const unsigned *>(blk + 8 + 16 * wi + 4 * i);
        }
        return r;
    }
    static __device__ __forceinline__ void decode(const Raw &r, const uint32_t *tab, Win &o) {
        decode_nibbles(r.q, tab, o);
        const float dl = __fmul_rn(r.d, (float)((int)r.scale - 32));
        o.s0 = dl;
        o.s1 = dl;
    }
};

/* Q2_0: 18-byte block of 64 weights, f16 d then 64 two-bit codes (four per byte, low bits
 * first); the weight is (code - 1) * d. Window wi is the 8 code bytes at 2 + 8 * wi. The four
 * codes map to the bytes -1, 0, 1, 2 with one PRMT. */
struct Q2_0 {
    static constexpr int kBlockBytes = 18;
    static constexpr int kWinPerBlock = 2;
    static constexpr int kTabWords = 4;
    static constexpr int kHostTabWords = 0;
    static constexpr bool kHasAdd = false;
    static constexpr bool kHasMin = false;
    struct Raw {
        unsigned q[4];
        float d;
    };
    static __device__ __forceinline__ Raw load(const unsigned char *blk, int wi) {
        Raw r;
        r.d = rd_half(blk);
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const unsigned lo = rd16(blk + 2 + 8 * wi + 4 * i);
            const unsigned hi = rd16(blk + 4 + 8 * wi + 4 * i);
            r.q[i] = lo | (hi << 16);
        }
        r.q[2] = 0;
        r.q[3] = 0;
        return r;
    }
    static __device__ __forceinline__ void decode(const Raw &r, const uint32_t *, Win &o) {
        const unsigned codes = 0x020100FFu; /* bytes -1, 0, 1, 2 */
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const unsigned word = r.q[i];
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const unsigned v = (word >> (8 * b)) & 0xFFu;
                const unsigned sel = (v & 0x03u) | ((v & 0x0Cu) << 2) | ((v & 0x30u) << 4) |
                                     ((v & 0xC0u) << 6);
                o.w[4 * i + b] = (int)__byte_perm(codes, 0u, sel);
            }
        }
        o.s0 = r.d;
        o.s1 = r.d;
    }
};

/* IQ2_XXS: 66-byte super-block of eight windows: f16 d then, per window, four grid indices
 * (one byte each) and a word holding four 7-bit sign indices and a 4-bit scale in its top
 * nibble. Each grid entry is eight bytes. */
struct Iq2Xxs {
    static constexpr int kBlockBytes = 66;
    static constexpr int kWinPerBlock = 8;
    static constexpr int kTabWords = 512;
    static constexpr int kHostTabWords = kTabWords;
    static constexpr bool kHasAdd = false;
    static constexpr bool kHasMin = false;
    struct Raw {
        unsigned a0, a1;
        float d;
    };
    static __device__ __forceinline__ Raw load(const unsigned char *blk, int wi) {
        Raw r;
        r.d = rd_half(blk);
        r.a0 = rd32(blk + 2 + 8 * wi);
        r.a1 = rd32(blk + 6 + 8 * wi);
        return r;
    }
    static __device__ __forceinline__ void decode(const Raw &r, const uint32_t *tab, Win &o) {
        const uint2 *grid = reinterpret_cast<const uint2 *>(tab);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 g = grid[(r.a0 >> (8 * l)) & 0xFFu];
            const unsigned sg = sign_byte((r.a1 >> (7 * l)) & 127u);
            o.w[2 * l] = apply_signs(g.x, sg & 0xFu);
            o.w[2 * l + 1] = apply_signs(g.y, sg >> 4);
        }
        const float db = half_step(r.d, r.a1 >> 28, 0.25f);
        o.s0 = db;
        o.s1 = db;
    }
};

/* IQ2_XS: 74-byte super-block: f16 d, 32 u16 (9-bit grid index, 7-bit sign index), eight
 * scale bytes. Window ib reads four of the u16 and scale byte ib, whose low nibble scales
 * weights 0..15 and high nibble weights 16..31. */
struct Iq2Xs {
    static constexpr int kBlockBytes = 74;
    static constexpr int kWinPerBlock = 8;
    static constexpr int kTabWords = 1024;
    static constexpr int kHostTabWords = kTabWords;
    static constexpr bool kHasAdd = false;
    static constexpr bool kHasMin = false;
    struct Raw {
        unsigned v0, v1;
        unsigned scale;
        float d;
    };
    static __device__ __forceinline__ Raw load(const unsigned char *blk, int wi) {
        Raw r;
        r.d = rd_half(blk);
        r.v0 = rd32(blk + 2 + 8 * wi);
        r.v1 = rd32(blk + 6 + 8 * wi);
        r.scale = blk[66 + wi];
        return r;
    }
    static __device__ __forceinline__ void decode(const Raw &r, const uint32_t *tab, Win &o) {
        const uint2 *grid = reinterpret_cast<const uint2 *>(tab);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const unsigned v = ((l < 2 ? r.v0 : r.v1) >> (16 * (l & 1))) & 0xFFFFu;
            const uint2 g = grid[v & 511u];
            const unsigned sg = sign_byte(v >> 9);
            o.w[2 * l] = apply_signs(g.x, sg & 0xFu);
            o.w[2 * l + 1] = apply_signs(g.y, sg >> 4);
        }
        o.s0 = half_step(r.d, r.scale & 0x0Fu, 0.25f);
        o.s1 = half_step(r.d, r.scale >> 4, 0.25f);
    }
};

/* IQ2_S: 82-byte super-block: f16 d, 32 grid index bytes, 32 sign bytes, eight bytes of
 * grid index high bits (two bits per index), eight scale bytes (nibbles as in IQ2_XS). */
struct Iq2S {
    static constexpr int kBlockBytes = 82;
    static constexpr int kWinPerBlock = 8;
    static constexpr int kTabWords = 2048;
    static constexpr int kHostTabWords = kTabWords;
    static constexpr bool kHasAdd = false;
    static constexpr bool kHasMin = false;
    struct Raw {
        unsigned qs, signs;
        unsigned qh, scale;
        float d;
    };
    static __device__ __forceinline__ Raw load(const unsigned char *blk, int wi) {
        Raw r;
        r.d = rd_half(blk);
        r.qs = rd32(blk + 2 + 4 * wi);
        r.signs = rd32(blk + 34 + 4 * wi);
        r.qh = blk[66 + wi];
        r.scale = blk[74 + wi];
        return r;
    }
    static __device__ __forceinline__ void decode(const Raw &r, const uint32_t *tab, Win &o) {
        const uint2 *grid = reinterpret_cast<const uint2 *>(tab);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const unsigned idx = ((r.qs >> (8 * l)) & 0xFFu) | ((r.qh << (8 - 2 * l)) & 0x300u);
            const uint2 g = grid[idx];
            const unsigned sg = (r.signs >> (8 * l)) & 0xFFu;
            o.w[2 * l] = apply_signs(g.x, sg & 0xFu);
            o.w[2 * l + 1] = apply_signs(g.y, sg >> 4);
        }
        o.s0 = half_step(r.d, r.scale & 0x0Fu, 0.25f);
        o.s1 = half_step(r.d, r.scale >> 4, 0.25f);
    }
};

/* IQ3_XXS: 98-byte super-block: f16 d, 64 grid index bytes, then per window a word with four
 * 7-bit sign indices and a 4-bit scale. Each grid entry is four bytes; a window reads eight
 * index bytes. */
struct Iq3Xxs {
    static constexpr int kBlockBytes = 98;
    static constexpr int kWinPerBlock = 8;
    static constexpr int kTabWords = 256;
    static constexpr int kHostTabWords = kTabWords;
    static constexpr bool kHasAdd = false;
    static constexpr bool kHasMin = false;
    struct Raw {
        unsigned q0, q1;
        unsigned a;
        float d;
    };
    static __device__ __forceinline__ Raw load(const unsigned char *blk, int wi) {
        Raw r;
        r.d = rd_half(blk);
        r.q0 = rd32(blk + 2 + 8 * wi);
        r.q1 = rd32(blk + 6 + 8 * wi);
        r.a = rd32(blk + 66 + 4 * wi);
        return r;
    }
    static __device__ __forceinline__ void decode(const Raw &r, const uint32_t *tab, Win &o) {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const unsigned idx = (l < 2 ? r.q0 : r.q1) >> (16 * (l & 1));
            const unsigned sg = sign_byte((r.a >> (7 * l)) & 127u);
            o.w[2 * l] = apply_signs(tab[idx & 0xFFu], sg & 0xFu);
            o.w[2 * l + 1] = apply_signs(tab[(idx >> 8) & 0xFFu], sg >> 4);
        }
        const float db = half_step(r.d, r.a >> 28, 0.5f);
        o.s0 = db;
        o.s1 = db;
    }
};

/* IQ3_S: 110-byte super-block: f16 d, 64 grid index bytes, eight bytes of index bit 8 (one
 * byte per window), 32 sign bytes, four bytes of scale nibbles (window ib: nibble ib & 1 of
 * byte ib >> 1; scale 1 + 2 * nibble). */
struct Iq3S {
    static constexpr int kBlockBytes = 110;
    static constexpr int kWinPerBlock = 8;
    static constexpr int kTabWords = 512;
    static constexpr int kHostTabWords = kTabWords;
    static constexpr bool kHasAdd = false;
    static constexpr bool kHasMin = false;
    struct Raw {
        unsigned q0, q1;
        unsigned qh, signs;
        unsigned scale;
        float d;
    };
    static __device__ __forceinline__ Raw load(const unsigned char *blk, int wi) {
        Raw r;
        r.d = rd_half(blk);
        r.q0 = rd32(blk + 2 + 8 * wi);
        r.q1 = rd32(blk + 6 + 8 * wi);
        r.qh = blk[66 + wi];
        r.signs = rd32(blk + 74 + 4 * wi);
        r.scale = (blk[106 + (wi >> 1)] >> (4 * (wi & 1))) & 0x0Fu;
        return r;
    }
    static __device__ __forceinline__ void decode(const Raw &r, const uint32_t *tab, Win &o) {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const unsigned idx = (l < 2 ? r.q0 : r.q1) >> (16 * (l & 1));
            const unsigned i1 = (idx & 0xFFu) | ((r.qh << (8 - 2 * l)) & 256u);
            const unsigned i2 = ((idx >> 8) & 0xFFu) | ((r.qh << (7 - 2 * l)) & 256u);
            const unsigned sg = (r.signs >> (8 * l)) & 0xFFu;
            o.w[2 * l] = apply_signs(tab[i1], sg & 0xFu);
            o.w[2 * l + 1] = apply_signs(tab[i2], sg >> 4);
        }
        const float db = __fmul_rn(r.d, (float)(1 + 2 * (int)r.scale));
        o.s0 = db;
        o.s1 = db;
    }
};

/* Q2_K: 84-byte super-block: 16 scale bytes (low nibble scales a group of 16, high nibble is
 * its minimum), 64 code bytes of 2-bit codes, f16 d at +80, f16 dmin at +82. Window wi
 * (0..7) is value range [32 wi, 32 wi + 32): its codes are bit pair 2 * (wi % 4) of the 32
 * bytes qs[32 * (wi / 4) ..), and its two groups use scale bytes 2 wi and 2 wi + 1. The
 * weight is d * (scale & 15) * code - dmin * (scale >> 4). */
struct Q2K {
    static constexpr int kBlockBytes = 84;
    static constexpr int kWinPerBlock = 8;
    static constexpr int kTabWords = 4;
    static constexpr int kHostTabWords = 0;
    static constexpr bool kHasAdd = false;
    static constexpr bool kHasMin = true;
    struct Raw {
        unsigned q[8];
        unsigned sc0, sc1;
        unsigned shift;
        float d, dmin;
    };
    static __device__ __forceinline__ Raw load(const unsigned char *blk, int wi) {
        Raw r;
        r.d = rd_half(blk + 80);
        r.dmin = rd_half(blk + 82);
        r.sc0 = blk[2 * wi];
        r.sc1 = blk[2 * wi + 1];
        r.shift = 2 * (wi & 3);
        const unsigned char *qs = blk + 16 + 32 * (wi >> 2);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            r.q[i] = *reinterpret_cast<const unsigned *>(qs + 4 * i);
        }
        return r;
    }
    static __device__ __forceinline__ void decode(const Raw &r, const uint32_t *, Win &o) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            o.w[i] = (int)((r.q[i] >> r.shift) & 0x03030303u);
        }
        o.s0 = __fmul_rn(r.d, (float)(r.sc0 & 0x0Fu));
        o.s1 = __fmul_rn(r.d, (float)(r.sc1 & 0x0Fu));
        o.m0 = __fmul_rn(r.dmin, (float)(r.sc0 >> 4));
        o.m1 = __fmul_rn(r.dmin, (float)(r.sc1 >> 4));
    }
};

/* The IQ1 shift of the grid values: weight = scale * (grid + delta), delta = +-0.125. */
#define GGUF_IQ1_DELTA 0.125f

/* IQ1_S: 50-byte super-block: f16 d, 32 grid index bytes, eight u16 qh words. Window ib
 * reads four index bytes and qh word ib: bits 3l..3l+2 are the high index bits of group l,
 * bits 12..14 the scale (2 * s + 1) and bit 15 the sign of the shift. Each grid entry is
 * eight signed bytes in {-1, 0, 1}, which are exactly the int8 weights. */
struct Iq1S {
    static constexpr int kBlockBytes = 50;
    static constexpr int kWinPerBlock = 8;
    static constexpr int kTabWords = 4096;
    static constexpr int kHostTabWords = kTabWords;
    static constexpr bool kHasAdd = true;
    static constexpr bool kHasMin = false;
    struct Raw {
        unsigned qs, qh;
        float d;
    };
    static __device__ __forceinline__ Raw load(const unsigned char *blk, int wi) {
        Raw r;
        r.d = rd_half(blk);
        r.qs = rd32(blk + 2 + 4 * wi);
        r.qh = rd16(blk + 34 + 2 * wi);
        return r;
    }
    static __device__ __forceinline__ void decode(const Raw &r, const uint32_t *tab, Win &o) {
        const uint2 *grid = reinterpret_cast<const uint2 *>(tab);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 g = grid[((r.qs >> (8 * l)) & 0xFFu) | (((r.qh >> (3 * l)) & 7u) << 8)];
            o.w[2 * l] = (int)g.x;
            o.w[2 * l + 1] = (int)g.y;
        }
        const float dl = __fmul_rn(r.d, (float)(2 * (int)((r.qh >> 12) & 7u) + 1));
        const float delta = (r.qh & 0x8000u) ? -GGUF_IQ1_DELTA : GGUF_IQ1_DELTA;
        o.s0 = dl;
        o.s1 = dl;
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            o.a[g] = delta;
        }
    }
};

/* IQ1_M: 56-byte super-block: 32 grid index bytes, 16 qh bytes (two per window: the low three
 * bits of each nibble are the high index bits of one group, bit 3 the sign of its shift), four
 * u16 scale words at +48. The f16 d has no field: the top nibbles of the four words, in order,
 * are its bits 0..3, 4..7, 8..11, 12..15. Word ib / 2 carries the 3-bit scales (2 * s + 1) of
 * window ib at shifts 6 * (ib % 2) (groups 0 and 1) and 6 * (ib % 2) + 3 (groups 2 and 3). */
struct Iq1M {
    static constexpr int kBlockBytes = 56;
    static constexpr int kWinPerBlock = 8;
    static constexpr int kTabWords = 4096;
    static constexpr int kHostTabWords = kTabWords;
    static constexpr bool kHasAdd = true;
    static constexpr bool kHasMin = false;
    struct Raw {
        unsigned qs, qh;
        unsigned word, dbits, shift;
    };
    static __device__ __forceinline__ Raw load(const unsigned char *blk, int wi) {
        Raw r;
        r.qs = rd32(blk + 4 * wi);
        r.qh = rd16(blk + 32 + 2 * wi);
        const unsigned w0 = rd16(blk + 48), w1 = rd16(blk + 50), w2 = rd16(blk + 52),
                       w3 = rd16(blk + 54);
        r.dbits = (w0 >> 12) | ((w1 >> 8) & 0x00F0u) | ((w2 >> 4) & 0x0F00u) | (w3 & 0xF000u);
        r.word = wi < 4 ? (wi < 2 ? w0 : w1) : (wi < 6 ? w2 : w3);
        r.shift = 6 * (wi & 1);
        return r;
    }
    static __device__ __forceinline__ void decode(const Raw &r, const uint32_t *tab, Win &o) {
        const uint2 *grid = reinterpret_cast<const uint2 *>(tab);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const unsigned nib = (r.qh >> (4 * (l & 1) + 8 * (l >> 1))) & 0x0Fu;
            const uint2 g = grid[((r.qs >> (8 * l)) & 0xFFu) | ((nib & 7u) << 8)];
            o.w[2 * l] = (int)g.x;
            o.w[2 * l + 1] = (int)g.y;
            o.a[l] = (nib & 8u) ? -GGUF_IQ1_DELTA : GGUF_IQ1_DELTA;
        }
        const float d = __half2float(__ushort_as_half((unsigned short)r.dbits));
        o.s0 = __fmul_rn(d, (float)(2 * (int)((r.word >> r.shift) & 7u) + 1));
        o.s1 = __fmul_rn(d, (float)(2 * (int)((r.word >> (r.shift + 3)) & 7u) + 1));
    }
};


/* Byte j of the result is bit j of the low nibble of nib (0 or 1). */
__device__ __forceinline__ unsigned bits_to_bytes(unsigned nib) {
    return (nib * 0x00204081u) & 0x01010101u;
}

/* Loads the four 32-bit words at p (2-byte aligned). */
__device__ __forceinline__ void load_words(const unsigned char *p, unsigned *q, int n) {
#pragma unroll
    for (int i = 0; i < n; ++i) {
        q[i] = rd32(p + 4 * i);
    }
}

/* Q4_0: 18-byte block, f16 d then 16 code bytes; weight j (0..15) is the low nibble of byte j
 * minus 8, weight 16 + j the high nibble. */
struct Q4_0 {
    static constexpr int kBlockBytes = 18;
    static constexpr int kWinPerBlock = 1;
    static constexpr int kTabWords = 4;
    static constexpr int kHostTabWords = 0;
    static constexpr bool kHasAdd = false;
    static constexpr bool kHasMin = false;
    struct Raw {
        unsigned q[4];
        float d;
    };
    static __device__ __forceinline__ Raw load(const unsigned char *blk, int) {
        Raw r;
        r.d = rd_half(blk);
        load_words(blk + 2, r.q, 4);
        return r;
    }
    static __device__ __forceinline__ void decode(const Raw &r, const uint32_t *, Win &o) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            o.w[i] = (int)__vsub4(r.q[i] & 0x0F0F0F0Fu, 0x08080808u);
            o.w[4 + i] = (int)__vsub4((r.q[i] >> 4) & 0x0F0F0F0Fu, 0x08080808u);
        }
        o.s0 = r.d;
        o.s1 = r.d;
    }
};

/* Q5_0: 22-byte block, f16 d, the 32 fifth bits qh, then 16 code bytes laid out like Q4_0;
 * the weight is (nibble | fifth bit << 4) - 16. */
struct Q5_0 {
    static constexpr int kBlockBytes = 22;
    static constexpr int kWinPerBlock = 1;
    static constexpr int kTabWords = 4;
    static constexpr int kHostTabWords = 0;
    static constexpr bool kHasAdd = false;
    static constexpr bool kHasMin = false;
    struct Raw {
        unsigned q[4];
        unsigned qh;
        float d;
    };
    static __device__ __forceinline__ Raw load(const unsigned char *blk, int) {
        Raw r;
        r.d = rd_half(blk);
        r.qh = rd32(blk + 2);
        load_words(blk + 6, r.q, 4);
        return r;
    }
    static __device__ __forceinline__ void decode(const Raw &r, const uint32_t *, Win &o) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const unsigned low = (r.q[i] & 0x0F0F0F0Fu) | (bits_to_bytes((r.qh >> (4 * i)) & 0xFu) << 4);
            const unsigned high = ((r.q[i] >> 4) & 0x0F0F0F0Fu) |
                                  (bits_to_bytes((r.qh >> (16 + 4 * i)) & 0xFu) << 4);
            o.w[i] = (int)__vsub4(low, 0x10101010u);
            o.w[4 + i] = (int)__vsub4(high, 0x10101010u);
        }
        o.s0 = r.d;
        o.s1 = r.d;
    }
};

/* Q8_0: 34-byte block, f16 d then 32 int8 weights. */
struct Q8_0 {
    static constexpr int kBlockBytes = 34;
    static constexpr int kWinPerBlock = 1;
    static constexpr int kTabWords = 4;
    static constexpr int kHostTabWords = 0;
    static constexpr bool kHasAdd = false;
    static constexpr bool kHasMin = false;
    struct Raw {
        unsigned q[8];
        float d;
    };
    static __device__ __forceinline__ Raw load(const unsigned char *blk, int) {
        Raw r;
        r.d = rd_half(blk);
        load_words(blk + 2, r.q, 8);
        return r;
    }
    static __device__ __forceinline__ void decode(const Raw &r, const uint32_t *, Win &o) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            o.w[i] = (int)r.q[i];
        }
        o.s0 = r.d;
        o.s1 = r.d;
    }
};

/* Q4_K: 144-byte super-block: f16 d, f16 dmin, 12 packed scale bytes, 128 code bytes. Window g
 * (0..7) reads the 32 bytes at 32 * (g / 2) and takes their low nibbles for even g, high for odd;
 * the weight is d * sc * q - dmin * mn (the product rounded before the subtraction). */
struct Q4K {
    static constexpr int kBlockBytes = 144;
    static constexpr int kWinPerBlock = 8;
    static constexpr int kTabWords = 4;
    static constexpr int kHostTabWords = 0;
    static constexpr bool kHasAdd = false;
    static constexpr bool kHasMin = true;
    struct Raw {
        unsigned q[8];
        unsigned shift;
        float s, m;
    };
    static __device__ __forceinline__ Raw load(const unsigned char *blk, int g) {
        Raw r;
        const float d = rd_half(blk);
        const float dmin = rd_half(blk + 2);
        unsigned sc, mn;
        scale_min_k4(blk + 4, g, sc, mn);
        r.s = __fmul_rn(d, (float)sc);
        r.m = __fmul_rn(dmin, (float)mn);
        r.shift = 4 * (g & 1);
        load_words(blk + 16 + 32 * (g >> 1), r.q, 8);
        return r;
    }
    static __device__ __forceinline__ void decode(const Raw &r, const uint32_t *, Win &o) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            o.w[i] = (int)((r.q[i] >> r.shift) & 0x0F0F0F0Fu);
        }
        o.s0 = r.s;
        o.s1 = r.s;
        o.m0 = r.m;
        o.m1 = r.m;
    }
};

/* Q5_K: 176-byte super-block: Q4_K's header, then 32 high-bit bytes, then 128 code bytes. The
 * fifth bit of weight l of window g is bit g of high-bit byte l. */
struct Q5K {
    static constexpr int kBlockBytes = 176;
    static constexpr int kWinPerBlock = 8;
    static constexpr int kTabWords = 4;
    static constexpr int kHostTabWords = 0;
    static constexpr bool kHasAdd = false;
    static constexpr bool kHasMin = true;
    struct Raw {
        unsigned q[8];
        unsigned h[8];
        unsigned shift;
        unsigned g;
        float s, m;
    };
    static __device__ __forceinline__ Raw load(const unsigned char *blk, int g) {
        Raw r;
        const float d = rd_half(blk);
        const float dmin = rd_half(blk + 2);
        unsigned sc, mn;
        scale_min_k4(blk + 4, g, sc, mn);
        r.s = __fmul_rn(d, (float)sc);
        r.m = __fmul_rn(dmin, (float)mn);
        r.shift = 4 * (g & 1);
        r.g = (unsigned)g;
        load_words(blk + 16, r.h, 8);
        load_words(blk + 48 + 32 * (g >> 1), r.q, 8);
        return r;
    }
    static __device__ __forceinline__ void decode(const Raw &r, const uint32_t *, Win &o) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            o.w[i] = (int)(((r.q[i] >> r.shift) & 0x0F0F0F0Fu) | (((r.h[i] >> r.g) & 0x01010101u) << 4));
        }
        o.s0 = r.s;
        o.s1 = r.s;
        o.m0 = r.m;
        o.m1 = r.m;
    }
};

/* Q6_K: 210-byte super-block: 128 low-nibble bytes, 64 two-bit bytes, 16 int8 group scales, f16 d
 * last. Window wi (half = wi / 4, quadrant qd = wi % 4) reads the 32 low bytes at 64 * half +
 * 32 * (qd % 2) (the low nibble for qd < 2, the high for qd >= 2) and the 32 high bytes at
 * 128 + 32 * half (bit pair 2 * qd); the weight is (q - 32) * d * scale[8 * half + 2 * qd + g]. */
struct Q6K {
    static constexpr int kBlockBytes = 210;
    static constexpr int kWinPerBlock = 8;
    static constexpr int kTabWords = 4;
    static constexpr int kHostTabWords = 0;
    static constexpr bool kHasAdd = false;
    static constexpr bool kHasMin = false;
    struct Raw {
        unsigned q[8];
        unsigned h[8];
        unsigned low_shift;
        unsigned high_shift;
        float s0, s1;
    };
    static __device__ __forceinline__ Raw load(const unsigned char *blk, int wi) {
        Raw r;
        const int half = wi >> 2, qd = wi & 3;
        const float d = rd_half(blk + 208);
        r.s0 = __fmul_rn(d, (float)(int)(signed char)blk[192 + 8 * half + 2 * qd]);
        r.s1 = __fmul_rn(d, (float)(int)(signed char)blk[192 + 8 * half + 2 * qd + 1]);
        r.low_shift = 4 * (qd >> 1);
        r.high_shift = 2 * qd;
        load_words(blk + 64 * half + 32 * (qd & 1), r.q, 8);
        load_words(blk + 128 + 32 * half, r.h, 8);
        return r;
    }
    static __device__ __forceinline__ void decode(const Raw &r, const uint32_t *, Win &o) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const unsigned v = ((r.q[i] >> r.low_shift) & 0x0F0F0F0Fu) |
                               (((r.h[i] >> r.high_shift) & 0x03030303u) << 4);
            o.w[i] = (int)__vsub4(v, 0x20202020u);
        }
        o.s0 = r.s0;
        o.s1 = r.s1;
    }
};

/* Q3_K: 110-byte super-block: 32 high-bit bytes, 64 two-bit bytes, 12 packed 6-bit group scales,
 * f16 d last. Window wi (half = wi / 4, j = wi % 4) reads the 32 bytes at 32 + 32 * half (bit
 * pair 2 * j) and bit wi of the 32 high-bit bytes: a set bit adds 4, and the weight is
 * (code + 4 * bit - 4) * (d * (scale - 32)) with the two groups 2 * wi and 2 * wi + 1. */
struct Q3K {
    static constexpr int kBlockBytes = 110;
    static constexpr int kWinPerBlock = 8;
    static constexpr int kTabWords = 4;
    static constexpr int kHostTabWords = 0;
    static constexpr bool kHasAdd = false;
    static constexpr bool kHasMin = false;
    struct Raw {
        unsigned q[8];
        unsigned h[8];
        unsigned shift;
        unsigned wi;
        float s0, s1;
    };
    static __device__ __forceinline__ int group_scale(const unsigned char *ks, int s) {
        const int row = s >> 2, col = s & 3;
        const unsigned nib = (ks[(col + 4 * row) & 7] >> (4 * (row >> 1))) & 0x0Fu;
        const unsigned h2 = (ks[8 + col] >> (2 * row)) & 3u;
        return (int)(nib | (h2 << 4)) - 32;
    }
    static __device__ __forceinline__ Raw load(const unsigned char *blk, int wi) {
        Raw r;
        const float d = rd_half(blk + 108);
        r.s0 = __fmul_rn(d, (float)group_scale(blk + 96, 2 * wi));
        r.s1 = __fmul_rn(d, (float)group_scale(blk + 96, 2 * wi + 1));
        r.shift = 2 * (wi & 3);
        r.wi = (unsigned)wi;
        load_words(blk, r.h, 8);
        load_words(blk + 32 + 32 * (wi >> 2), r.q, 8);
        return r;
    }
    static __device__ __forceinline__ void decode(const Raw &r, const uint32_t *, Win &o) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const unsigned v = ((r.q[i] >> r.shift) & 0x03030303u) |
                               (((r.h[i] >> r.wi) & 0x01010101u) << 2);
            o.w[i] = (int)__vsub4(v, 0x04040404u);
        }
        o.s0 = r.s0;
        o.s1 = r.s1;
    }
};

}  // namespace gguf

#endif /* INFERENCE_CUDA_RESIDENT_GGUF_DECODE_CUH */
