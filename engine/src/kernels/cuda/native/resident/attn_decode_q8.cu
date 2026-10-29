/*
 * The KV8 split-KV decode attention kernel of the resident decode step: the same split-KV
 * decode attention as attn_decode.cu, reading the int8 codes and the fp16 scales of the frozen
 * KV8 planes that kv8_append writes instead of an fp16 cache.
 *
 * The split-KV scheme with a fixed-order combine follows
 * this repository's own attn_decode.cu (the technique is public, Flash-Decoding, Dao et
 * al. 2023); the packed int8 layout with one fp16 scale per 32 values read in contiguous
 * 64-cell tiles and dequantized in registers is the design idea of Strata's laguna_kv8 flash
 * decode. The code is original CUDA C
 * for sm_86 (sm_72 compiles).
 *
 * What it computes, per token t of m and query head h (kv head h / G, G = 8):
 *   out[t][h][d] = sum_j softmax_j(scale * q[t][h] . K[j][kv]) * V[j][kv][d] * sigmoid(gate)
 * where K and V are the dequantized values code * f32(stored fp16 scale), read once per kv head
 * for all G query heads, dequantized in registers (no scratch beyond `part`). Accumulation is
 * f32, every reduction has a fixed order and there is no float atomic.
 *
 * Two launches on one stream (as attn_decode):
 *   attn_q8_part_k     grid (max_splits, n_kv_head, m), 256 threads. Tiles of 64 keys (one
 *                      contiguous 64-cell run of a linear cache); online softmax per warp-head;
 *                      one partial (max, sum, weighted V) per query head goes to `part`.
 *   attn_q8_combine_k  grid (n_head, m), head_dim threads. Merges the partials of the active
 *                      splits in index order, divides, applies the gate, writes out.
 * kv_len is read from the device when the kernels run, so one captured graph serves every
 * position. A kv_len outside [1, cells - (m - 1)] (a ring takes any positive value below 2^30)
 * writes NaN to out.
 */
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>
#include <cstdlib>
#ifdef INFERENCE_CUDA_ARMS_QUALIFICATION
#include <atomic>
#include <cstdio>
#endif

#include "abi.h"
#include "device.h"
#include "resident/attn_decode_q8.h"

namespace {

constexpr int kGroup = 8;   // query heads per kv head; the lane layout of reduce8 assumes 8
constexpr int kQuant = 32;  // values per fp16 scale: the frozen group of the KV8 format
constexpr int kTile = 64;   // keys per tile: the contiguous 64-cell run the kernel reads
constexpr int kMaxM = 16;
constexpr int kMaxSplits = 128;  // one thread per split in the combine statistics step

struct Args {
    const float *q;          // [m][n_head][D] f32
    const float *proj;       // NULL (no gate) or [m][ld] f32; head h's gate is at h * 2 * D + D
    const signed char *k_codes;     // [n_kv_head][cells][D] int8
    const unsigned short *k_scales; // [n_kv_head][cells][D / 32] fp16
    const signed char *v_codes;
    const unsigned short *v_scales;
    const int *kv_len;       // device word: keys visible to token 0; token t sees kv_len + t
    float *part;
    float *out;              // [m][n_head * D] f32
    int ld, m, n_head, n_kv_head, cells, max_splits, min_keys;
    int window, ring;        // 0: all positions / linear cells; else last `window` / cell p % ring
    float scale;
    const float *sinks = nullptr;  // private launch args only; the external ABI is unchanged
};

// Key range plan from the number of keys alone, as attn_decode.cu: n_split splits of `chunk`
// keys (a multiple of the tile) cover n_keys; split s owns [s * chunk, min(n_keys, (s+1)*chunk)).
// The qualification FFI keeps a separate HOST mirror of this arithmetic; the pin
// test pins both to exact values so the two copies cannot drift.
__device__ inline void split_plan(int n_keys, int max_splits, int min_keys, int &chunk,
                                  int &n_active) {
    const int want = (n_keys + min_keys - 1) / min_keys;
    const int n_split = min(max(want, 1), max_splits);
    chunk = (((n_keys + n_split - 1) / n_split) + kTile - 1) / kTile * kTile;
    n_active = (n_keys + chunk - 1) / chunk;
}

// kv_len must leave room for the m tokens in a linear cache; a ring takes any positive length.
__device__ inline bool kv_ok(const Args &a, int kv_len) {
    return kv_len >= 1 && (a.ring > 0 ? kv_len < (1 << 30) : kv_len <= a.cells - (a.m - 1));
}

// Visible keys of token t: the last min(n_keys, window) positions. first = position of key 0.
__device__ inline void visible(const Args &a, int n_keys, int &first, int &count) {
    count = a.window > 0 ? min(n_keys, a.window) : n_keys;
    first = n_keys - count;
}

// Cache cell of the i-th visible key.
__device__ inline int cache_cell(const Args &a, int first, int i) {
    return a.ring > 0 ? (first + i) % a.ring : first + i;
}

__device__ inline float warp_max_a(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ inline float warp_sum_a(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ inline float h2f(unsigned short bits) { return __half2float(__ushort_as_half(bits)); }

// Reduces eight per-lane partial sums (one per query head) over the 32 lanes with 9 shuffles,
// as attn_decode.cu: lane l with (l & 3) == 0 holds the full sum of head
// ((l >> 4) & 1) * 4 + ((l >> 3) & 1) * 2 + ((l >> 2) & 1); the order is fixed.
__device__ inline float reduce8(const float (&a)[8], int lane, int &head) {
    const unsigned full = 0xffffffffu;
    const bool h16 = lane & 16, h8 = lane & 8, h4 = lane & 4;
    float b[4], c[2];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float keep = h16 ? a[i + 4] : a[i], give = h16 ? a[i] : a[i + 4];
        b[i] = keep + __shfl_xor_sync(full, give, 16);
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const float keep = h8 ? b[i + 2] : b[i], give = h8 ? b[i] : b[i + 2];
        c[i] = keep + __shfl_xor_sync(full, give, 8);
    }
    const float keep = h4 ? c[1] : c[0], give = h4 ? c[0] : c[1];
    float d = keep + __shfl_xor_sync(full, give, 4);
    d += __shfl_xor_sync(full, d, 2);
    d += __shfl_xor_sync(full, d, 1);
    head = (h16 ? 4 : 0) + (h8 ? 2 : 0) + (h4 ? 1 : 0);
    return d;
}

template <int D>
__global__ void __launch_bounds__(256) attn_q8_part_k(Args a) {
    constexpr int G = kGroup, NCH = D / 128;  // chunks of 4 dims per lane: 2 (D 256) or 1 (D 128)
    static_assert(D == 128 || D == 256, "head_dim 128 or 256");
    __shared__ float s_sc[G][kTile + 1];      // scores of the tile, head-major, padded
    __shared__ __align__(16) float s_p[kTile][G];  // probabilities, key-major (float4 reads)
    __shared__ float s_corr[G];
    const int s = blockIdx.x, kvh = blockIdx.y, t = blockIdx.z;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int kv_len = *a.kv_len;
    if (!kv_ok(a, kv_len)) {
        return;
    }
    const int n_keys = kv_len + t;
    int first, count;
    visible(a, n_keys, first, count);
    int chunk, n_active;
    split_plan(count, a.max_splits, a.min_keys, chunk, n_active);
    if (s >= n_active) {
        return;
    }
    const int k_begin = s * chunk, k_end = min(count, k_begin + chunk);
    const int groups = D / kQuant;
    const size_t code_stride = (size_t)a.cells * D;    // cells of one kv head of codes
    const size_t scale_stride = (size_t)a.cells * groups;
    const bool own = tid < D;  // one dimension per thread; with D 128 only threads 0..127 own one

    // This lane's dims are 4 * lane .. +3, and for D 256 also 128 + 4 * lane .. +3; the group of
    // chunk c is 4c + lane / 8. q is pre-scaled once.
    float4 qv[G][2];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        const float *qh = a.q + ((size_t)t * a.n_head + (size_t)kvh * G + g) * D;
#pragma unroll
        for (int c = 0; c < NCH; ++c) {
            qv[g][c] = *reinterpret_cast<const float4 *>(qh + 128 * c + 4 * lane);
            qv[g][c].x *= a.scale; qv[g][c].y *= a.scale; qv[g][c].z *= a.scale; qv[g][c].w *= a.scale;
        }
    }
    float acc[G];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        acc[g] = 0.f;
    }
    float m_run = -INFINITY, l_run = 0.f;  // head `warp`'s running max and sum, same in every lane

    for (int base = k_begin; base < k_end; base += kTile) {
        const int nvalid = min(kTile, k_end - base);
        // V of this tile, one dimension per thread, codes and scale bits in registers, issued
        // first so its latency overlaps the scores; the threads that own no dimension take part
        // in score/softmax work and all barriers.
        signed char vc[kTile];
        unsigned short vs[kTile];
        if (own) {
            const signed char *vcol = a.v_codes + (size_t)kvh * code_stride + tid;
            const unsigned short *vrow =
                a.v_scales + (size_t)kvh * scale_stride + (tid / kQuant);
#pragma unroll 8
            for (int j = 0; j < kTile; ++j) {
                vc[j] = j < nvalid ? vcol[(size_t)cache_cell(a, first, base + j) * D]
                                   : (signed char)0;
                vs[j] = j < nvalid ? vrow[(size_t)cache_cell(a, first, base + j) * groups]
                                   : (unsigned short)0;
            }
        }
        // Phase 1: scores. Warp w scores keys w, w + 8, ... of the tile; each lane dequantizes
        // its 4 dims of each key in registers against the key's stored fp16 scale.
#pragma unroll
        for (int i = 0; i < kTile / 8; ++i) {
            const int j = warp + 8 * i;
            if (j < nvalid) {
                const signed char *kc =
                    a.k_codes + (size_t)kvh * code_stride +
                    (size_t)cache_cell(a, first, base + j) * D;
                const unsigned short *ks = a.k_scales + (size_t)kvh * scale_stride +
                                           (size_t)cache_cell(a, first, base + j) * groups;
                float kf[8];
#pragma unroll
                for (int c = 0; c < NCH; ++c) {
                    const float sv = h2f(ks[4 * c + (lane >> 3)]);
                    const int word = *reinterpret_cast<const int *>(kc + 128 * c + 4 * lane);
                    kf[4 * c] = sv * (float)(signed char)(word & 0xff);
                    kf[4 * c + 1] = sv * (float)(signed char)((word >> 8) & 0xff);
                    kf[4 * c + 2] = sv * (float)(signed char)((word >> 16) & 0xff);
                    kf[4 * c + 3] = sv * (float)(signed char)((word >> 24) & 0xff);
                }
                float dots[8];
#pragma unroll
                for (int g = 0; g < G; ++g) {
                    float p = qv[g][0].x * kf[0];
                    p = fmaf(qv[g][0].y, kf[1], p); p = fmaf(qv[g][0].z, kf[2], p);
                    p = fmaf(qv[g][0].w, kf[3], p);
#pragma unroll
                    for (int c = 1; c < NCH; ++c) {
                        p = fmaf(qv[g][c].x, kf[4 * c], p); p = fmaf(qv[g][c].y, kf[4 * c + 1], p);
                        p = fmaf(qv[g][c].z, kf[4 * c + 2], p); p = fmaf(qv[g][c].w, kf[4 * c + 3], p);
                    }
                    dots[g] = p;
                }
                int head;
                const float sum = reduce8(dots, lane, head);
                if ((lane & 3) == 0) {
                    s_sc[head][j] = sum;
                }
            }
        }
        __syncthreads();
        // Phase 2: softmax statistics of head `warp` over the whole 64-key tile, in two 32-lane
        // passes with a fixed order: the exact max first, then the probabilities under it.
        {
            float m_new = m_run;
            for (int half = 0; half < kTile / 32; ++half) {
                const int j = half * 32 + lane;
                const float sc = j < nvalid ? s_sc[warp][j] : -INFINITY;
                m_new = fmaxf(m_new, warp_max_a(sc));
            }
            const float corr = expf(m_run - m_new);
            float psum = 0.f;
            for (int half = 0; half < kTile / 32; ++half) {
                const int j = half * 32 + lane;
                const float sc = j < nvalid ? s_sc[warp][j] : -INFINITY;
                const float p = j < nvalid ? expf(sc - m_new) : 0.f;
                psum += p;  // reduced below, in lane order within each half
                s_p[j][warp] = p;
            }
            l_run = l_run * corr + warp_sum_a(psum);
            m_run = m_new;
            if (lane == 0) {
                s_corr[warp] = corr;
            }
        }
        __syncthreads();
        // Phase 3: acc[g] (dimension tid of head g) += p[g][j] * V[j][tid], dequantized here.
#pragma unroll
        for (int g = 0; g < G; ++g) {
            acc[g] *= s_corr[g];
        }
        // D128 lanes128..255 never loaded V registers; all lanes still join every barrier.
        if (own) {
#pragma unroll 4
            for (int j = 0; j < kTile; ++j) {
                const float v = h2f(vs[j]) * (float)vc[j];
                const float4 pa = *reinterpret_cast<const float4 *>(&s_p[j][0]);
                const float4 pb = *reinterpret_cast<const float4 *>(&s_p[j][4]);
                acc[0] = fmaf(pa.x, v, acc[0]); acc[1] = fmaf(pa.y, v, acc[1]);
                acc[2] = fmaf(pa.z, v, acc[2]); acc[3] = fmaf(pa.w, v, acc[3]);
                acc[4] = fmaf(pb.x, v, acc[4]); acc[5] = fmaf(pb.y, v, acc[5]);
                acc[6] = fmaf(pb.z, v, acc[6]); acc[7] = fmaf(pb.w, v, acc[7]);
            }
        }
    }
    // Partial of (token t, kv head, split s): all stats first, then all accumulators, as attn_decode.
    const size_t cell = ((size_t)t * a.n_kv_head + kvh) * a.max_splits + s;
    const size_t n_cells = (size_t)a.m * a.n_kv_head * a.max_splits;
    float *ml = a.part + cell * G * 2;
    float *ac = a.part + n_cells * G * 2 + cell * G * D;
    if (lane == 0) {
        ml[warp * 2] = m_run;
        ml[warp * 2 + 1] = l_run;
    }
    if (own) {
#pragma unroll
        for (int g = 0; g < G; ++g) {
            ac[(size_t)g * D + tid] = acc[g];
        }
    }
}

#ifdef INFERENCE_CUDA_ARMS_QUALIFICATION
// Which kernel ACTUALLY executed on the device: the first thread of block (0,0,0)
// writes its identity here before any early return, and the fixture reads
// the word back after each leg. Unlike the host-side dispatch tag, this cannot pass
// if a launch statement resolves to a different kernel than intended.
__device__ int g_kernel_ran = 0;
#endif

// Separate variant preserves legacy register allocation and flag-off performance.
template <int D>
__global__ void __launch_bounds__(256, 2) attn_q8_register_v_part_k(Args a) {
    constexpr int G = kGroup, NCH = D / 128;  // chunks of 4 dims per lane: 2 (D 256) or 1 (D 128)
    static_assert(D == 256, "register V is qualified for head_dim 256");
    __shared__ float s_sc[G][kTile + 1];      // scores of the tile, head-major, padded
    __shared__ __align__(16) float s_p[kTile][G];  // probabilities, key-major (float4 reads)
    __shared__ float s_corr[G];
    const int s = blockIdx.x, kvh = blockIdx.y, t = blockIdx.z;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
#ifdef INFERENCE_CUDA_ARMS_QUALIFICATION
    if (s == 0 && kvh == 0 && t == 0 && tid == 0) { g_kernel_ran = 1; }
#endif
    const int kv_len = *a.kv_len;
    if (!kv_ok(a, kv_len)) {
        return;
    }
    const int n_keys = kv_len + t;
    int first, count;
    visible(a, n_keys, first, count);
    int chunk, n_active;
    split_plan(count, a.max_splits, a.min_keys, chunk, n_active);
    if (s >= n_active) {
        return;
    }
    const int k_begin = s * chunk, k_end = min(count, k_begin + chunk);
    const int groups = D / kQuant;
    const size_t code_stride = (size_t)a.cells * D;    // cells of one kv head of codes
    const size_t scale_stride = (size_t)a.cells * groups;
    const bool own = tid < D;  // one dimension per thread; with D 128 only threads 0..127 own one

    // This lane's dims are 4 * lane .. +3, and for D 256 also 128 + 4 * lane .. +3; the group of
    // chunk c is 4c + lane / 8. q is pre-scaled once.
    float4 qv[G][2];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        const float *qh = a.q + ((size_t)t * a.n_head + (size_t)kvh * G + g) * D;
#pragma unroll
        for (int c = 0; c < NCH; ++c) {
            qv[g][c] = *reinterpret_cast<const float4 *>(qh + 128 * c + 4 * lane);
            qv[g][c].x *= a.scale; qv[g][c].y *= a.scale; qv[g][c].z *= a.scale; qv[g][c].w *= a.scale;
        }
    }
    float acc[G];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        acc[g] = 0.f;
    }
    float m_run = -INFINITY, l_run = 0.f;  // head `warp`'s running max and sum, same in every lane

    for (int base = k_begin; base < k_end; base += kTile) {
        const int nvalid = min(kTile, k_end - base);
        // Phase 1: scores. Warp w scores keys w, w + 8, ... of the tile; each lane dequantizes
        // its 4 dims of each key in registers against the key's stored fp16 scale.
#pragma unroll
        for (int i = 0; i < kTile / 8; ++i) {
            const int j = warp + 8 * i;
            if (j < nvalid) {
                const signed char *kc =
                    a.k_codes + (size_t)kvh * code_stride +
                    (size_t)cache_cell(a, first, base + j) * D;
                const unsigned short *ks = a.k_scales + (size_t)kvh * scale_stride +
                                           (size_t)cache_cell(a, first, base + j) * groups;
                float kf[8];
#pragma unroll
                for (int c = 0; c < NCH; ++c) {
                    const float sv = h2f(ks[4 * c + (lane >> 3)]);
                    const int word = *reinterpret_cast<const int *>(kc + 128 * c + 4 * lane);
                    kf[4 * c] = sv * (float)(signed char)(word & 0xff);
                    kf[4 * c + 1] = sv * (float)(signed char)((word >> 8) & 0xff);
                    kf[4 * c + 2] = sv * (float)(signed char)((word >> 16) & 0xff);
                    kf[4 * c + 3] = sv * (float)(signed char)((word >> 24) & 0xff);
                }
                float dots[8];
#pragma unroll
                for (int g = 0; g < G; ++g) {
                    float p = qv[g][0].x * kf[0];
                    p = fmaf(qv[g][0].y, kf[1], p); p = fmaf(qv[g][0].z, kf[2], p);
                    p = fmaf(qv[g][0].w, kf[3], p);
#pragma unroll
                    for (int c = 1; c < NCH; ++c) {
                        p = fmaf(qv[g][c].x, kf[4 * c], p); p = fmaf(qv[g][c].y, kf[4 * c + 1], p);
                        p = fmaf(qv[g][c].z, kf[4 * c + 2], p); p = fmaf(qv[g][c].w, kf[4 * c + 3], p);
                    }
                    dots[g] = p;
                }
                int head;
                const float sum = reduce8(dots, lane, head);
                if ((lane & 3) == 0) {
                    s_sc[head][j] = sum;
                }
            }
        }
        __syncthreads();
        // Phase 2: softmax statistics of head `warp` over the whole 64-key tile, in two 32-lane
        // passes with a fixed order: the exact max first, then the probabilities under it.
        {
            float m_new = m_run;
            for (int half = 0; half < kTile / 32; ++half) {
                const int j = half * 32 + lane;
                const float sc = j < nvalid ? s_sc[warp][j] : -INFINITY;
                m_new = fmaxf(m_new, warp_max_a(sc));
            }
            const float corr = expf(m_run - m_new);
            float psum = 0.f;
            for (int half = 0; half < kTile / 32; ++half) {
                const int j = half * 32 + lane;
                const float sc = j < nvalid ? s_sc[warp][j] : -INFINITY;
                const float p = j < nvalid ? expf(sc - m_new) : 0.f;
                psum += p;  // reduced below, in lane order within each half
                s_p[j][warp] = p;
            }
            l_run = l_run * corr + warp_sum_a(psum);
            m_run = m_new;
            if (lane == 0) {
                s_corr[warp] = corr;
            }
        }
        __syncthreads();
        // Phase 3: acc[g] (dimension tid of head g) += p[g][j] * V[j][tid], dequantized here.
#pragma unroll
        for (int g = 0; g < G; ++g) {
            acc[g] *= s_corr[g];
        }
        const signed char *vcol = a.v_codes + (size_t)kvh * code_stride + tid;
        const unsigned short *vrow =
            a.v_scales + (size_t)kvh * scale_stride + (tid / kQuant);
        // Eight constant-index values fit registers without changing the ascending-key FMA chain.
#pragma unroll 1
        for (int first_j = 0; first_j < kTile; first_j += 8) {
            float values[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int j = first_j + i;
                values[i] = 0.f;
                if (own && j < nvalid) {
                    const int row = cache_cell(a, first, base + j);
                    values[i] = h2f(vrow[(size_t)row * groups]) *
                                (float)vcol[(size_t)row * D];
                }
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int j = first_j + i;
                const float v = values[i];
                const float4 pa = *reinterpret_cast<const float4 *>(&s_p[j][0]);
                const float4 pb = *reinterpret_cast<const float4 *>(&s_p[j][4]);
                acc[0] = fmaf(pa.x, v, acc[0]); acc[1] = fmaf(pa.y, v, acc[1]);
                acc[2] = fmaf(pa.z, v, acc[2]); acc[3] = fmaf(pa.w, v, acc[3]);
                acc[4] = fmaf(pb.x, v, acc[4]); acc[5] = fmaf(pb.y, v, acc[5]);
                acc[6] = fmaf(pb.z, v, acc[6]); acc[7] = fmaf(pb.w, v, acc[7]);
            }
        }
    }
    // Partial of (token t, kv head, split s): all stats first, then all accumulators, as attn_decode.
    const size_t cell = ((size_t)t * a.n_kv_head + kvh) * a.max_splits + s;
    const size_t n_cells = (size_t)a.m * a.n_kv_head * a.max_splits;
    float *ml = a.part + cell * G * 2;
    float *ac = a.part + n_cells * G * 2 + cell * G * D;
    if (lane == 0) {
        ml[warp * 2] = m_run;
        ml[warp * 2 + 1] = l_run;
    }
    if (own) {
#pragma unroll
        for (int g = 0; g < G; ++g) {
            ac[(size_t)g * D + tid] = acc[g];
        }
    }
}

// Prefetch variant: a SEPARATE copy of the register-V kernel that
// keeps the eight GQA heads' query chunks in shared memory (the 4160-byte tile buffers
// leave ample room) and holds key j+8's raw code words in registers before key j's dot
// chain, so the latency-bound serial scan (measured 127 regs, 2 blocks/SM) overlaps loads
// with compute. The multiply/fmaf order per head is unchanged, so it is bitwise-equal to
// the legacy kernel; INFERENCE_CUDA_KV8_ATTN_V_PREFETCH=1 (exactly "1", read once)
// selects it. The legacy kernel above is untouched.
template <int D>
__global__ void __launch_bounds__(256, 2) attn_q8_register_v_prefetch_k(Args a) {
    constexpr int G = kGroup, NCH = D / 128;  // chunks of 4 dims per lane: 2 (D 256) or 1 (D 128)
    static_assert(D == 256, "register V is qualified for head_dim 256");
    __shared__ float s_sc[G][kTile + 1];      // scores of the tile, head-major, padded
    __shared__ __align__(16) float s_p[kTile][G];  // probabilities, key-major (float4 reads)
    __shared__ float s_corr[G];
    const int s = blockIdx.x, kvh = blockIdx.y, t = blockIdx.z;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
#ifdef INFERENCE_CUDA_ARMS_QUALIFICATION
    if (s == 0 && kvh == 0 && t == 0 && tid == 0) { g_kernel_ran = 2; }
#endif
    const int kv_len = *a.kv_len;
    if (!kv_ok(a, kv_len)) {
        return;
    }
    const int n_keys = kv_len + t;
    int first, count;
    visible(a, n_keys, first, count);
    int chunk, n_active;
    split_plan(count, a.max_splits, a.min_keys, chunk, n_active);
    if (s >= n_active) {
        return;
    }
    const int k_begin = s * chunk, k_end = min(count, k_begin + chunk);
    const int groups = D / kQuant;
    const size_t code_stride = (size_t)a.cells * D;    // cells of one kv head of codes
    const size_t scale_stride = (size_t)a.cells * groups;
    const bool own = tid < D;  // one dimension per thread; with D 128 only threads 0..127 own one

    // This lane's dims are 4 * lane .. +3, and for D 256 also 128 + 4 * lane .. +3; the group
    // of chunk c is 4c + lane / 8. q is pre-scaled once. This variant stages every head's
    // chunk in shared memory (one float4 per lane per chunk, written once); the stored f32
    // bits are those the register path computes, so the dot chains stay bitwise-identical.
    __shared__ __align__(16) float s_q[G][NCH][32][4];

    // Warp w fills tile w (G == 8 tiles, 8 warps): distinct smem cells per warp, no
    // write/write overlap. The values written are identical to the pre-change all-warp
    // fill, so the syncthreads barrier plus the reads below see the same bits.
    for (int g = warp; g < G; g += 8) {
        const float *qh = a.q + ((size_t)t * a.n_head + (size_t)kvh * G + g) * D;
#pragma unroll
        for (int c = 0; c < NCH; ++c) {
            const float4 qq = *reinterpret_cast<const float4 *>(qh + 128 * c + 4 * lane);
            s_q[g][c][lane][0] = qq.x * a.scale;
            s_q[g][c][lane][1] = qq.y * a.scale;
            s_q[g][c][lane][2] = qq.z * a.scale;
            s_q[g][c][lane][3] = qq.w * a.scale;
        }
    }
    __syncthreads();
    float acc[G];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        acc[g] = 0.f;
    }
    float m_run = -INFINITY, l_run = 0.f;  // head `warp`'s running max and sum, same in every lane

    for (int base = k_begin; base < k_end; base += kTile) {
        const int nvalid = min(kTile, k_end - base);
        // Phase 1: scores. Warp w scores keys w, w + 8, ... of the tile; each lane dequantizes
        // its 4 dims of each key in registers against the key's stored fp16 scale. This loads
        // key j+8's raw words before key j's dot chain (the headroom the q move freed); the
        // arithmetic of both paths is the same multiply-then-fmaf sequence over the same bits.

        int kc_next[NCH] = {};
        unsigned short ks_next[NCH] = {};
        {
            const int j = warp;
            if (j < nvalid) {
                const int row = cache_cell(a, first, base + j);
                const signed char *kc =
                    a.k_codes + (size_t)kvh * code_stride + (size_t)row * D;
                const unsigned short *ks =
                    a.k_scales + (size_t)kvh * scale_stride + (size_t)row * groups;
#pragma unroll
                for (int c = 0; c < NCH; ++c) {
                    kc_next[c] = *reinterpret_cast<const int *>(kc + 128 * c + 4 * lane);
                    ks_next[c] = ks[4 * c + (lane >> 3)];
                }
            }
        }
#pragma unroll 1
        for (int i = 0; i < kTile / 8; ++i) {
            const int j = warp + 8 * i;
            if (j < nvalid) {
                const bool more = j + 8 < nvalid;
                int kc_hold[NCH];
                unsigned short ks_hold[NCH];
                if (more) {
                    const int row = cache_cell(a, first, base + j + 8);
                    const signed char *kc =
                        a.k_codes + (size_t)kvh * code_stride + (size_t)row * D;
                    const unsigned short *ks =
                        a.k_scales + (size_t)kvh * scale_stride + (size_t)row * groups;
#pragma unroll
                    for (int c = 0; c < NCH; ++c) {
                        kc_hold[c] = *reinterpret_cast<const int *>(kc + 128 * c + 4 * lane);
                        ks_hold[c] = ks[4 * c + (lane >> 3)];
                    }
                }
                float kf[8];
#pragma unroll
                for (int c = 0; c < NCH; ++c) {
                    const float sv = h2f(ks_next[c]);
                    const int word = kc_next[c];
                    kf[4 * c] = sv * (float)(signed char)(word & 0xff);
                    kf[4 * c + 1] = sv * (float)(signed char)((word >> 8) & 0xff);
                    kf[4 * c + 2] = sv * (float)(signed char)((word >> 16) & 0xff);
                    kf[4 * c + 3] = sv * (float)(signed char)((word >> 24) & 0xff);
                }
                if (more) {
#pragma unroll
                    for (int c = 0; c < NCH; ++c) {
                        kc_next[c] = kc_hold[c];
                        ks_next[c] = ks_hold[c];
                    }
                }
                float dots[8];
#pragma unroll
                for (int g = 0; g < G; ++g) {
                    float p = s_q[g][0][lane][0] * kf[0];
                    p = fmaf(s_q[g][0][lane][1], kf[1], p);
                    p = fmaf(s_q[g][0][lane][2], kf[2], p);
                    p = fmaf(s_q[g][0][lane][3], kf[3], p);
#pragma unroll
                    for (int c = 1; c < NCH; ++c) {
                        p = fmaf(s_q[g][c][lane][0], kf[4 * c], p);
                        p = fmaf(s_q[g][c][lane][1], kf[4 * c + 1], p);
                        p = fmaf(s_q[g][c][lane][2], kf[4 * c + 2], p);
                        p = fmaf(s_q[g][c][lane][3], kf[4 * c + 3], p);
                    }
                    dots[g] = p;
                }
                int head;
                const float sum = reduce8(dots, lane, head);
                if ((lane & 3) == 0) {
                    s_sc[head][j] = sum;
                }
            }
        }

        __syncthreads();
        // Phase 2: softmax statistics of head `warp` over the whole 64-key tile, in two 32-lane
        // passes with a fixed order: the exact max first, then the probabilities under it.
        {
            float m_new = m_run;
            for (int half = 0; half < kTile / 32; ++half) {
                const int j = half * 32 + lane;
                const float sc = j < nvalid ? s_sc[warp][j] : -INFINITY;
                m_new = fmaxf(m_new, warp_max_a(sc));
            }
            const float corr = expf(m_run - m_new);
            float psum = 0.f;
            for (int half = 0; half < kTile / 32; ++half) {
                const int j = half * 32 + lane;
                const float sc = j < nvalid ? s_sc[warp][j] : -INFINITY;
                const float p = j < nvalid ? expf(sc - m_new) : 0.f;
                psum += p;  // reduced below, in lane order within each half
                s_p[j][warp] = p;
            }
            l_run = l_run * corr + warp_sum_a(psum);
            m_run = m_new;
            if (lane == 0) {
                s_corr[warp] = corr;
            }
        }
        __syncthreads();
        // Phase 3: acc[g] (dimension tid of head g) += p[g][j] * V[j][tid], dequantized here.
#pragma unroll
        for (int g = 0; g < G; ++g) {
            acc[g] *= s_corr[g];
        }
        const signed char *vcol = a.v_codes + (size_t)kvh * code_stride + tid;
        const unsigned short *vrow =
            a.v_scales + (size_t)kvh * scale_stride + (tid / kQuant);
        // Eight constant-index values fit registers without changing the ascending-key FMA chain.
#pragma unroll 1
        for (int first_j = 0; first_j < kTile; first_j += 8) {
            float values[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int j = first_j + i;
                values[i] = 0.f;
                if (own && j < nvalid) {
                    const int row = cache_cell(a, first, base + j);
                    values[i] = h2f(vrow[(size_t)row * groups]) *
                                (float)vcol[(size_t)row * D];
                }
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int j = first_j + i;
                const float v = values[i];
                const float4 pa = *reinterpret_cast<const float4 *>(&s_p[j][0]);
                const float4 pb = *reinterpret_cast<const float4 *>(&s_p[j][4]);
                acc[0] = fmaf(pa.x, v, acc[0]); acc[1] = fmaf(pa.y, v, acc[1]);
                acc[2] = fmaf(pa.z, v, acc[2]); acc[3] = fmaf(pa.w, v, acc[3]);
                acc[4] = fmaf(pb.x, v, acc[4]); acc[5] = fmaf(pb.y, v, acc[5]);
                acc[6] = fmaf(pb.z, v, acc[6]); acc[7] = fmaf(pb.w, v, acc[7]);
            }
        }
    }
    // Partial of (token t, kv head, split s): all stats first, then all accumulators, as attn_decode.
    const size_t cell = ((size_t)t * a.n_kv_head + kvh) * a.max_splits + s;
    const size_t n_cells = (size_t)a.m * a.n_kv_head * a.max_splits;
    float *ml = a.part + cell * G * 2;
    float *ac = a.part + n_cells * G * 2 + cell * G * D;
    if (lane == 0) {
        ml[warp * 2] = m_run;
        ml[warp * 2 + 1] = l_run;
    }
    if (own) {
#pragma unroll
        for (int g = 0; g < G; ++g) {
            ac[(size_t)g * D + tid] = acc[g];
        }
    }
}

template <int D>
__global__ void __launch_bounds__(D) attn_q8_combine_k(Args a) {
    constexpr int G = kGroup;
    __shared__ float s_w[kMaxSplits], s_l[kMaxSplits], s_red[D / 32];
    const int h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    const int kvh = h / G, g = h % G;
    float *o = a.out + ((size_t)t * a.n_head + h) * D + d;
    const int kv_len = *a.kv_len;
    if (!kv_ok(a, kv_len)) {
        *o = __int_as_float(0x7fc00000);  // NaN: the caller's non-finite check ends the job
        return;
    }
    const int n_keys = kv_len + t;
    int first, count;
    visible(a, n_keys, first, count);
    int chunk, n_active;
    split_plan(count, a.max_splits, a.min_keys, chunk, n_active);
    const size_t n_cells = (size_t)a.m * a.n_kv_head * a.max_splits;
    const size_t cell0 = ((size_t)t * a.n_kv_head + kvh) * a.max_splits;
    const float *ml = a.part + (cell0 * G + g) * 2;
    const float *ac = a.part + n_cells * G * 2 + (cell0 * G + g) * D + d;
    const size_t ml_step = (size_t)G * 2, ac_step = (size_t)G * D;
    // Statistics: thread s < n_active reads split s; the block max is exact in any order.
    float ms = -INFINITY;
    if (d < n_active) {
        ms = ml[d * ml_step];
        s_l[d] = ml[d * ml_step + 1];
    }
    const float wm = warp_max_a(ms);
    if ((d & 31) == 0) {
        s_red[d >> 5] = wm;
    }
    __syncthreads();
    float M = s_red[0];
#pragma unroll
    for (int i = 1; i < D / 32; ++i) {
        M = fmaxf(M, s_red[i]);
    }
    if (a.sinks != nullptr) {
        M = fmaxf(M, a.sinks[h]);
    }
    if (d < n_active) {
        s_w[d] = expf(ms - M);
    }
    __syncthreads();
    // The zero-value sink enters the denominator once, not once per split; no QK scaling.
    float L = a.sinks == nullptr ? 0.f : expf(a.sinks[h] - M);
    for (int s = 0; s < n_active; ++s) {
        L = fmaf(s_l[s], s_w[s], L);
    }
    float A = 0.f;  // index order; the loads do not depend on the weights, so they overlap
#pragma unroll 16
    for (int s = 0; s < n_active; ++s) {
        A = fmaf(ac[s * ac_step], s_w[s], A);
    }
    const float sig = a.proj == NULL
                          ? 1.f
                          : 1.f / (1.f + expf(-a.proj[(size_t)t * a.ld + (size_t)h * 2 * D + D + d]));
    *o = (A / L) * sig;
}

bool aligned_to(const void *p, size_t n) { return ((uintptr_t)p % n) == 0; }

bool valid(const inference_cuda_attn_decode_q8_args *a) {
    if (a == NULL || a->q == NULL || a->k_codes == NULL || a->k_scales == NULL ||
        a->v_codes == NULL || a->v_scales == NULL || a->kv_len == NULL || a->part == NULL ||
        a->out == NULL) {
        return false;
    }
    const bool d_ok = (a->head_dim == 128 || a->head_dim == 256) && a->head_dim % kQuant == 0;
    const bool gate_ok = a->proj == NULL ? true : a->ld >= a->n_head * 2 * a->head_dim;
    // A ring needs a window, must hold the window plus the m - 1 later tokens of the batch, and
    // must fit the cells: cache_cell() returns a cell below ring. 64-bit sum: window may be INT_MAX.
    const bool ring_fits = a->ring <= a->cells;
    const bool ring_ok = a->window >= 0 && a->ring >= 0 &&
                         (a->ring == 0 || (a->window > 0 && ring_fits &&
                                           (long long)a->ring >= (long long)a->window + a->m - 1));
    const bool scale_ok = a->scale > 0.f && a->scale < 1e30f;
    return d_ok && gate_ok && ring_ok && a->m >= 1 && a->m <= kMaxM && a->n_kv_head >= 1 &&
           a->n_kv_head <= 64 && a->n_head == a->n_kv_head * kGroup && a->cells >= 1 &&
           a->max_splits >= 1 && a->max_splits <= kMaxSplits && a->min_keys >= kTile &&
           a->min_keys % kTile == 0 && a->min_keys <= (1 << 20) &&
           scale_ok && aligned_to(a->q, 16) && (a->proj == NULL || aligned_to(a->proj, 4)) &&
           aligned_to(a->k_codes, 4) && aligned_to(a->k_scales, 4) &&
           aligned_to(a->v_codes, 4) && aligned_to(a->v_scales, 4) && aligned_to(a->kv_len, 4) &&
           aligned_to(a->part, 16) && aligned_to(a->out, 4);
}

// Explicit arm control lives under INFERENCE_CUDA_ARMS_QUALIFICATION below (t_force_prefetch,
// set_arms): feature-disabled builds compile none of it.

bool register_v_enabled() {
    // Set INFERENCE_CUDA_KV8_ATTN_R1=0 before startup to restore the original kernel.
    static const bool enabled = [] {
        const char *value = std::getenv("INFERENCE_CUDA_KV8_ATTN_R1");
        return value == nullptr || (value[0] == '1' && value[1] == '\0');
    }();
    return enabled;
}

bool v_prefetch_enabled() {
    // INFERENCE_CUDA_KV8_ATTN_V_PREFETCH=1 (exactly "1") selects the prefetch kernel.
    // Read ONCE per process, like register_v_enabled: production pays one getenv at
    // startup, not one per launch; qualification builds toggle arms per launch through
    // the thread-local set_arms entry below.
    static const bool enabled = [] {
        const char *value = std::getenv("INFERENCE_CUDA_KV8_ATTN_V_PREFETCH");
        return value != nullptr && value[0] == '1' && value[1] == '\0';
    }();
    return enabled;
}

#ifdef INFERENCE_CUDA_ARMS_QUALIFICATION
// ---- Qualification-only machinery (cargo feature cuda-arms-qualification). Feature-disabled
// builds compile none of this: the receipt, the forced-arm override, the dispatch tag,
// and the test FFIs all live behind the define. ----

// Thread-local on purpose: only the thread that forces an arm sees the override, so a
// forced arm can never leak into another thread's launches (the device fixture
// additionally runs in a child process).
thread_local int t_force_prefetch = -1;

// Which kernel the LAST register-V launch dispatched: 0 none, 1 the legacy kernel,
// 2 the prefetch kernel. The fixture reads it after each leg, so a leg that silently
// resolved to the wrong kernel (or to the original non-register-V kernel) fails.
static std::atomic<int> g_dispatched{0};

// The prefetch kernel's launch receipt: an instantiation with local-memory spills or a
// 256-thread limit refuses to launch, loudly, printing regs/smem/spills once.
bool arm_receipt() {
    static const bool ok = [] {
        const void *kernel = reinterpret_cast<const void *>(attn_q8_register_v_prefetch_k<256>);
        cudaFuncAttributes attr{};
        const cudaError_t err = cudaFuncGetAttributes(&attr, kernel);
        int sms = -1;
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
        std::fprintf(stderr,
                     "attn_q8 register_v prefetch receipt: %s regs=%d smem=%d spill=%d "
                     "maxThreads=%d SMs=%d\n",
                     cudaGetErrorString(err), attr.numRegs,
                     (int)attr.sharedSizeBytes, (int)attr.localSizeBytes,
                     attr.maxThreadsPerBlock, sms);
        return err == cudaSuccess && attr.localSizeBytes == 0 && attr.maxThreadsPerBlock >= 256;
    }();
    if (!ok) {
        std::fprintf(stderr, "attn_q8 register_v prefetch receipt: REFUSING to launch\n");
    }
    return ok;
}
#endif

template <int D>
Failure enqueue(const Args &a, cudaStream_t s) {
    Failure failure;
    if constexpr (D == 256) {
        if (register_v_enabled()) {
            const bool pre =
#ifdef INFERENCE_CUDA_ARMS_QUALIFICATION
                t_force_prefetch >= 0 ? t_force_prefetch == 1 :
#endif
                v_prefetch_enabled();
            if (pre) {
#ifdef INFERENCE_CUDA_ARMS_QUALIFICATION
                if (!arm_receipt()) {
                    failure.record(cudaErrorLaunchOutOfResources, INFERENCE_CUDA_STAGE_LAUNCH);
                    return failure;
                }
#endif
                attn_q8_register_v_prefetch_k<D><<<dim3(a.max_splits, a.n_kv_head, a.m), 256, 0, s>>>(a);
#ifdef INFERENCE_CUDA_ARMS_QUALIFICATION
                g_dispatched.store(2, std::memory_order_relaxed);
#endif
                failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
                if (failure.ok()) {
                    attn_q8_combine_k<D><<<dim3(a.n_head, a.m), D, 0, s>>>(a);
                    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
                }
                return failure;
            }
            // The statement below is main's launch, verbatim (same indentation).
            attn_q8_register_v_part_k<D><<<dim3(a.max_splits, a.n_kv_head, a.m), 256, 0, s>>>(a);
#ifdef INFERENCE_CUDA_ARMS_QUALIFICATION
            g_dispatched.store(1, std::memory_order_relaxed);
#endif
        } else {
            attn_q8_part_k<D><<<dim3(a.max_splits, a.n_kv_head, a.m), 256, 0, s>>>(a);
        }
    } else {
        attn_q8_part_k<D><<<dim3(a.max_splits, a.n_kv_head, a.m), 256, 0, s>>>(a);
    }
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    if (failure.ok()) {
        attn_q8_combine_k<D><<<dim3(a.n_head, a.m), D, 0, s>>>(a);
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    }
    return failure;
}

}  // namespace

// Floats of `part` for the rule above (the layout of attn_decode); the Rust side pins the same
// numbers.
extern "C" size_t inference_cuda_attn_decode_q8_part_floats(int m, int n_kv_head, int head_dim,
                                                            int max_splits) {
    return (size_t)m * n_kv_head * max_splits * kGroup * (2 + head_dim);
}

static int launch_decode(const inference_cuda_attn_decode_q8_args *in,
    const float *sinks, void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (!valid(in) || (sinks != nullptr && !aligned_to(sinks, sizeof(float)))) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const Args a = {(const float *)in->q,
                        (const float *)in->proj,
                        (const signed char *)in->k_codes,
                        (const unsigned short *)in->k_scales,
                        (const signed char *)in->v_codes,
                        (const unsigned short *)in->v_scales,
                        (const int *)in->kv_len,
                        (float *)in->part,
                        (float *)in->out,
                        in->ld,
                        in->m,
                        in->n_head,
                        in->n_kv_head,
                        in->cells,
                        in->max_splits,
                        in->min_keys,
                        in->window,
                        in->ring,
                        in->scale, sinks};
        const cudaStream_t s = (cudaStream_t)stream;
        const Failure failure = in->head_dim == 128 ? enqueue<128>(a, s) : enqueue<256>(a, s);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}


extern "C" int inference_cuda_attn_decode_q8_launch(const inference_cuda_attn_decode_q8_args *in,
    void *stream, int *out_stage) {
    return launch_decode(in, nullptr, stream, out_stage);
}
#ifdef INFERENCE_CUDA_ARMS_QUALIFICATION
// Test/harness-only arm control, on THIS thread only (see t_force_prefetch).
extern "C" int inference_cuda_attn_decode_q8_set_arms(int prefetch) {
    if (prefetch < -1 || prefetch > 1) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    t_force_prefetch = prefetch;
    return 0;
}
#endif

#ifdef INFERENCE_CUDA_ARMS_QUALIFICATION
// The register count and static shared-memory bytes of the two register-V kernels: the
// fixture asserts they are DIFFERENT kernels before comparing their outputs.
extern "C" int inference_cuda_attn_decode_q8_arm_attrs(int prefetch, int *regs, int *smem) {
    if (regs == nullptr || smem == nullptr || (prefetch != 0 && prefetch != 1)) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    const void *kernel = prefetch
                             ? reinterpret_cast<const void *>(attn_q8_register_v_prefetch_k<256>)
                             : reinterpret_cast<const void *>(attn_q8_register_v_part_k<256>);
    cudaFuncAttributes attr{};
    const cudaError_t err = cudaFuncGetAttributes(&attr, kernel);
    if (err != cudaSuccess) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    *regs = attr.numRegs;
    *smem = (int)attr.sharedSizeBytes;
    return 0;
}

// Which kernel the last register-V launch dispatched (0 none, 1 legacy, 2 prefetch).
extern "C" int inference_cuda_attn_decode_q8_dispatched() {
    return g_dispatched.load(std::memory_order_relaxed);
}

// Reset/read the device-written kernel identity around each fixture leg.
extern "C" int inference_cuda_attn_decode_q8_kernel_identity_reset() {
    const int zero = 0;
    if (cudaMemcpyToSymbol(g_kernel_ran, &zero, sizeof(int)) != cudaSuccess) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    // The copy rides the default stream while fixture legs launch on the context
    // stream: order the reset against the following leg explicitly.
    return cudaDeviceSynchronize() == cudaSuccess ? 0 : INFERENCE_CUDA_INVALID_ARGUMENT;
}

extern "C" int inference_cuda_attn_decode_q8_kernel_identity(int *ran) {
    if (ran == nullptr) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    return cudaMemcpyFromSymbol(ran, g_kernel_ran, sizeof(int)) == cudaSuccess
               ? 0
               : INFERENCE_CUDA_INVALID_ARGUMENT;
}

// Device-side split-plan probe: calls the same __device__ helper the kernels call,
// so the fixture can pin BOTH copies, not just the host mirror.
__global__ void attn_q8_split_plan_probe_k(int n_keys, int max_splits, int min_keys,
                                           int *chunk, int *n_active) {
    int c, n;
    split_plan(n_keys, max_splits, min_keys, c, n);
    *chunk = c;
    *n_active = n;
}

// The device split_plan computes its sums in int: any intermediate wider than INT_MAX
// wraps on device, so BOTH the host mirror and the probe below reject those inputs
// before the arithmetic, instead of accepting them and computing the wrong plan.
static bool split_plan_fits(int n_keys, int max_splits, int min_keys) {
    const long long kMax = 2147483647LL;
    const long long want = ((long long)n_keys + min_keys - 1) / min_keys;
    const long long n_split = want < 1 ? 1 : (want > max_splits ? (long long)max_splits : want);
    const long long c0 = ((long long)n_keys + n_split - 1) / n_split;
    const long long c = (c0 + kTile - 1) / kTile * kTile;
    // Every bound is on the ADDITION, not on a + b - 1: C evaluates (a + b) - 1,
    // so a + b must itself fit or the device overflows before the subtraction.
    return (long long)n_keys + min_keys <= kMax       // want's sum
        && (long long)n_keys + n_split <= kMax        // chunk's sum
        && c0 + kTile <= kMax                         // the tile round-up
        && (long long)n_keys + c <= kMax;             // n_active's sum
}

extern "C" int inference_cuda_attn_decode_q8_split_plan_device(int n_keys, int max_splits,
                                                               int min_keys, int *chunk,
                                                               int *n_active) {
    if (chunk == nullptr || n_active == nullptr || n_keys < 1 || max_splits < 1
        || min_keys < 1 || !split_plan_fits(n_keys, max_splits, min_keys)) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    int *device_out = nullptr;
    if (cudaMalloc(&device_out, 2 * sizeof(int)) != cudaSuccess) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    attn_q8_split_plan_probe_k<<<1, 1>>>(n_keys, max_splits, min_keys, device_out,
                                         device_out + 1);
    const cudaError_t launch_err = cudaGetLastError();
    const cudaError_t sync_err = cudaDeviceSynchronize();
    int host_out[2] = {0, 0};
    const cudaError_t copy_err =
        cudaMemcpy(host_out, device_out, 2 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(device_out);
    if (launch_err != cudaSuccess || sync_err != cudaSuccess || copy_err != cudaSuccess) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    *chunk = host_out[0];
    *n_active = host_out[1];
    return 0;
}

extern "C" int inference_cuda_attn_decode_q8_split_plan(int n_keys, int max_splits, int min_keys,
                                                        int *chunk, int *n_active) {
    if (chunk == nullptr || n_active == nullptr || n_keys < 1 || max_splits < 1
        || min_keys < 1) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    // Test-only host mirror of the __device__ split_plan above (which cannot be called
    // from host code), so the fixture can assert the caps 64/61/41 give distinct
    // (chunk, n_active) plans at 65,536 keys before running any device leg. Degenerate
    // inputs (n_keys < 1, max_splits < 1, min_keys < 1) are rejected HERE: the device
    // helper divides by zero on them and stays byte-identical to base.
    // Widened arithmetic throughout, and split_plan_fits rejects inputs whose int
    // intermediates would overflow on device: the mirror must not accept a plan the
    // device copy computes wrongly. The pin test asserts exact values for the shipped
    // caps, so any drift between the two copies fails loudly.
    if (!split_plan_fits(n_keys, max_splits, min_keys)) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    const long long want = ((long long)n_keys + min_keys - 1) / min_keys;
    const long long n_split = want < 1 ? 1 : (want > max_splits ? (long long)max_splits : want);
    const long long c = ((((long long)n_keys + n_split - 1) / n_split) + kTile - 1) / kTile * kTile;
    const long long n = ((long long)n_keys + c - 1) / c;
    if (c > 2147483647LL || n > 2147483647LL || c < 1 || n < 1) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    *chunk = (int)c;
    *n_active = (int)n;
    return 0;
}
#endif

extern "C" int inference_cuda_attn_decode_q8_sinks_launch(const inference_cuda_attn_decode_q8_args *in,
    const float *sinks, void *stream, int *out_stage) {
    if (sinks == nullptr) {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    return launch_decode(in, sinks, stream, out_stage);
}
