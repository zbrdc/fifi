// Paged-only KV8 partial/combine. A separate translation unit preserves resident
// kernel register allocation; identity-page tests guard numerical drift.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"
#include "resident/attn_decode_q8.h"

namespace {

constexpr int kGroup = 8;   // query heads per kv head; the lane layout of reduce8 assumes 8
constexpr int kQuant = 32;  // values per fp16 scale: the frozen group of the KV8 format
constexpr int kTile = 64;   // keys per tile: the contiguous 64-cell run the kernel reads
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
};

// Key range plan from the number of keys alone, as attn_decode.cu: n_split splits of `chunk`
// keys (a multiple of the tile) cover n_keys; split s owns [s * chunk, min(n_keys, (s+1)*chunk)).
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
__global__ void __launch_bounds__(256) attn_q8_paged_part_k(Args a, int visible_keys, int output_group) {
    constexpr int G = kGroup, NCH = D / 128;  // chunks of 4 dims per lane: 2 (D 256) or 1 (D 128)
    static_assert(D == 128 || D == 256, "head_dim 128 or 256");
    __shared__ float s_sc[G][kTile + 1];      // scores of the tile, head-major, padded
    __shared__ __align__(16) float s_p[kTile][G];  // probabilities, key-major (float4 reads)
    __shared__ float s_corr[G];
    const int s = blockIdx.x, kvh = blockIdx.y, t = blockIdx.z;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int kv_len = visible_keys;
    if (!kv_ok(a, kv_len)) {
        return;
    }
    const int n_keys = kv_len + t;
    int first, count;
    visible(a, n_keys, first, count);
    int chunk, n_active;
    split_plan(count, 1, a.min_keys, chunk, n_active);
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
    const size_t cell = ((size_t)t * a.n_kv_head + kvh) * a.max_splits + output_group;
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
__global__ void __launch_bounds__(D) attn_q8_paged_combine_k(Args a, int visible_keys) {
    constexpr int G = kGroup;
    __shared__ float s_w[kMaxSplits], s_l[kMaxSplits], s_red[D / 32];
    const int h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    const int kvh = h / G, g = h % G;
    float *o = a.out + ((size_t)t * a.n_head + h) * D + d;
    const int kv_len = visible_keys;
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
    if (d < n_active) {
        s_w[d] = expf(ms - M);
    }
    __syncthreads();
    float L = 0.f;  // index order
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

Args arguments(const inference_cuda_attn_decode_q8_args *in) {
    return {(const float *)in->q,
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
                        in->scale};
}

bool paged_geometry(const inference_cuda_attn_decode_q8_args *a, int visible_keys) {
    return a != NULL && a->m == 1 && a->n_kv_head >= 1 && a->n_kv_head <= 64 &&
           a->n_head == kGroup * a->n_kv_head && (a->head_dim == 128 || a->head_dim == 256) &&
           a->max_splits == 64 && a->min_keys >= kTile && a->min_keys <= (1 << 20) &&
           a->min_keys % kTile == 0 && a->cells >= 1 && a->cells <= a->min_keys * 64 &&
           visible_keys >= 1 && visible_keys <= a->cells && a->window == 0 && a->ring == 0;
}

bool valid_partial(const inference_cuda_attn_decode_q8_args *a, int visible_keys, int group) {
    return paged_geometry(a, visible_keys) && a->cells == a->min_keys &&
           group >= 0 && group < 64 && a->scale > 0.f && a->scale < 1e30f &&
           a->q != NULL && aligned_to(a->q, 16) && a->part != NULL && aligned_to(a->part, 16) &&
           a->k_codes != NULL && aligned_to(a->k_codes, 4) &&
           a->k_scales != NULL && aligned_to(a->k_scales, 4) &&
           a->v_codes != NULL && aligned_to(a->v_codes, 4) &&
           a->v_scales != NULL && aligned_to(a->v_scales, 4);
}

bool valid_combine(const inference_cuda_attn_decode_q8_args *a, int visible_keys) {
    return paged_geometry(a, visible_keys) &&
           a->part != NULL && aligned_to(a->part, 16) &&
           a->out != NULL && aligned_to(a->out, 4) &&
           (a->proj == NULL || (aligned_to(a->proj, 4) &&
                               a->ld >= a->n_head * 2 * a->head_dim));
}


}  // namespace

// Selective page entries: unused pointers are allowed null and are never dereferenced.
extern "C" int inference_cuda_attn_decode_q8_partial_launch(
    const inference_cuda_attn_decode_q8_args *in, int visible_keys, int group,
    void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (!valid_partial(in, visible_keys, group)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const Args a = arguments(in);
        const cudaStream_t s = (cudaStream_t)stream;
        if (in->head_dim == 128) {
            attn_q8_paged_part_k<128><<<dim3(1, a.n_kv_head, 1), 256, 0, s>>>(a, visible_keys, group);
        } else {
            attn_q8_paged_part_k<256><<<dim3(1, a.n_kv_head, 1), 256, 0, s>>>(a, visible_keys, group);
        }
        Failure failure;
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_attn_decode_q8_combine_launch(
    const inference_cuda_attn_decode_q8_args *in, int visible_keys,
    void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (!valid_combine(in, visible_keys)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const Args a = arguments(in);
        const cudaStream_t s = (cudaStream_t)stream;
        if (in->head_dim == 128) {
            attn_q8_paged_combine_k<128><<<dim3(a.n_head, 1), 128, 0, s>>>(a, visible_keys);
        } else {
            attn_q8_paged_combine_k<256><<<dim3(a.n_head, 1), 256, 0, s>>>(a, visible_keys);
        }
        Failure failure;
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
