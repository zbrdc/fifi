/*
 * RD-8b: decode attention of the resident Qwen3.6 decode step (decision 0047).
 *
 * Design guidance only, no code copied. Idea from MegaCapybara v1.35 static RE,
 * research/megacapybara-re-lessons.md section 6.1 C10 (split-KV decode attention with a
 * fixed-order combine and a tuned minimum number of keys per split); the technique is also public
 * (Flash-Decoding, Dao et al. 2023). The cache layout and the q/gate row layout are those of the
 * vendored attn_prep (the user's own xavier-qwen code, RD-4V). The code is original CUDA C for
 * sm_86 (sm_72 compiles), written from the shapes of this repository.
 *
 * What it computes, per token t of m and query head h (kv head h / G, G = n_head / n_kv_head):
 *   out[t][h][d] = sum_j softmax_j(scale * q[t][h] . K[j][kv]) * V[j][kv][d] * sigmoid(gate[t][h][d])
 * over the keys j < kv_len + t. K and V are fp16 and are read once per kv head for all G query
 * heads. q and the gate come from attn_prep (q after RMSNorm and rope; the gate is the second
 * half of each q row of the projection). Accumulation is f32.
 * Parameters beyond the Qwen3.6 configuration (head 256, gate, linear cache), kept so that a
 * second family needs no rewrite (MEL-A1): head_dim 128 or 256 (template), proj == NULL for no
 * output gate, and window / ring for sliding-window layers: with window W > 0 token t sees only
 * the last min(kv_len + t, W) positions (the key at the query's own position counts as the first of
 * them), and with ring R > 0 (window + m - 1 <= R <= cache_rows) position p lives in cell p % R.
 *
 * Two launches on one stream:
 *   attn_part_k    grid (max_splits, n_kv_head, m), 256 threads. A block owns one key range of one
 *                  kv head and the G query heads that share it. Tiles of 32 keys, online softmax;
 *                  one partial (max, sum, weighted V) per query head goes to `part`.
 *   attn_combine_k grid (n_head, m), 256 threads (one per head dimension). Merges the partials
 *                  of the active splits in index order, divides, applies the gate, writes out.
 * kv_len is read from device memory when the kernels run, so one captured graph serves every
 * position. The key ranges follow from kv_len, the token index t and window alone (split_plan), so
 * the two kernels agree and the grid never changes. There is no float atomic and every reduction has a fixed order.
 * A call whose kv_len is outside [1, cache_rows - (m - 1)] writes NaN to out.
 */
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#include "abi.h"
#include "resident/attn_decode.h"
#include "device.h"


namespace {

constexpr int kGroup = 8;        // query heads per kv head; the lane layout of reduce8 assumes 8
constexpr int kTile = 32;        // keys per tile; also the granule of a key range
constexpr int kMaxM = 16;
constexpr int kMaxSplits = 128;  // one thread per split in the combine statistics step

struct Args {
    const float *q;          // [m][n_head][D] f32
    const float *proj;       // NULL (no gate) or [m][ld] f32; head h's gate is at h * 2 * D + D
    const __half *k_cache;   // [cache_rows][n_kv_head][D] fp16
    const __half *v_cache;
    const int *kv_len;       // device word: keys visible to token 0; token t sees kv_len + t
    float *part;
    float *out;              // [m][n_head * D] f32
    int ld, m, n_head, n_kv_head, cache_rows, max_splits, min_keys;
    int window, ring;        // 0: all positions / linear rows; else last `window` positions / cell p % ring
    float scale;
};

// Key range plan from the number of keys alone. n_split splits of `chunk` keys (a multiple of the
// tile) cover n_keys; split s owns [s * chunk, min(n_keys, (s + 1) * chunk)). Only n_active
// splits hold keys; the other blocks of the grid exit.
__device__ inline void split_plan(int n_keys, int max_splits, int min_keys, int &chunk,
                                  int &n_active) {
    const int want = (n_keys + min_keys - 1) / min_keys;
    const int n_split = min(max(want, 1), max_splits);
    chunk = (((n_keys + n_split - 1) / n_split) + kTile - 1) / kTile * kTile;
    n_active = (n_keys + chunk - 1) / chunk;
}

// kv_len must leave room for the m tokens in a linear cache; a ring takes any positive length.
// Otherwise both kernels treat the call as invalid.
__device__ inline bool kv_ok(const Args &a, int kv_len) {
    return kv_len >= 1 && (a.ring > 0 ? kv_len < (1 << 30) : kv_len <= a.cache_rows - (a.m - 1));
}

// Visible keys of token t: the last min(n_keys, window) positions. first = position of key 0.
__device__ inline void visible(const Args &a, int n_keys, int &first, int &count) {
    count = a.window > 0 ? min(n_keys, a.window) : n_keys;
    first = n_keys - count;
}

// Cache row of the i-th visible key.
__device__ inline int cache_row(const Args &a, int first, int i) {
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

// Reduces eight per-lane partial sums (one per query head) over the 32 lanes with 9 shuffles
// instead of 40: each level halves the number of values a lane keeps and doubles the lanes that
// share a head. On return, lane l with (l & 3) == 0 holds the full sum of head
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

// 128 registers (launch bound 2 blocks per SM) measured faster than 155 with one block per SM.
template <int D>
__global__ void __launch_bounds__(256, 2) attn_part_k(Args a) {
    constexpr int G = kGroup, NCH = D / 128;  // NCH chunks of 4 dims per lane: 8 dims (D 256) or 4 (D 128)
    static_assert(D == 128 || D == 256, "head_dim 128 or 256");
    __shared__ float s_sc[G][kTile + 1];  // scores of the tile, head-major, padded
    __shared__ __align__(16) float s_p[kTile][G];  // read as float4 in phase 3; probabilities, key-major so a key's 8 are contiguous
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
    const size_t row_stride = (size_t)a.n_kv_head * D;

    // This lane's dims are 4 * lane .. +3, and for D 256 also 128 + 4 * lane .. +3. q is pre-scaled once.
    float4 qv[G][NCH];
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
        // V of this tile, one dimension per thread, issued first so its latency overlaps the scores.
        // With D 128 only threads 0..127 own a dimension; the others take part in the barriers only.
        const bool own = tid < D;
        unsigned short vr[kTile];
        const unsigned short *vcol =
            reinterpret_cast<const unsigned short *>(a.v_cache) + (size_t)kvh * D + tid;
#pragma unroll
        for (int j = 0; j < kTile; ++j) {
            vr[j] = own && j < nvalid ? vcol[(size_t)cache_row(a, first, base + j) * row_stride] : (unsigned short)0;
        }
        // Phase 1: scores. Warp w scores keys w, w + 8, w + 16, w + 24 of the tile.
#pragma unroll
        for (int i = 0; i < kTile / 8; ++i) {
            const int j = warp + 8 * i;
            if (j < nvalid) {
                const __half *kr = a.k_cache + (size_t)cache_row(a, first, base + j) * row_stride + (size_t)kvh * D;
                float kf[4 * NCH];
#pragma unroll
                for (int c = 0; c < NCH; ++c) {
                    const uint2 k4 = *reinterpret_cast<const uint2 *>(kr + 128 * c + 4 * lane);
                    kf[4 * c] = h2f(k4.x & 0xffffu); kf[4 * c + 1] = h2f(k4.x >> 16);
                    kf[4 * c + 2] = h2f(k4.y & 0xffffu); kf[4 * c + 3] = h2f(k4.y >> 16);
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
        // Phase 2: online softmax statistics, warp w for head w.
        {
            const float sc = lane < nvalid ? s_sc[warp][lane] : -INFINITY;
            const float m_new = fmaxf(m_run, warp_max_a(sc));
            const float p = lane < nvalid ? expf(sc - m_new) : 0.f;
            const float corr = expf(m_run - m_new);
            l_run = l_run * corr + warp_sum_a(p);
            m_run = m_new;
            s_p[lane][warp] = p;
            if (lane == 0) {
                s_corr[warp] = corr;
            }
        }
        __syncthreads();
        // Phase 3: acc[g] (dimension tid of head g) += p[g][j] * V[j][tid].
#pragma unroll
        for (int g = 0; g < G; ++g) {
            acc[g] *= s_corr[g];
        }
#pragma unroll
        for (int j = 0; j < kTile; ++j) {
            const float v = h2f(vr[j]);
            const float4 pa = *reinterpret_cast<const float4 *>(&s_p[j][0]);
            const float4 pb = *reinterpret_cast<const float4 *>(&s_p[j][4]);
            acc[0] = fmaf(pa.x, v, acc[0]); acc[1] = fmaf(pa.y, v, acc[1]);
            acc[2] = fmaf(pa.z, v, acc[2]); acc[3] = fmaf(pa.w, v, acc[3]);
            acc[4] = fmaf(pb.x, v, acc[4]); acc[5] = fmaf(pb.y, v, acc[5]);
            acc[6] = fmaf(pb.z, v, acc[6]); acc[7] = fmaf(pb.w, v, acc[7]);
        }
    }
    // Partial of (token t, kv head, split s): all stats first, then all accumulators.
    const size_t cell = ((size_t)t * a.n_kv_head + kvh) * a.max_splits + s;
    const size_t n_cells = (size_t)a.m * a.n_kv_head * a.max_splits;
    float *ml = a.part + cell * G * 2;
    float *ac = a.part + n_cells * G * 2 + cell * G * D;
    if (lane == 0) {
        ml[warp * 2] = m_run;
        ml[warp * 2 + 1] = l_run;
    }
    if (tid < D) {
#pragma unroll
        for (int g = 0; g < G; ++g) {
            ac[(size_t)g * D + tid] = acc[g];
        }
    }
}

// Element of the projection row that holds the gate of (token t, head h, dimension d).
template <int D>
__device__ inline size_t gate_index(const Args &a, int t, int h, int d) {
    const size_t row = (size_t)t * a.ld;
    return row + (size_t)h * 2 * D + D + d;
}

template <int D>
__global__ void __launch_bounds__(D) attn_combine_k(Args a) {
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
                          : 1.f / (1.f + expf(-a.proj[gate_index<D>(a, t, h, d)]));
    *o = (A / L) * sig;
}

bool aligned_to(const void *p, size_t n) { return ((uintptr_t)p % n) == 0; }

bool valid(const inference_cuda_attn_decode_args *a) {
    if (a == NULL || a->q == NULL || a->k_cache == NULL || a->v_cache == NULL ||
        a->kv_len == NULL || a->part == NULL || a->out == NULL) {
        return false;
    }
    const bool d_ok = a->head_dim == 128 || a->head_dim == 256;
    const bool gate_ok = a->proj == NULL ? true : a->ld >= a->n_head * 2 * a->head_dim;
    // A ring needs a window, must hold the window plus the m - 1 later tokens of the batch, and
    // must fit in the cache: cache_row() returns a row below ring. 64-bit sum: window may be INT_MAX.
    const bool ring_fits = a->ring <= a->cache_rows;
    const bool ring_ok = a->window >= 0 && a->ring >= 0 &&
                         (a->ring == 0 || (a->window > 0 && ring_fits &&
                                           (long long)a->ring >= (long long)a->window + a->m - 1));
    const bool scale_ok = a->scale > 0.f && a->scale < 1e30f;
    return d_ok && gate_ok && ring_ok && a->m >= 1 && a->m <= kMaxM && a->n_kv_head >= 1 &&
           a->n_kv_head <= 64 && a->n_head == a->n_kv_head * kGroup && a->cache_rows >= 1 &&
           a->max_splits >= 1 && a->max_splits <= kMaxSplits && a->min_keys >= kTile &&
           a->min_keys % kTile == 0 && a->min_keys <= (1 << 20) &&
           scale_ok && aligned_to(a->q, 16) && (a->proj == NULL || aligned_to(a->proj, 4)) &&
           aligned_to(a->k_cache, 16) && aligned_to(a->v_cache, 16) && aligned_to(a->kv_len, 4) &&
           aligned_to(a->part, 16) && aligned_to(a->out, 4);
}

template <int D>
Failure enqueue(const Args &a, cudaStream_t s) {
    Failure failure;
    attn_part_k<D><<<dim3(a.max_splits, a.n_kv_head, a.m), 256, 0, s>>>(a);
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    if (failure.ok()) {
        attn_combine_k<D><<<dim3(a.n_head, a.m), D, 0, s>>>(a);
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    }
    return failure;
}

}  // namespace

// Floats of `part` for the rule above; the Rust side pins the same numbers.
extern "C" size_t inference_cuda_attn_decode_part_floats(int m, int n_kv_head, int head_dim,
                                                         int max_splits) {
    return (size_t)m * n_kv_head * max_splits * kGroup * (2 + head_dim);
}

extern "C" int inference_cuda_attn_decode_launch(const inference_cuda_attn_decode_args *in,
                                                 void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (!valid(in)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const Args a = {(const float *)in->q, (const float *)in->proj,
                        (const __half *)in->k_cache, (const __half *)in->v_cache,
                        (const int *)in->kv_len, (float *)in->part, (float *)in->out,
                        in->ld, in->m, in->n_head, in->n_kv_head, in->cache_rows,
                        in->max_splits, in->min_keys, in->window, in->ring, in->scale};
        const cudaStream_t s = (cudaStream_t)stream;
        const Failure failure = in->head_dim == 128 ? enqueue<128>(a, s) : enqueue<256>(a, s);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
